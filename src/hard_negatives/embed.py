import daft
import numpy as np
import pyarrow as pa
from daft import DataFrame, DataType, Series, col

from hard_negatives.prepare_model import load_model
from hard_negatives.runner import SlurmRunner


def embedding_factory(
    *,
    model_name: str,
    embedding_dim: int,
    query_instruction: str | None = None,
    max_seq_len: int = 2048,
    batch_size: int = 32,
    rows_per_call: int = 4096,
    attn_implementation: str = "flash_attention_3",
    gpus: int | float = 1,
    max_concurrency: int | None = None,
):
    embedding = DataType.embedding(DataType.float16(), embedding_dim)

    @daft.cls(gpus=gpus, max_concurrency=max_concurrency)
    class PairEmbedding:
        def __init__(self):
            self.model = None

        def encode(self, texts, prompt_name):
            return self.model.encode(
                texts,
                prompt_name=prompt_name,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float16)

        @daft.method.batch(
            return_dtype=DataType.struct(
                {"query_embedding": embedding, "doc_embedding": embedding}
            ),
            batch_size=rows_per_call,
        )
        def embed(self, query: Series, doc: Series):
            if self.model is None:
                self.model = load_model(
                    model_name,
                    max_seq_len,
                    query_instruction,
                    attn_implementation,
                    device="cuda" if gpus else None,
                )
            q, d = (
                self.encode(query.to_pylist(), "query"),
                self.encode(doc.to_pylist(), "document"),
            )
            fixed = lambda e: pa.FixedSizeListArray.from_arrays(
                pa.array(e.ravel()), embedding_dim
            )
            return pa.StructArray.from_arrays(
                [fixed(q), fixed(d)], ["query_embedding", "doc_embedding"]
            )

    return PairEmbedding().embed


class EmbedPairs:
    """Stage 2: (query, doc, hashes) parquet -> same rows + query_embedding, doc_embedding (fp16, L2-normalized).
    One model per GPU actor; each UDF call encodes rows_per_call queries and docs."""

    def __init__(self, query_column: str, document_column: str, **factory_kwargs):
        self.query_column, self.document_column = query_column, document_column
        self.embed = embedding_factory(**factory_kwargs)

    def __call__(self, df: DataFrame) -> DataFrame:
        df = df.with_column(
            "emb", self.embed(col(self.query_column), col(self.document_column))
        )
        return (
            df.with_column("query_embedding", col("emb")["query_embedding"])
            .with_column("doc_embedding", col("emb")["doc_embedding"])
            .exclude("emb")
        )


def embed_stage(args):
    job = SlurmRunner(
        f"{args.save_path}/pairs",
        f"{args.save_path}/emb",
        args.num_tasks,
        args.copy_input,
    )
    job.run(
        EmbedPairs(
            args.query_column,
            args.document_column,
            model_name=args.model_name,
            embedding_dim=args.embedding_dim,
            query_instruction=args.query_instruction,
            max_seq_len=args.max_seq_len,
            batch_size=args.mini_batch,
            rows_per_call=args.rows_per_call,
            attn_implementation=args.attn_implementation,
            gpus=args.gpus,
            max_concurrency=args.max_concurrency,
        ),
        use_ray=args.ray,
    )
