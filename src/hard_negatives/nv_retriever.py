import daft
from daft import col
from datasets import load_dataset

from hard_negatives.runner import init_runner


def nv_retriever_dataset(args):
    """Stage 4 (one node): NV-Retriever selection -> (anchor, positive, negative_1..N) parquet files.
    Keeps, per pair, the first N negatives with score < nvembed_threshold * positive_score; drops pairs with fewer."""
    init_runner(args.ray)
    N, threshold = args.max_num_negatives, args.nvembed_threshold
    scores = daft.read_parquet(f"{args.save_path}/scores/**/*.parquet")
    corpus = scores.select(col("doc_hash"), col(args.document_column)).distinct()

    @daft.func
    def select(
        positive_score: float, hashes: list[int], negative_scores: list[float]
    ) -> list[int]:
        return [
            h
            for h, s in zip(hashes, negative_scores)
            if h != -1 and s < threshold * positive_score
        ][:N]

    df = scores.with_column(
        "selected",
        select(col("positive_score"), col("negative_hashes"), col("negative_scores")),
    )
    df = df.where(col("selected").list_count() == N).select(
        col(args.query_column).alias("anchor"),
        col(args.document_column).alias("positive"),
        *[col("selected")[i].alias(f"hash_{i + 1}") for i in range(N)],
    )
    for i in range(1, N + 1):
        negatives = corpus.select(
            col("doc_hash").alias(f"hash_{i}"),
            col(args.document_column).alias(f"negative_{i}"),
        )
        df = df.join(negatives, on=f"hash_{i}").exclude(f"hash_{i}")
    df.select(
        "anchor", "positive", *[f"negative_{i}" for i in range(1, N + 1)]
    ).write_parquet(args.prepared_dataset, write_mode="overwrite")
    if args.upload:
        load_dataset(
            "parquet", data_files=f"{args.prepared_dataset}/*.parquet", split="train"
        ).push_to_hub(args.path_to_hub_upload)
