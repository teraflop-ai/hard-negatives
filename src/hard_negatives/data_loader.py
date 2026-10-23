import daft
from daft import DataType, col

from hard_negatives.runner import init_runner


def text_hash(expr):
    return (
        expr.hash().shift_right(1).cast(DataType.int64())
    )  # 63-bit xxhash: uint64 -> int64 without overflow nulls


def prepare_stage(args):
    """Stage 1 (one node): many small (query, doc) parquet files -> num_shards small (query, doc, query_hash, doc_hash)
    parquet files with exact duplicate pairs removed. Hashes are the ids used by every later stage."""
    init_runner(args.ray)
    (
        daft.read_parquet(args.input_path)
        .select(col(args.query_column), col(args.document_column))
        .distinct()
        .with_column("query_hash", text_hash(col(args.query_column)))
        .with_column("doc_hash", text_hash(col(args.document_column)))
        .repartition(args.num_shards)
        .write_parquet(f"{args.save_path}/pairs", write_mode="overwrite")
    )
