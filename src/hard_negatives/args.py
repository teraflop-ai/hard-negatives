import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine hard negatives from query-document pair parquet files, in stages."
    )
    parser.add_argument(
        "stage",
        choices=["prepare", "embed", "search", "nv_retriever"],
        help="prepare: dedup pairs + hashes (one node). embed: Slurm array, embeds its share of the pair files. "
        "search: Slurm array x GPUs, exact top-K negatives per pair. nv_retriever: training dataset (one node).",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="prepare stage: glob of the input (query, doc) parquet files",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="saved_dataset",
        help="Root for pairs/, emb/, scores/",
    )
    parser.add_argument("--query_column", type=str, default="query")
    parser.add_argument("--document_column", type=str, default="document")
    parser.add_argument(
        "--num_shards",
        type=int,
        default=256,
        help="prepare stage: number of output parquet files",
    )
    parser.add_argument(
        "--ray",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run Daft on a local Ray cluster",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        help="embed/search stages: override SLURM_ARRAY_TASK_COUNT",
    )
    parser.add_argument(
        "--copy_input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="embed stage: stage input files to /tmp first",
    )
    # embed
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument(
        "--query_instruction",
        type=str,
        default="Instruct: Given a natural language description, retrieve the code that implements it\nQuery:",
        help="Replaces the model's saved `query` prompt; documents use the saved `document` prompt",
    )
    parser.add_argument("--embedding_dim", type=int, default=2560)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_3")
    parser.add_argument(
        "--mini_batch", type=int, default=32, help="GPU batch inside encode()"
    )
    parser.add_argument(
        "--rows_per_call",
        type=int,
        default=4096,
        help="Rows per Daft UDF call (length-sorted inside)",
    )
    parser.add_argument(
        "--gpus", type=float, default=1, help="GPUs per Daft embedding actor"
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        default=int(os.environ.get("SLURM_GPUS_ON_NODE", 0)) or None,
        help="Number of embedding actors (= GPUs on the node); defaults to SLURM_GPUS_ON_NODE",
    )
    # search
    parser.add_argument(
        "--num_negatives", type=int, default=100, help="K candidates stored per pair"
    )
    parser.add_argument(
        "--query_chunk_size", type=int, default=1024, help="Queries per GEMM"
    )
    parser.add_argument(
        "--doc_chunk_size",
        type=int,
        default=1_000_000,
        help="Corpus rows resident on the GPU per pass",
    )
    # nv_retriever
    parser.add_argument(
        "--nvembed_threshold",
        type=float,
        default=0.95,
        help="Keep negatives with score < threshold * positive score",
    )
    parser.add_argument(
        "--max_num_negatives",
        type=int,
        default=4,
        help="Negatives per pair in the training dataset",
    )
    parser.add_argument("--prepared_dataset", type=str, default="./mined_dataset")
    parser.add_argument(
        "--upload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Push the training dataset to the hub",
    )
    parser.add_argument(
        "--path_to_hub_upload", type=str, default="TeraflopAI/mined_example"
    )
    return parser.parse_args()
