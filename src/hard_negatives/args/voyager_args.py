import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine hard negatives from a Hugging Face dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="lightonai/DenseOn",
        help="The sentence transformer model to use.",
    )
    parser.add_argument(
        "--max_seq_len",
        default=8192,
        type=int,
        help="The seqlen to use during encoding.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="The input path to parquet files.",
    )
    parser.add_argument("--num_dim", type=int, default=768)
    parser.add_argument("--hnsw_m", type=int, default=64)
    parser.add_argument(
        "--hnsw_ef",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--mini_batch",
        type=int,
        default=50,
        help=f"Batch size for encoding (default: 50)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with limited data",
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=2048,
        help=f"Number of negative samples to keep per positive (default: 2048)",
    )
    parser.add_argument(
        "--nvembed_threshold",
        type=float,
        default=0.95,
        help=f"Threshold for filtering negatives: keep negatives with sim < threshold * positive_sim (default: 0.95)",
    )
    parser.add_argument(
        "--max_negatives_filter",
        type=int,
        default=10,
        help=f"Maximum number of negatives to keep after threshold filtering (default: 10)",
    )
    parser.add_argument(
        "--k_buffer",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--gather_chunk_size",
        type=int,
        default=2000,
    )
    parser.add_argument("--query_batch_size", type=int, default=1000)
    parser.add_argument(
        "--path_to_hub_upload", type=str, default="TeraflopAI/mined_example"
    )
    parser.add_argument(
        "--report_output_dir",
        type=str,
        default="report_directory",
        help="Directory to save filtering analysis reports",
    )

    return parser.parse_args()
