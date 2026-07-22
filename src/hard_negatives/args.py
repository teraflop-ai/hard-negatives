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
        help="The input path to parquet files.",
    )
    parser.add_argument("--num_dim", type=int, default=768)
    parser.add_argument(
        "--mini_batch",
        type=int,
        default=50,
        help="Batch size for encoding (default: 50)",
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        default="./saved_dataset",
        help="Input dataset to use for filtering.",
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=10,
        help="Number of negative samples to keep per positive",
    )
    parser.add_argument(
        "--nvembed_threshold",
        type=float,
        default=0.95,
        help="Threshold for filtering negatives: keep negatives with sim < threshold * positive_sim",
    )
    parser.add_argument(
        "--max_num_negatives",
        type=int,
        default=4,
        help="Maximum number of negatives to keep after threshold filtering",
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
    parser.add_argument(
        "--query_column",
        type=str,
        default="query",
        help="The name of query column from the pair dataset.",
    )
    parser.add_argument(
        "--document_column",
        type=str,
        default="document",
        help="The name of the document column from the pair dataset.",
    )
    parser.add_argument(
        "--query_batch_size", type=int, default=1000, help="The batch to scan the index"
    )
    parser.add_argument(
        "--path_to_hub_upload", type=str, default="TeraflopAI/mined_example"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="saved_dataset",
        help="whether to save the queries, documents, and scores to local disk.",
    )
    parser.add_argument(
        "--upload",
        type=bool,
        default=False,
        help="whether to upload the queries, documents, and scores to Hugging Face hub.",
    )
    parser.add_argument(
        "--local",
        type=bool,
        default=True,
        help="whether to saved the mined dataset to disk.",
    )
    parser.add_argument(
        "--prepared_dataset",
        type=str,
        default="./mined_dataset",
        help="The saved mined dataset to use for training.",
    )
    parser.add_argument(
        "--report_output_dir",
        type=str,
        default="report_directory",
        help="Directory to save filtering analysis reports",
    )

    return parser.parse_args()
