## Installation
```bash
uv pip install hard-negatives
```

## Usage
Mines hard negatives from existing query document pair datasets.
```bash
# 1. prepare (one node, CPU): dedup pairs, add content hashes, write --num_shards small parquet files
sbatch examples/prepare.sbatch --input_path "/data/pairs/*.parquet" --query_column anchor --document_column positive --num_shards 1024

# 2. embed (Slurm array, one task per node, one Qwen3-Embedding-4B actor per GPU): query and doc embeddings per pair
sbatch examples/embed.sbatch --query_column anchor --document_column positive

# 3. search (Slurm array, one task per node, all GPUs): exact top-K negatives per pair, positives and duplicates masked
sbatch examples/search.sbatch --query_column anchor --document_column positive

# 4. training dataset (one node, CPU): anchor, positive, negative_1..N after the NV-Retriever 95% filter
sbatch examples/nv_retriever.sbatch --query_column anchor --document_column positive
```

## Contributing
```bash
uvx ruff format
uvx ruff check --select I --fix .
```

```bibtex
@software{shippole2026hardnegatives,
  title        = {Hard Negatives: A Python Library for Mining Hard Negative Examples},
  author       = {Shippole, Enrico and Chaffin, Antoine and Aarsen, Tom},
  year         = {2026},
  url          = {https://github.com/teraflop-ai/hard-negatives},
  version      = {1.0.0},
  note         = {Python package available on PyPI at \url{https://pypi.org/project/hard-negatives/}}
}
```