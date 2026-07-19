## Usage
Create and upload the mined hard negatives.
```bash
accelerate launch --multi_gpu main.py --input_path /home/henry/hard_negatives/parquets/test_pairs.parquet
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