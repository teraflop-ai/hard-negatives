import torch
from sentence_transformers import SentenceTransformer


def load_model(model_name: str, max_seq_len: int):
    model = SentenceTransformer(
        model_name,
        model_kwargs={
            "attn_implementation": "sdpa",
            "dtype": torch.bfloat16,
        },
    )
    model.max_seq_length = max_seq_len
    model = torch.compile(model)
    return model
