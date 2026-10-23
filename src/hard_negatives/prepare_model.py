import torch
from sentence_transformers import SentenceTransformer


def load_model(
    model_name: str,
    max_seq_len: int,
    query_instruction: str | None = None,
    attn_implementation: str = "flash_attention_3",
    device=None,
):
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={
            "attn_implementation": attn_implementation,
            "dtype": torch.bfloat16,
        },
    )
    model.max_seq_length = max_seq_len
    model.prompts.setdefault("document", "")
    if query_instruction is not None:
        model.prompts["query"] = query_instruction
    return model
