from pathlib import Path
from typing import Optional

from datasets import Dataset, Features, Value, load_dataset, load_from_disk


def nv_retriever_dataset(
    *,
    num_negatives: int,
    query_column: str = "query",
    document_column: str = "document",
    input_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    split: str = "train",
    local: bool = False,
    nvembed_threshold: Optional[float] = None,
    output_path: Optional[str] = None,
    output_repo_id: Optional[str] = None,
) -> Dataset:
    if num_negatives < 1:
        raise ValueError("num_negatives must be at least 1")

    if local:
        if input_path is None:
            raise ValueError("input_path is required when local=True")

        root = Path(input_path)
        documents_ds = load_from_disk(root / "documents")
        queries_ds = load_from_disk(root / "queries")
        scores_ds = load_from_disk(root / "scores")
    else:
        if repo_id is None:
            raise ValueError("repo_id is required when local=False")

        documents_ds = load_dataset(repo_id, "documents", split=split)
        queries_ds = load_dataset(repo_id, "queries", split=split)
        scores_ds = load_dataset(repo_id, "scores", split=split)

    documents = dict(
        zip(map(int, documents_ds["document_id"]), documents_ds[document_column])
    )
    queries = dict(zip(map(int, queries_ds["query_id"]), queries_ds[query_column]))

    columns = [
        "anchor",
        "positive",
        *(f"negative_{i}" for i in range(1, num_negatives + 1)),
    ]
    features = Features({column: Value("string") for column in columns})

    def rows():
        for row in scores_ds:
            query_id = int(row["query_id"])
            document_ids = list(map(int, row["document_ids"]))
            scores = list(map(float, row["scores"]))

            if not document_ids or len(document_ids) != len(scores):
                continue

            positive_id, positive_score = document_ids[0], scores[0]
            negatives = list(zip(document_ids[1:], scores[1:]))

            if nvembed_threshold is not None:
                cutoff = nvembed_threshold * positive_score
                negatives = [
                    (document_id, score)
                    for document_id, score in negatives
                    if score < cutoff
                ]

            negatives = negatives[:num_negatives]
            if len(negatives) < num_negatives:
                continue

            yield {
                "anchor": queries[query_id],
                "positive": documents[positive_id],
                **{
                    f"negative_{i}": documents[document_id]
                    for i, (document_id, _) in enumerate(negatives, start=1)
                },
            }

    dataset = Dataset.from_generator(rows, features=features)

    if output_path:
        dataset.save_to_disk(output_path)

    target_repo = output_repo_id or (repo_id if not local else None)
    if target_repo:
        dataset.push_to_hub(target_repo, split=split)

    return dataset
