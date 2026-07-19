from datasets import load_dataset


def load_pair_dataset(path: str, query: str = "query", document: str = "document"):
    data = load_dataset("parquet", data_files=path, split="train")

    query_to_id = {}
    document_to_id = {}
    positives = {}

    for query, document in zip(data[query], data[document]):
        query_id = query_to_id.setdefault(query, len(query_to_id))
        document_id = document_to_id.setdefault(document, len(document_to_id))
        positives.setdefault(query_id, set()).add(document_id)

    queries = {query_id: query for query, query_id in query_to_id.items()}
    documents = {
        document_id: document for document, document_id in document_to_id.items()
    }
    positives = {
        query_id: sorted(document_ids) for query_id, document_ids in positives.items()
    }

    return queries, documents, positives
