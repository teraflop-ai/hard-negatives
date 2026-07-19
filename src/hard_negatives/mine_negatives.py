from pathlib import Path

import sys
import numpy as np

from accelerate import Accelerator
from datasets import Dataset, Features, Value
from hard_negatives.data_loader import load_pair_dataset

from voyager import Index, Space

from hard_negatives.prepare_model import load_model
from hard_negatives.embed import distributed_encode
from hard_negatives.reporting import create_report
from hard_negatives.upload_dataset import upload_text_dataset


def mine_negatives(args):
    dataset_name = Path(args.input_path).stem

    queries, unique_index_to_doc_text, positives = load_pair_dataset(args.input_path)

    model = load_model(args.model_name, args.max_seq_len)

    accelerator = Accelerator()
    accelerator.prepare(model)

    # build distributed index for all unique documents
    unique_doc_indices = sorted(unique_index_to_doc_text.keys())
    process_indices = unique_doc_indices[
        accelerator.process_index :: accelerator.num_processes
    ]

    if accelerator.is_main_process:
        index = Index(
            Space.Cosine,
            num_dimensions=args.num_dim,
            M=args.hnsw_m,
            ef_construction=args.hnsw_ef,
        )
        doc_id_to_embedding = {}

        def handle_documents(embeddings, indices):
            index.add_items(embeddings, indices)
            doc_id_to_embedding.update(zip(indices, embeddings))
    else:
        handle_documents = None

    distributed_encode(
        unique_index_to_doc_text,
        process_indices,
        model=model,
        accelerator=accelerator,
        chunk_size=args.gather_chunk_size,
        batch_size=args.mini_batch,
        label="documents",
        on_main_process=handle_documents,
    )

    # Process queries in chunks
    process_indices = list(range(len(queries)))[
        accelerator.process_index :: accelerator.num_processes
    ]

    if accelerator.is_main_process:
        query_id_to_embedding = {}

        def handle_queries(embeddings, indices):
            query_id_to_embedding.update(zip(indices, embeddings))
    else:
        handle_queries = None

    distributed_encode(
        queries,
        process_indices,
        model=model,
        accelerator=accelerator,
        chunk_size=args.gather_chunk_size,
        batch_size=args.mini_batch,
        label="queries",
        on_main_process=handle_queries,
    )

    # exit non-main processes after ALL processing is complete
    if not accelerator.is_main_process:
        print(f"Rank {accelerator.process_index} finished all processing, exiting")
        sys.exit(0)

    if accelerator.is_main_process:
        # Prepare three datasets

        upload_text_dataset(
            unique_index_to_doc_text,
            id_column="document_id",
            text_column="document",
            config_name="documents",
            dataset_name=dataset_name,
            hub_path=args.path_to_hub_upload,
        )

        upload_text_dataset(
            queries,
            id_column="query_id",
            text_column="query",
            config_name="queries",
            dataset_name=dataset_name,
            hub_path=args.path_to_hub_upload,
        )

        # 3. Scores dataset (query_id, document_ids, scores)
        scores_rows = []

        # Batch query the index
        query_ids_list = list(positives.keys())
        num_query_batches = (
            len(query_ids_list) + args.query_batch_size - 1
        ) // args.query_batch_size

        all_query_results = {}  # Store results: query_id -> (indexes, distances)

        for batch_idx in range(num_query_batches):
            start_idx = batch_idx * args.query_batch_size
            end_idx = min((batch_idx + 1) * args.query_batch_size, len(query_ids_list))

            batch_query_ids = query_ids_list[start_idx:end_idx]
            query_embeddings_batch = [
                query_id_to_embedding[qid] for qid in batch_query_ids
            ]

            print(
                f"Querying index for batch {batch_idx + 1}/{num_query_batches} ({len(batch_query_ids)} queries)..."
            )
            # Query with extra buffer to account for potential duplicates and positives
            k_value = min(
                args.num_negatives + args.k_buffer, len(unique_index_to_doc_text)
            )
            batch_indexes, batch_distances = index.query(
                query_embeddings_batch, k=k_value
            )

            # Store results
            for i, qid in enumerate(batch_query_ids):
                all_query_results[qid] = (batch_indexes[i], batch_distances[i])

        print("Index querying complete")

        # Process results for each query
        for query_id in query_ids_list:
            indexes, distances = all_query_results[query_id]
            positives_list = positives[query_id]

            for positive in positives_list:
                document_ids = []
                scores_list = []

                # Calculate similarity for positive (first element)
                positive_simi = np.dot(
                    query_id_to_embedding[query_id], doc_id_to_embedding[positive]
                ) / (
                    np.linalg.norm(query_id_to_embedding[query_id])
                    * np.linalg.norm(doc_id_to_embedding[positive])
                )

                # Add positive as first element
                document_ids.append(positive)
                scores_list.append(float(positive_simi))

                # Add negatives with their distances converted to similarities
                negatives = []
                negative_scores = []
                for ind, distance in zip(indexes, distances):
                    if ind not in positives_list:
                        negatives.append(ind)
                        negative_scores.append(
                            1 - distance
                        )  # Convert distance to similarity
                        if len(negatives) >= args.num_negatives:
                            break

                # Only create row if we have enough negatives
                if len(negatives) < args.num_negatives:
                    continue

                # Add negative document IDs and scores
                document_ids.extend(negatives)
                scores_list.extend(negative_scores)

                scores_rows.append(
                    {
                        "query_id": query_id,
                        "document_ids": document_ids,
                        "scores": scores_list,
                    }
                )

        # Analyze filtering impact
        print("\n" + "=" * 50)
        print("Filtering Analysis")
        print("=" * 50)
        filtered_count = 0
        total_negatives_before = 0
        total_negatives_after = 0

        for row in scores_rows:
            positive_score = row["scores"][0]  # First score is the positive
            negative_scores = row["scores"][1:]  # Rest are negatives
            total_negatives_before += len(negative_scores)

            # Apply threshold filter
            threshold_value = args.nvembed_threshold * positive_score
            filtered_negatives = [
                score for score in negative_scores if score < threshold_value
            ]

            # Apply max negatives filter if specified
            filtered_negatives = filtered_negatives[: args.max_negatives_filter]

            total_negatives_after += len(filtered_negatives)

            # Count rows with at least max_negatives_filter negatives after filtering
            if len(filtered_negatives) >= args.max_negatives_filter:
                filtered_count += 1

        create_report(
            args,
            dataset_name,
            scores_rows,
            filtered_count,
        )

        scores_features = Features(
            {
                "query_id": Value("int64"),
                "document_ids": [Value("int64")],
                "scores": [Value("float64")],
            }
        )

        scores_dataset = Dataset.from_list(scores_rows, features=scores_features)
        scores_dataset.push_to_hub(
            args.path_to_hub_upload,
            config_name="scores",
            data_dir="scores",
            split=dataset_name,
        )
        print(f"Pushed scores dataset with {len(scores_rows)} query-document pairs")

        print(f"All three datasets pushed successfully for {dataset_name}")
