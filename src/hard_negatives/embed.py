from accelerate.utils import gather_object

def distributed_encode(
    texts,
    process_indices,
    *,
    model,
    accelerator,
    chunk_size,
    batch_size,
    label,
    on_main_process=None,
):
    local_dataset = [texts[i] for i in process_indices]
    num_chunks = (len(local_dataset) + chunk_size - 1) // chunk_size

    if accelerator.is_main_process:
        print(f"Processing {label} in {num_chunks} chunks...")

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(local_dataset))

        chunk_dataset = local_dataset[start_idx:end_idx]
        chunk_indices = process_indices[start_idx:end_idx]

        embeddings = model.encode(
            chunk_dataset,
            batch_size=batch_size,
            show_progress_bar=accelerator.is_main_process,
        )

        full_embeddings = gather_object(embeddings)
        full_indices = gather_object(chunk_indices)

        if accelerator.is_main_process and on_main_process is not None:
            on_main_process(full_embeddings, full_indices)

        del embeddings, full_embeddings, full_indices

        if accelerator.is_main_process:
            print(f"Processed {label} chunk {chunk_idx + 1}/{num_chunks}")

    accelerator.wait_for_everyone()