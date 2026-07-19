from hard_negatives.nv_retriever import nv_retriever_dataset

dataset = nv_retriever_dataset(
    input_path="/home/henry/hard_negatives/saved_dataset",
    num_negatives=5,
    nvembed_threshold=0.95,
    local=True,
    output_path="./mined_dataset",
)

print(dataset)
print(dataset[0])
