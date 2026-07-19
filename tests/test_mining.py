from hard_negatives.args import parse_args
from hard_negatives.nv_retriever import nv_retriever_dataset

args = parse_args()
dataset = nv_retriever_dataset(args)

print(dataset)
print(dataset[0])
