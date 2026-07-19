from hard_negatives.args import parse_args
from hard_negatives.mine_negatives import mine_negatives
from hard_negatives.nv_retriever import nv_retriever_dataset


def main():
    args = parse_args()
    mine_negatives(args)
    nv_retriever_dataset(args)


if __name__ == "__main__":
    main()
