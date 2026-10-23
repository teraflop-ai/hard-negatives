from hard_negatives.args import parse_args
from hard_negatives.data_loader import prepare_stage
from hard_negatives.embed import embed_stage
from hard_negatives.mine_negatives import search_stage
from hard_negatives.nv_retriever import nv_retriever_dataset

STAGES = {
    "prepare": prepare_stage,
    "embed": embed_stage,
    "search": search_stage,
    "nv_retriever": nv_retriever_dataset,
}


def main():
    args = parse_args()
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
