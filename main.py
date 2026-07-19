from hard_negatives.args.voyager_args import parse_args
from hard_negatives.mine_negatives import mine_negatives

def main():
    args = parse_args()
    mine_negatives(args)


if __name__ == "__main__":
    main()
