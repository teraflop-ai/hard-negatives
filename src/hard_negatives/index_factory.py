from usearch.index import Index


def usearch_index(args):
    index = Index(
        ndim=args.num_dim,
        metric="cos",
        dtype="f32",
    )
    return index


def cuvs_ivf_index():
    pass


def build_index(args, index_type: str = "usearch"):
    if index_type == "usearch":
        return usearch_index(args=args)
    else:
        raise ValueError("Index not supported")
