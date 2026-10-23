import os

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from accelerate import Accelerator

from hard_negatives.runner import parquet_files, shard_files, slurm_task


def read(file, columns):
    t = pq.read_table(file, columns=columns)
    emb = lambda c: torch.from_numpy(
        t[c].combine_chunks().flatten().to_numpy().reshape(len(t), -1)
    )
    ints = lambda c: torch.from_numpy(t[c].to_numpy())
    return {c: (emb(c) if c.endswith("_embedding") else ints(c)) for c in columns}


def corpus_chunks(files, rows):
    """Stream every document in the corpus: yields (doc_embedding, doc_hash) of ~rows rows at a time."""
    batch, n = [], 0
    for f in files + [None]:
        if f is not None:
            batch.append(read(f, ["doc_embedding", "doc_hash"]))
            n += len(batch[-1]["doc_hash"])
        if batch and (n >= rows or f is None):
            yield tuple(
                torch.cat([b[c] for b in batch]) for c in ("doc_embedding", "doc_hash")
            )
            batch, n = [], 0


def matches(sorted_keys, keys):
    """All positions in sorted_keys equal to each key: returns (index into keys, position in sorted_keys)."""
    lo, hi = (
        torch.searchsorted(sorted_keys, keys),
        torch.searchsorted(sorted_keys, keys, right=True),
    )
    counts = hi - lo
    which = torch.repeat_interleave(torch.arange(len(keys), device=keys.device), counts)
    offsets = torch.arange(
        int(counts.sum()), device=keys.device
    ) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    return which, torch.repeat_interleave(lo, counts) + offsets


def dedup_rows(scores, hashes):
    """Keep the best-scoring copy of each hash per row (-1 / -inf for the rest), rows re-sorted by score."""
    order = hashes.sort(
        dim=1, stable=True
    ).indices  # rows arrive sorted by score desc, so the first of equal hashes wins
    sorted_hashes = hashes.gather(1, order)
    dup = torch.zeros_like(hashes, dtype=torch.bool).scatter_(
        1, order[:, 1:], sorted_hashes[:, 1:] == sorted_hashes[:, :-1]
    )
    scores, idx = scores.masked_fill(dup, -torch.inf).sort(1, descending=True)
    return scores, hashes.masked_fill(dup, -1).gather(1, idx)


def search_stage(args):
    """Stage 3 (Slurm array x GPUs): each process takes a share of the embedded files (its queries), streams the whole
    corpus through the GPU, and writes those files back with top-K negatives referenced by doc_hash. Every positive
    of a query (all pairs sharing its query_hash) and every duplicate of those docs is masked."""
    acc = Accelerator()
    device = acc.device
    task_id, count = slurm_task()
    num_tasks = args.num_tasks or count
    K, QC, DC = args.num_negatives, args.query_chunk_size, args.doc_chunk_size
    corpus = parquet_files(f"{args.save_path}/emb")
    files = shard_files(corpus, task_id, num_tasks)[
        acc.process_index :: acc.num_processes
    ]
    if not files:
        return

    pairs = pq.read_table(
        f"{args.save_path}/pairs", columns=["query_hash", "doc_hash"]
    ).sort_by("query_hash")
    pair_q, pair_d = (
        torch.from_numpy(pairs["query_hash"].to_numpy()),
        torch.from_numpy(pairs["doc_hash"].to_numpy()),
    )
    mine = [read(f, ["query_embedding", "doc_embedding", "query_hash"]) for f in files]
    sizes = [len(m["query_hash"]) for m in mine]
    queries, q_hash = (
        torch.cat([m[c] for m in mine]) for c in ("query_embedding", "query_hash")
    )
    positive_scores = torch.cat(
        [
            (m["query_embedding"].float() * m["doc_embedding"].float()).sum(1)
            for m in mine
        ]
    )
    del mine

    best_scores = torch.full((len(queries), K), -torch.inf, device=device)
    best_hashes = torch.full((len(queries), K), -1, dtype=torch.long, device=device)
    for d, d_hash in corpus_chunks(corpus, DC):
        d, d_hash = d.to(device), d_hash.to(device)
        d_sorted, d_order = d_hash.sort()
        for a in range(0, len(queries), QC):
            q = queries[a : a + QC].to(device)
            local, positives = matches(
                pair_q, q_hash[a : a + QC]
            )  # every positive doc of each query in the chunk
            which, pos = matches(d_sorted, pair_d[positives].to(device))
            scores = q @ d.T
            scores[local.to(device)[which], d_order[pos]] = -torch.inf
            v, i = scores.topk(min(K, len(d)), 1)
            merged, j = torch.cat([best_scores[a : a + QC], v.float()], 1).topk(K, 1)
            best_scores[a : a + QC] = merged
            best_hashes[a : a + QC] = torch.cat(
                [best_hashes[a : a + QC], d_hash[i]], 1
            ).gather(1, j)
    best_scores, best_hashes = dedup_rows(best_scores.cpu(), best_hashes.cpu())

    fixed = lambda x: pa.FixedSizeListArray.from_arrays(
        pa.array(x.contiguous().numpy().ravel()), x.shape[1]
    )
    out_dir = f"{args.save_path}/scores/task_{task_id:05d}"
    os.makedirs(out_dir, exist_ok=True)
    start = 0
    for f, n in zip(files, sizes):
        t = pq.read_table(
            f,
            columns=[args.query_column, args.document_column, "query_hash", "doc_hash"],
        )
        rows = slice(start, start + n)
        t = t.append_column("positive_score", pa.array(positive_scores[rows].numpy()))
        t = t.append_column("negative_hashes", fixed(best_hashes[rows]))
        t = t.append_column("negative_scores", fixed(best_scores[rows]))
        pq.write_table(t, f"{out_dir}/{os.path.basename(f)}")
        start += n
