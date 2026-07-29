#!/usr/bin/env python
"""Export the Cosmos-backed Graph Index to a local snapshot for in-process serving.

Pulls every document from the entities, triples and food containers in parallel
across physical partitions, then writes:

    <out>/<container>.vecs.npy    float16 (N, 1024), L2-normalised
    <out>/<container>.payload.parquet
    <out>/triples.csr.npz         forward and reverse adjacency
    <out>/manifest.json

Usage:
    python scripts/build_local_index.py --config my.yaml --out data/local_index
"""
import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIM = 1024

# Fields to pull per container. `e` is the embedding and is stripped from the
# payload after being packed into the vector array.
PROJECTIONS = {
    "entities": ["c.id", "c.n", "c.t", "c.r", "c.d", "c.e"],
    "triples": ["c.id", "c.s", "c.p", "c.o", "c.f", "c.d", "c.e"],
    "food": None,  # SELECT * — the pipeline returns whole documents as sources
}
FOOD_DROP = {"e", "_rid", "_self", "_etag", "_attachments", "_ts"}


async def _pull_range(container, feed_range, projection, progress):
    """Read every document in one physical partition."""
    cols = "*" if projection is None else ", ".join(projection)
    vecs, payloads = [], []
    async for doc in container.query_items(
        query=f"SELECT {cols} FROM c", feed_range=feed_range, max_item_count=1000
    ):
        emb = doc.pop("e", None)
        if emb is None or len(emb) != DIM:
            continue
        vecs.append(np.asarray(emb, dtype=np.float32))
        if projection is None:
            doc = {k: v for k, v in doc.items() if k not in FOOD_DROP}
        payloads.append(doc)
        progress[0] += 1
    return vecs, payloads


async def _report(progress, total, label, stop):
    t0 = time.perf_counter()
    while not stop.is_set():
        await asyncio.sleep(5)
        n, dt = progress[0], time.perf_counter() - t0
        rate = n / dt if dt else 0
        eta = (total - n) / rate / 60 if rate else 0
        pct = 100 * n / total if total else 0
        print(f"    {label}: {n:,}/{total:,} ({pct:5.1f}%)  {rate:6.0f} docs/s  ETA {eta:5.1f} min",
              flush=True)


async def export_container(db, name, out_dir):
    container = db.get_container_client(name)
    total = 0
    async for row in container.query_items(query="SELECT VALUE COUNT(1) FROM c"):
        total = row
    ranges = [fr async for fr in container.read_feed_ranges()]
    print(f"  {name}: {total:,} docs across {len(ranges)} partitions", flush=True)

    progress, stop = [0], asyncio.Event()
    reporter = asyncio.create_task(_report(progress, total, name, stop))
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *[_pull_range(container, fr, PROJECTIONS[name], progress) for fr in ranges]
    )
    stop.set()
    await asyncio.gather(reporter, return_exceptions=True)

    vecs = [v for chunk, _ in results for v in chunk]
    payloads = [p for _, chunk in results for p in chunk]
    dt = time.perf_counter() - t0
    print(f"  {name}: pulled {len(vecs):,} in {dt/60:.1f} min ({len(vecs)/dt:.0f} docs/s)", flush=True)

    arr = np.vstack(vecs).astype(np.float32)
    del vecs
    # Pre-normalise so cosine similarity is a plain dot product at query time.
    arr /= np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
    arr = arr.astype(np.float16)
    np.save(os.path.join(out_dir, f"{name}.vecs.npy"), arr)
    print(f"  {name}: wrote vectors {arr.shape} {arr.nbytes/1e9:.2f} GB", flush=True)

    # Normalise payload keys across shards so Arrow gets a stable schema.
    keys = sorted({k for p in payloads for k in p})
    table = pa.table({k: pa.array([p.get(k) for p in payloads]) for k in keys})
    pq.write_table(table, os.path.join(out_dir, f"{name}.payload.parquet"), compression="zstd")
    print(f"  {name}: wrote payload ({len(keys)} cols)", flush=True)
    return len(payloads)


def build_csr(out_dir):
    """Forward (subject -> rows) and reverse (object -> rows) adjacency.

    Cosmos can only serve the forward direction cheaply because `s` is the
    partition key; the reverse direction is what makes ingredient-level seed
    entities reachable.
    """
    table = pq.read_table(os.path.join(out_dir, "triples.payload.parquet"), columns=["s", "o"])
    subj = table.column("s").to_pylist()
    obj = table.column("o").to_pylist()

    vocab: dict[str, int] = {}
    def nid(name):
        if name not in vocab:
            vocab[name] = len(vocab)
        return vocab[name]

    s_ids = np.fromiter((nid(x or "") for x in subj), dtype=np.int32, count=len(subj))
    o_ids = np.fromiter((nid(x or "") for x in obj), dtype=np.int32, count=len(obj))

    def csr(ids):
        order = np.argsort(ids, kind="stable").astype(np.int32)
        counts = np.bincount(ids, minlength=len(vocab))
        indptr = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return indptr, order

    f_indptr, f_indices = csr(s_ids)
    r_indptr, r_indices = csr(o_ids)
    np.savez(
        os.path.join(out_dir, "triples.csr.npz"),
        fwd_indptr=f_indptr, fwd_indices=f_indices,
        rev_indptr=r_indptr, rev_indices=r_indices,
    )
    with open(os.path.join(out_dir, "triples.vocab.json"), "w") as fh:
        json.dump(vocab, fh)
    print(f"  csr: {len(vocab):,} distinct nodes, "
          f"{len(f_indices):,} forward + {len(r_indices):,} reverse edges", flush=True)


async def build(cosmos_cfg: dict, out_dir: str, containers: list[str] | None = None) -> dict:
    """Export `containers` (default entities,triples,food) to `out_dir` and return the manifest.

    Callable from a running server (e.g. the background auto-build path in
    `gi_query.py`), not just the CLI below — takes the already-parsed
    `cosmos:` config dict rather than a file path.
    """
    containers = containers or ["entities", "triples", "food"]
    os.makedirs(out_dir, exist_ok=True)

    cred = AzureCliCredential(tenant_id=cosmos_cfg["tenant_id"])
    client = CosmosClient(cosmos_cfg["uri"], credential=cred)
    db = client.get_database_client(cosmos_cfg["database_name"])

    print(f"Exporting {cosmos_cfg['uri']} -> {out_dir}", flush=True)
    t0 = time.perf_counter()
    counts = {}
    try:
        for name in containers:
            counts[name] = await export_container(db, name, out_dir)
    finally:
        await client.close()
        await cred.close()

    if "triples" in counts:
        build_csr(out_dir)

    manifest = {
        "source": cosmos_cfg["uri"],
        "database": cosmos_cfg["database_name"],
        "dim": DIM,
        "counts": counts,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nDone in {manifest['elapsed_min']:.1f} min -> {out_dir}", flush=True)
    return manifest


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="my.yaml")
    ap.add_argument("--out", default="data/local_index")
    ap.add_argument("--containers", default="entities,triples,food")
    args = ap.parse_args()

    cosmos_cfg = yaml.safe_load(open(args.config))["cosmos"]
    await build(cosmos_cfg, args.out, args.containers.split(","))


if __name__ == "__main__":
    asyncio.run(main())
