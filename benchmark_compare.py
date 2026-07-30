#!/usr/bin/env python
"""Compare Cosmos DB retrieval against the local GPU index, stage by stage.

Runs the same `retrieval.retrieve()` pipeline against both backends so the
comparison isolates the index, then reports overlap between the two result sets
as a correctness check. Local search is exact, so disagreement indicates a bug
(stale snapshot, normalisation mismatch, id misalignment) rather than a recall
tradeoff.

    python benchmark_compare.py --config my.yaml --runs 3
    python benchmark_compare.py --skip-cosmos          # local only, fast
"""
import argparse
import asyncio
import os
import statistics
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval import CosmosBackend, LocalBackend, retrieve  # noqa: E402

QUESTION = ("I am searching for a high-calorie protein snack for long-distance "
            "running that fits in a running belt")

STAGES = ("entity_search", "graph_traversal", "source_fetch")


def _mean(xs):
    return statistics.mean(xs) if xs else float("nan")


async def _embed(cfg, question, device):
    from gi_builder import EmbedClient
    embedder = EmbedClient(cfg)
    if device == "cuda":
        import gi_builder
        model, _ = gi_builder._get_embed_model()
        if next(model.parameters()).device.type != "cuda":
            model.to("cuda")
    await embedder.embed("warmup")
    t = time.perf_counter()
    emb = await embedder.embed(question)
    return emb, time.perf_counter() - t


async def _run_cosmos(cfg, q_emb, runs):
    from azure.cosmos.aio import CosmosClient
    from azure.identity.aio import AzureCliCredential

    ccfg = cfg["cosmos"]
    cred = AzureCliCredential(tenant_id=ccfg["tenant_id"])
    client = CosmosClient(ccfg["uri"], credential=cred)
    db = client.get_database_client(ccfg["database_name"])
    kg = cfg.get("kg", {})
    backend = CosmosBackend(
        db.get_container_client(kg.get("entities_container", "entities")),
        db.get_container_client(kg.get("triples_container", "triples")),
        db.get_container_client("food"),
    )
    out = []
    for _ in range(runs):
        out.append(await retrieve(backend, q_emb, cfg))
    await client.close()
    await cred.close()
    return out


async def _run_local(cfg, q_emb, runs, index, reverse):
    backend = LocalBackend(index, reverse_edges=reverse)
    await retrieve(backend, q_emb, cfg)  # warm CUDA
    return [await retrieve(backend, q_emb, cfg) for _ in range(runs)]


def _report(label, results, embed_s):
    t = {s: _mean([r.timings.get(s, 0) for r in results]) for s in STAGES}
    total = sum(t.values())
    r0 = results[0]
    print(f"\n  {label}")
    print(f"    embed              {embed_s:8.3f}s")
    for s in STAGES:
        print(f"    {s:18s} {t[s]:8.3f}s")
    print(f"    {'RETRIEVAL TOTAL':18s} {total:8.3f}s   "
          f"({r0.stats.get('pk_triples',0)} PK + {r0.stats.get('vec_triples',0)} vec "
          f"-> {len(r0.triples)} triples, {len(r0.source_chunks)} docs)")
    return total, t


def _overlap(a, b, key):
    sa = {x.get(key) for x in a}
    sb = {x.get(key) for x in b}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _dominance(local, cosmos, key):
    """Explain disagreement between exact and approximate search.

    Both report cosine similarity, higher is better. If every hit only local
    found scores above every hit only Cosmos found, then exact search strictly
    won and the gap is DiskANN's PQ compression losing neighbours — not a bug
    in the snapshot.
    """
    lm = {x.get(key): x.get("score", 0.0) for x in local}
    cm = {x.get(key): x.get("score", 0.0) for x in cosmos}
    only_l = [lm[k] for k in lm if k not in cm]
    only_c = [cm[k] for k in cm if k not in lm]
    if not only_l or not only_c:
        return "identical result sets"
    if min(only_l) >= max(only_c):
        return (f"exact search wins — {len(only_l)} hits Cosmos missed all score "
                f">= {min(only_l):.4f}, vs Cosmos-only max {max(only_c):.4f}")
    return (f"MIXED — local-only min {min(only_l):.4f} < Cosmos-only max "
            f"{max(only_c):.4f}; investigate the snapshot")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="my.yaml")
    ap.add_argument("--index", default="data/local_index")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--skip-cosmos", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    print("=" * 72)
    print(f"  Retrieval comparison — {args.runs} runs")
    print(f"  Question: {QUESTION[:60]}...")
    print("=" * 72)

    q_emb, embed_s = await _embed(cfg, QUESTION, args.device)
    print(f"\n  embedder on {args.device}: {embed_s*1000:.0f} ms")

    from gi_index import LocalGraphIndex
    index = LocalGraphIndex(args.index, device=args.device)

    local_fwd = await _run_local(cfg, q_emb, args.runs, index, reverse=False)
    local_rev = await _run_local(cfg, q_emb, args.runs, index, reverse=True)

    cosmos = None
    if not args.skip_cosmos:
        cosmos = await _run_cosmos(cfg, q_emb, args.runs)

    if cosmos:
        c_total, _ = _report("Cosmos DB (Sweden Central)", cosmos, embed_s)
    l_total, _ = _report("Local GPU index (forward edges — parity with Cosmos)", local_fwd, embed_s)
    r_total, _ = _report("Local GPU index (forward + reverse edges)", local_rev, embed_s)

    if cosmos:
        print("\n  " + "-" * 68)
        print(f"  Speedup (retrieval only):  {c_total / l_total:.0f}x")
        print(f"  End-to-end incl. embed:    {(c_total+embed_s) / (l_total+embed_s):.0f}x")
        ce, le = cosmos[0].seed_entities, local_fwd[0].seed_entities
        print("\n  Agreement with Cosmos:")
        print(f"    seed entities  Jaccard {_overlap(ce, le, 'name'):.3f}")
        print(f"    source docs    Jaccard {_overlap(cosmos[0].source_chunks, local_fwd[0].source_chunks, 'id'):.3f}")
        print(f"    entity quality {_dominance(le, ce, 'name')}")


if __name__ == "__main__":
    asyncio.run(main())
