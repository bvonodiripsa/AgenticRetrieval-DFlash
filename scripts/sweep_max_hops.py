#!/usr/bin/env python
"""Sweep query.max_hops against the local index and report triple/entity yield vs latency.

Cosmos priced each extra hop at another wave of N partition-key queries, so
`max_hops` was pinned at 1 in `my.yaml`. Against the local CSR graph a hop is
an array slice, not a network call -- this checks whether raising it actually
buys more grounding triples, and at what (now much smaller) latency cost.

Retrieval-only, no LLM required -- runs standalone against `data/local_index`.

    python scripts/sweep_max_hops.py --config my.yaml --hops 1,2,3,4
"""
import argparse
import asyncio
import os
import statistics
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import LocalBackend, retrieve  # noqa: E402

QUESTIONS = [
    "I am searching for a high-calorie protein snack for long-distance running that fits in a running belt",
    "gluten free breakfast options with high fiber",
    "vegan protein powder for post-workout recovery",
]


async def _embed_all(cfg, device):
    from gi_builder import EmbedClient
    embedder = EmbedClient(cfg)
    if device == "cuda":
        import gi_builder
        model, _ = gi_builder._get_embed_model()
        if next(model.parameters()).device.type != "cuda":
            model.to("cuda")
    await embedder.embed("warmup")
    return [await embedder.embed(q) for q in QUESTIONS]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="my.yaml")
    ap.add_argument("--index", default="data/local_index")
    ap.add_argument("--hops", default="1,2,3,4")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    hop_values = [int(h) for h in args.hops.split(",")]

    print("=" * 88)
    print(f"  max_hops sweep -- {len(QUESTIONS)} questions x {args.runs} runs, hops={hop_values}")
    print("=" * 88)

    print("\nEmbedding questions...")
    q_embs = await _embed_all(cfg, args.device)

    from gi_index import LocalGraphIndex
    index = LocalGraphIndex(args.index, device=args.device)
    backend = LocalBackend(index, reverse_edges=bool(cfg.get("index", {}).get("reverse_edges", True)))

    print(f"\n{'hops':>5} {'pk_tot':>8} {'vec_tot':>8} {'uniq_tot':>9} {'entities_avg':>13} "
          f"{'graph_ms_mean':>14} {'graph_ms_max':>13}")

    baseline = None
    for hops in hop_values:
        run_cfg = {**cfg, "query": {**cfg.get("query", {}), "max_hops": hops}}
        pk_counts, vec_counts, uniq_counts, graph_times = [], [], [], []
        for q_emb in q_embs:
            for _ in range(args.runs):
                result = await retrieve(backend, q_emb, run_cfg)
                pk_counts.append(result.stats.get("pk_triples", 0))
                vec_counts.append(result.stats.get("vec_triples", 0))
                uniq_counts.append(len(result.triples))
                graph_times.append(result.timings.get("graph_traversal", 0.0) * 1000)

        row = {
            "pk_tot": sum(pk_counts) / len(pk_counts),
            "vec_tot": sum(vec_counts) / len(vec_counts),
            "uniq_tot": sum(uniq_counts) / len(uniq_counts),
            "graph_ms_mean": statistics.mean(graph_times),
            "graph_ms_max": max(graph_times),
        }
        print(f"{hops:>5} {row['pk_tot']:>8.1f} {row['vec_tot']:>8.1f} {row['uniq_tot']:>9.1f} "
              f"{'':>13} {row['graph_ms_mean']:>14.2f} {row['graph_ms_max']:>13.2f}")

        if baseline is None:
            baseline = row
        else:
            pk_gain = row["pk_tot"] - baseline["pk_tot"]
            print(f"      +{pk_gain:>6.1f} PK triples vs hops={hop_values[0]}, "
                  f"+{row['graph_ms_mean']-baseline['graph_ms_mean']:.3f}ms graph_traversal")

    print("\nNote: max_triples caps the final unique-triple count in my.yaml's query section, "
          "so uniq_tot may plateau even as pk_tot grows -- raise max_triples too if it does.")


if __name__ == "__main__":
    asyncio.run(main())
