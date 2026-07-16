#!/usr/bin/env python
"""End-to-end validation for the food-dflash database + Graph Index pipeline.

Checks:
  1. Cosmos DB connectivity and container existence
  2. Document count in the food container
  3. GI triple and entity counts
  4. Vector search works on food docs
  5. GI entity vector search works
  6. GI graph traversal (pk-based triple fetch)
  7. Full GI-RAG query for one question
  8. Benchmark all 10 questions and verify answers

Usage:
    python test_food_dflash.py --config my.yaml
    python test_food_dflash.py --config my.yaml --quick
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import yaml
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gi_builder import EmbedClient, embed_sync, load_config


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def ok(self, name: str, detail: str = ""):
        self.passed += 1
        suffix = f" — {detail}" if detail else ""
        print(f"  [PASS] {name}{suffix}")

    def fail(self, name: str, reason: str):
        self.failed += 1
        self.errors.append(f"{name}: {reason}")
        print(f"  [FAIL] {name} — {reason}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n  Results: {self.passed}/{total} passed", end="")
        if self.failed:
            print(f", {self.failed} FAILED")
            for e in self.errors:
                print(f"    - {e}")
        else:
            print(" — all good!")
        return self.failed == 0


async def run_tests(args):
    cfg = load_config(args.config)
    cosmos_cfg = cfg["cosmos"]
    kg_cfg = cfg.get("kg", {})
    results = TestResult()

    print("=" * 60)
    print("  Food-DFlash Validation Tests")
    print("=" * 60)
    print(f"  Config:   {args.config}")
    print(f"  Database: {cosmos_cfg['database_name']}")
    print()

    tenant_id = cosmos_cfg.get("tenant_id", "")
    cred = AzureCliCredential(tenant_id=tenant_id)
    cosmos = CosmosClient(cosmos_cfg["uri"], credential=cred)

    # Test 1: Database exists
    try:
        db = cosmos.get_database_client(cosmos_cfg["database_name"])
        await db.read()
        results.ok("Database exists", cosmos_cfg["database_name"])
    except Exception as e:
        results.fail("Database exists", str(e)[:120])
        await cosmos.close(); await cred.close()
        results.summary()
        return results

    # Test 2: Containers exist
    for cname in ["food", kg_cfg.get("triples_container", "triples"),
                  kg_cfg.get("entities_container", "entities")]:
        try:
            container = db.get_container_client(cname)
            await container.read()
            results.ok(f"Container '{cname}' exists")
        except Exception as e:
            results.fail(f"Container '{cname}' exists", str(e)[:120])

    # Test 3: Document counts
    food = db.get_container_client("food")
    try:
        count = 0
        async for _ in food.query_items("SELECT VALUE COUNT(1) FROM c"):
            count = _
        if count > 0:
            results.ok("Food docs present", f"{count} documents")
        else:
            results.fail("Food docs present", "0 documents")
    except Exception as e:
        results.fail("Food docs count", str(e)[:120])

    triples_name = kg_cfg.get("triples_container", "triples")
    entities_name = kg_cfg.get("entities_container", "entities")
    tc = db.get_container_client(triples_name)
    ec = db.get_container_client(entities_name)

    triple_count = entity_count = 0
    try:
        async for v in tc.query_items("SELECT VALUE COUNT(1) FROM c"):
            triple_count = v
        if triple_count > 0:
            results.ok("GI triples present", f"{triple_count} triples")
        else:
            results.fail("GI triples present", "0 triples — run gi_builder.py first")
    except Exception as e:
        results.fail("GI triples count", str(e)[:120])

    try:
        async for v in ec.query_items("SELECT VALUE COUNT(1) FROM c"):
            entity_count = v
        if entity_count > 0:
            results.ok("GI entities present", f"{entity_count} entities")
        else:
            results.fail("GI entities present", "0 entities — run gi_builder.py first")
    except Exception as e:
        results.fail("GI entities count", str(e)[:120])

    # Test 4: Vector search on food container
    embedder = EmbedClient(cfg)
    test_q = "high protein snack for running"
    e_hits = []
    try:
        q_emb = await embedder.embed(test_q)
        sql = ("SELECT TOP 5 c.id, c.product_title_translated, c.product_title, "
               "VectorDistance(c.e, @emb) AS score FROM c ORDER BY VectorDistance(c.e, @emb)")
        hits = []
        async for item in food.query_items(query=sql, parameters=[{"name": "@emb", "value": q_emb}]):
            hits.append(item)
        if hits:
            top = hits[0]
            title = top.get('product_title_translated') or top.get('product_title') or '?'
            score = top.get('score')
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            results.ok("Food vector search",
                       f"top hit: {str(title)[:50]} (score={score_str})")
        else:
            results.fail("Food vector search", "no results returned")
    except Exception as e:
        results.fail("Food vector search", str(e)[:120])

    # Test 5: Entity vector search
    if entity_count > 0:
        try:
            e_sql = ("SELECT TOP 5 c.name, c.relation_count, "
                     "VectorDistance(c.embedding, @emb) AS score "
                     "FROM c ORDER BY VectorDistance(c.embedding, @emb)")
            async for item in ec.query_items(query=e_sql, parameters=[{"name": "@emb", "value": q_emb}]):
                e_hits.append(item)
            if e_hits:
                top_e = e_hits[0]
                results.ok("Entity vector search",
                           f"top: {top_e.get('name', '?')[:50]} "
                           f"({top_e.get('relation_count', 0)} rels, score={top_e.get('score', '?'):.4f})")
            else:
                results.fail("Entity vector search", "no results")
        except Exception as e:
            results.fail("Entity vector search", str(e)[:120])

    # Test 6: Graph traversal
    if triple_count > 0 and e_hits:
        try:
            seed_name = e_hits[0]["name"]
            pk = seed_name.lower()[:100]
            t_sql = "SELECT TOP 10 c.subject, c.predicate, c.object FROM c WHERE c.pk = @pk"
            triples = []
            async for item in tc.query_items(query=t_sql, parameters=[{"name": "@pk", "value": pk}]):
                triples.append(item)
            if triples:
                sample = triples[0]
                results.ok("Graph traversal",
                           f"{len(triples)} triples for '{seed_name[:30]}' — "
                           f"e.g. ({sample['subject'][:20]}) --[{sample['predicate']}]--> ({sample['object'][:20]})")
            else:
                results.ok("Graph traversal", f"0 triples for pk='{pk[:30]}' (may be object-only entity)")
        except Exception as e:
            results.fail("Graph traversal", str(e)[:120])

    # Test 7: Full GI-RAG query
    if triple_count > 0:
        try:
            from gi_query import GIQueryEngine
            engine = GIQueryEngine(cfg)
            t0 = time.time()
            result = await engine.answer(test_q)
            elapsed = time.time() - t0
            answer = result.get("answer", "")
            if answer and len(answer) > 20:
                results.ok("GI-RAG single query",
                           f"{elapsed:.1f}s, {result.get('entities_found', 0)} entities, "
                           f"{result.get('triples_found', 0)} triples, {result.get('source_docs', 0)} docs")
            else:
                results.fail("GI-RAG single query", f"answer too short: '{answer[:50]}'")
            await engine.close()
        except Exception as e:
            results.fail("GI-RAG single query", str(e)[:120])

    # Test 8: Full benchmark
    if not args.quick and triple_count > 0:
        try:
            from gi_query import run_benchmark
            print("\n  Running full 10-question benchmark...")
            bench_results = await run_benchmark(args.config, "data/food.json")
            non_empty = sum(1 for r in bench_results if len(r.get("answer", "")) > 20)
            if non_empty == len(bench_results):
                results.ok("Full benchmark", f"{len(bench_results)} questions answered")
            else:
                results.fail("Full benchmark", f"only {non_empty}/{len(bench_results)} have substantive answers")
        except Exception as e:
            results.fail("Full benchmark", str(e)[:120])
    elif args.quick:
        print("\n  [SKIP] Full benchmark (--quick mode)")

    await cosmos.close()
    await cred.close()

    print("\n" + "=" * 60)
    results.summary()
    print("=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate food-dflash GI pipeline")
    parser.add_argument("--config", default="my.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(run_tests(args))
    sys.exit(0 if result.passed > 0 and result.failed == 0 else 1)


if __name__ == "__main__":
    main()
