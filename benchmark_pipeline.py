#!/usr/bin/env python
"""Benchmark the DFlash pipeline stages against the new Cosmos DB (divdet-provisioned)."""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kg_builder import EmbedClient, load_config
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential
import openai

CONFIG_FILE = "my.yaml"
QUESTION = "I am searching for a high-calorie protein snack for long-distance running that fits in a running belt"


async def benchmark():
    cfg = load_config(CONFIG_FILE)
    cosmos_cfg = cfg["cosmos"]
    kg_cfg = cfg.get("kg", {})
    query_cfg = cfg.get("query", {})
    llm_cfg = cfg.get("llm", {})

    seed_k = int(query_cfg.get("seed_entities_k", 10))
    max_hops = int(query_cfg.get("max_hops", 1))
    max_triples = int(query_cfg.get("max_triples", 40))
    max_source = int(query_cfg.get("max_source_chunks", 15))
    vec_k = int(query_cfg.get("vector_augment_k", 12))

    cred = AzureCliCredential(tenant_id=cosmos_cfg["tenant_id"])
    client = CosmosClient(cosmos_cfg["uri"], credential=cred)
    db = client.get_database_client(cosmos_cfg["database_name"])
    entities_ctr = db.get_container_client(kg_cfg.get("entities_container", "entities"))
    triples_ctr = db.get_container_client(kg_cfg.get("triples_container", "triples"))
    food_ctr = db.get_container_client("food")

    embedder = EmbedClient(cfg)
    llm_client = openai.AsyncOpenAI(base_url=llm_cfg["endpoint"], api_key=llm_cfg.get("api_key", "dummy"))

    print("=" * 70)
    print(f"  DFlash Pipeline Benchmark")
    print(f"  Question: {QUESTION[:70]}...")
    print("=" * 70)

    # Warm up embedder
    _ = await embedder.embed("warmup")

    timings = {}

    # ── Stage 1: Embed ──
    t0 = time.perf_counter()
    q_emb = await embedder.embed(QUESTION)
    timings["Embed"] = time.perf_counter() - t0
    print(f"\n  1. Embed:           {timings['Embed']:.2f}s")

    # ── Stage 2: Entity Search (vector) ──
    t0 = time.perf_counter()
    seed_entities = []
    async for item in entities_ctr.query_items(
        query=("SELECT TOP @k c.n AS name, c.t AS description, c.r AS relation_count, c.d AS source_chunks, "
               "VectorDistance(c.e, @emb) AS score FROM c ORDER BY VectorDistance(c.e, @emb)"),
        parameters=[{"name": "@k", "value": seed_k}, {"name": "@emb", "value": q_emb}]):
        seed_entities.append(item)
    timings["Entity Search"] = time.perf_counter() - t0
    print(f"  2. Entity Search:   {timings['Entity Search']:.2f}s  ({len(seed_entities)} entities)")

    entity_names = [e["name"] for e in seed_entities[:10]]

    # ── Stage 3: Graph Traversal (PK hops + vector triples, parallel) ──
    t0 = time.perf_counter()

    async def _graph_traversal():
        all_t = []
        visited = set()
        names = list(entity_names)
        for hop in range(max_hops):
            batch = [n for n in names if n not in visited]
            if not batch:
                break
            for n in batch[:10]:
                visited.add(n)

            async def _fetch_pk(name):
                r = []
                async for triple in triples_ctr.query_items(
                    query="SELECT c.s AS subject, c.p AS predicate, c.o AS object, c.f AS confidence, c.d AS source_chunks FROM c WHERE c.s = @pk",
                    parameters=[{"name": "@pk", "value": name}]):
                    r.append(triple)
                return r

            results = await asyncio.gather(*[_fetch_pk(n) for n in batch[:10]])
            for r in results:
                all_t.extend(r)
            if hop == 0 and len(all_t) < max_triples:
                names = list({t["object"] for t in all_t if t["object"] not in visited})[:5]
        return all_t

    async def _triple_vec():
        r = []
        async for t in triples_ctr.query_items(
            query=("SELECT TOP @k c.s AS subject, c.p AS predicate, c.o AS object, c.f AS confidence, c.d AS source_chunks, "
                   "VectorDistance(c.e, @emb) AS score FROM c ORDER BY VectorDistance(c.e, @emb)"),
            parameters=[{"name": "@k", "value": 30}, {"name": "@emb", "value": q_emb}]):
            r.append(t)
        return r

    async def _food_vec():
        r = []
        async for doc in food_ctr.query_items(
            query=("SELECT TOP @k c.id, c.product_id, c.product_title_translated, c.brand, "
                   "c.claims_translated, c.ingredients_translated, c.pack_size_translated, "
                   "VectorDistance(c.e, @emb) AS score FROM c ORDER BY VectorDistance(c.e, @emb)"),
            parameters=[{"name": "@k", "value": vec_k}, {"name": "@emb", "value": q_emb}]):
            r.append(doc)
        return r

    pk_triples, vec_triples, vec_food = await asyncio.gather(
        _graph_traversal(), _triple_vec(), _food_vec()
    )

    all_triples_raw = pk_triples + vec_triples
    seen_keys = set()
    all_triples = []
    for t in all_triples_raw:
        key = f"{t.get('subject','')}|{t.get('predicate','')}|{t.get('object','')}"
        if key not in seen_keys:
            seen_keys.add(key)
            all_triples.append(t)
    all_triples = all_triples[:max_triples]
    timings["Graph Traversal"] = time.perf_counter() - t0
    print(f"  3. Graph Traversal: {timings['Graph Traversal']:.2f}s  ({len(pk_triples)} PK + {len(vec_triples)} vec = {len(all_triples)} unique)")

    # ── Stage 4: Source Fetch ──
    t0 = time.perf_counter()
    source_chunk_ids = set()
    for t in all_triples:
        for cid in t.get("source_chunks", []):
            source_chunk_ids.add(cid)
    for e in seed_entities[:5]:
        for cid in e.get("source_chunks", []):
            source_chunk_ids.add(cid)

    source_ids = list(source_chunk_ids)[:max_source]
    source_chunks = []
    if source_ids:
        for batch_start in range(0, len(source_ids), 20):
            batch = source_ids[batch_start:batch_start + 20]
            ids_param = ", ".join(f'"{sid}"' for sid in batch)
            async for doc in food_ctr.query_items(query=f"SELECT * FROM c WHERE c.id IN ({ids_param})"):
                for k in ("e", "_rid", "_self", "_etag", "_attachments", "_ts"):
                    doc.pop(k, None)
                source_chunks.append(doc)

    seen_ids = {doc.get("id") for doc in source_chunks}
    for doc in vec_food:
        if doc.get("id") not in seen_ids:
            source_chunks.append(doc)
            seen_ids.add(doc.get("id"))

    timings["Source Fetch"] = time.perf_counter() - t0
    print(f"  4. Source Fetch:    {timings['Source Fetch']:.2f}s  ({len(source_chunks)} docs)")

    # ── Stage 5: LLM (DFlash) ──
    graph_lines = []
    for e in seed_entities[:8]:
        graph_lines.append(f"Entity: {e['name']} (relations: {e.get('relation_count', 0)})")
    for t in all_triples[:30]:
        graph_lines.append(f"  {t['subject']} --[{t['predicate']}]--> {t['object']}")
    graph_context = "\n".join(graph_lines)

    source_lines = []
    for doc in source_chunks[:15]:
        parts = []
        for field in ["product_title_translated", "brand", "claims_translated",
                       "ingredients_translated", "pack_size_translated"]:
            val = doc.get(field)
            if val:
                parts.append(f"{field}: {val}")
        if parts:
            source_lines.append(f"[{doc.get('product_id','?')}] " + " | ".join(parts))
    source_text = "\n".join(source_lines)

    prompt = f"""Based on the following knowledge graph and source documents, answer the question.

KNOWLEDGE GRAPH:
{graph_context}

SOURCE DOCUMENTS:
{source_text}

QUESTION: {QUESTION}

Provide a detailed, expert answer with specific product recommendations."""

    t0 = time.perf_counter()
    resp = await llm_client.chat.completions.create(
        model=llm_cfg["model"],
        messages=[
            {"role": "system", "content": "You are a helpful food product expert. Always recommend products."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=int(llm_cfg.get("max_tokens", 1200)),
    )
    answer = resp.choices[0].message.content if resp.choices else ""
    timings["LLM"] = time.perf_counter() - t0
    print(f"  5. LLM (DFlash):   {timings['LLM']:.2f}s  ({len(answer)} chars)")

    timings["Total"] = sum(timings.values())

    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Stage':<20} {'Time':>8}")
    print(f"  {'-'*20} {'-'*8}")
    for stage, t in timings.items():
        print(f"  {stage:<20} {t:>7.2f}s")
    print(f"{'='*70}")

    print(f"\n  Answer preview ({len(answer)} chars):")
    print(f"  {answer[:300]}...")

    await client.close()
    await cred.close()
    return timings


if __name__ == "__main__":
    asyncio.run(benchmark())
