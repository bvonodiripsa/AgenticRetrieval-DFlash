#!/usr/bin/env python
"""Online KG-RAG query engine for food products.

Given a question:
  1. Embed the question
  2. Vector-search the entity index for seed entities
  3. Fetch connected triples (graph traversal)
  4. Fetch source docs for provenance
  5. Single LLM call with structured graph context + source text

Target: 2-4 seconds per question.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import yaml
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential
from openai import AsyncOpenAI

from prompts_kg_food import GRAPHRAG_ANSWER_PROMPT, FALLBACK_ANSWER_PROMPT
from kg_builder import EmbedClient, embed_sync, load_config


# =============================================================================
# KG Query Engine
# =============================================================================

class KGQueryEngine:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._cosmos: CosmosClient | None = None
        self._cred: AzureCliCredential | None = None
        self._embedder = EmbedClient(cfg)

        llm_cfg = cfg.get("llm", {})
        self._llm = AsyncOpenAI(
            base_url=llm_cfg.get("endpoint", "http://localhost:8000/v1"),
            api_key=llm_cfg.get("api_key", "dummy"),
            timeout=120.0,
            max_retries=3,
        )
        self._llm_model = llm_cfg.get("model", "Qwen/Qwen2.5-32B-Instruct")
        self._max_tokens = int(cfg.get("query", {}).get("max_answer_tokens", 1024))

        cosmos_cfg = cfg["cosmos"]
        self._db_name = cosmos_cfg["database_name"]
        self._kg_cfg = cfg.get("kg", {})
        self._query_cfg = cfg.get("query", {})

    async def _get_cosmos(self) -> CosmosClient:
        if self._cosmos is None:
            cosmos_cfg = self._cfg["cosmos"]
            if cosmos_cfg.get("use_rbac_auth"):
                self._cred = AzureCliCredential(tenant_id=cosmos_cfg["tenant_id"])
                self._cosmos = CosmosClient(cosmos_cfg["uri"], credential=self._cred)
            else:
                self._cosmos = CosmosClient(cosmos_cfg["uri"], cosmos_cfg.get("key", ""))
        return self._cosmos

    async def close(self):
        if self._cosmos:
            await self._cosmos.close()
        if self._cred:
            await self._cred.close()

    async def answer(self, question: str) -> dict[str, Any]:
        """Enhanced KG-RAG pipeline: embed -> entities -> graph -> vector augment -> LLM."""
        timings: dict[str, float] = {}
        t_total = time.time()

        cosmos = await self._get_cosmos()
        db = cosmos.get_database_client(self._db_name)

        # Step 1: Embed question
        t0 = time.time()
        q_emb = await self._embedder.embed(question)
        timings["embed"] = time.time() - t0

        # Step 2: Find seed entities via vector search
        t0 = time.time()
        entities_container = db.get_container_client(
            self._kg_cfg.get("entities_container", "kg_entities_food")
        )
        seed_k = int(self._query_cfg.get("seed_entities_k", 20))

        sql = (
            "SELECT TOP @k c.name, c.description, c.relation_count, c.source_chunks, "
            "VectorDistance(c.embedding, @emb) AS score "
            "FROM c ORDER BY VectorDistance(c.embedding, @emb)"
        )
        seed_entities = []
        async for item in entities_container.query_items(
            query=sql,
            parameters=[
                {"name": "@k", "value": seed_k},
                {"name": "@emb", "value": q_emb},
            ],
        ):
            seed_entities.append(item)
        timings["entity_search"] = time.time() - t0

        if not seed_entities:
            timings["total"] = time.time() - t_total
            return {
                "answer": "No relevant entities found in the knowledge graph.",
                "entities": [],
                "triples": [],
                "timings": timings,
            }

        # Step 3: Graph traversal — fetch triples for seed entities (parallelized)
        t0 = time.time()
        triples_container = db.get_container_client(
            self._kg_cfg.get("triples_container", "kg_triples_food")
        )
        max_hops = int(self._query_cfg.get("max_hops", 2))
        max_triples = int(self._query_cfg.get("max_triples", 150))

        entity_names = [e["name"] for e in seed_entities[:10]]
        all_triples = []
        visited_entities: set[str] = set()

        async def _fetch_triples_for_entity(name: str):
            pk = name.lower()[:100]
            query = "SELECT * FROM c WHERE c.pk = @pk"
            results = []
            async for triple in triples_container.query_items(
                query=query,
                parameters=[{"name": "@pk", "value": pk}],
            ):
                results.append(triple)
            return results

        for hop in range(max_hops):
            if not entity_names:
                break
            batch_names = [n for n in entity_names if n not in visited_entities]
            if not batch_names:
                break

            for name in batch_names[:10]:
                visited_entities.add(name)

            tasks = [_fetch_triples_for_entity(n) for n in batch_names[:10]]
            results = await asyncio.gather(*tasks)
            for r in results:
                all_triples.extend(r)

            if hop == 0 and len(all_triples) < max_triples:
                entity_names = list({t["object"] for t in all_triples
                                    if t["object"] not in visited_entities})[:5]

        # Also do a vector search on triples for broader coverage
        triple_sql = (
            "SELECT TOP @k c.subject, c.predicate, c.object, c.confidence, c.source_chunks, "
            "VectorDistance(c.embedding, @emb) AS score "
            "FROM c ORDER BY VectorDistance(c.embedding, @emb)"
        )
        async for triple in triples_container.query_items(
            query=triple_sql,
            parameters=[
                {"name": "@k", "value": 30},
                {"name": "@emb", "value": q_emb},
            ],
        ):
            all_triples.append(triple)

        # Deduplicate triples
        seen_keys: set[str] = set()
        unique_triples = []
        for t in all_triples:
            key = f"{t.get('subject','')}|{t.get('predicate','')}|{t.get('object','')}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique_triples.append(t)
        all_triples = unique_triples[:max_triples]
        timings["graph_traversal"] = time.time() - t0

        # Step 4: Fetch source documents for provenance
        t0 = time.time()
        source_chunk_ids: set[str] = set()
        for t in all_triples:
            for cid in t.get("source_chunks", []):
                source_chunk_ids.add(cid)
        for e in seed_entities[:5]:
            for cid in e.get("source_chunks", []):
                source_chunk_ids.add(cid)

        max_source = int(self._query_cfg.get("max_source_chunks", 30))
        source_ids = list(source_chunk_ids)[:max_source]

        source_chunks = []
        food_container = db.get_container_client("food")

        # 4a: Fetch docs linked from KG triples/entities
        if source_ids:
            for batch_start in range(0, len(source_ids), 20):
                batch = source_ids[batch_start:batch_start + 20]
                ids_param = ", ".join(f'"{sid}"' for sid in batch)
                query = f"SELECT * FROM c WHERE c.id IN ({ids_param})"
                async for doc in food_container.query_items(
                    query=query,
                ):
                    doc.pop("e", None)
                    doc.pop("_rid", None)
                    doc.pop("_self", None)
                    doc.pop("_etag", None)
                    doc.pop("_attachments", None)
                    doc.pop("_ts", None)
                    source_chunks.append(doc)

        # 4b: Vector search augmentation — find additional relevant products directly
        vec_k = int(self._query_cfg.get("vector_augment_k", 15))
        seen_ids = {doc.get("id") for doc in source_chunks}
        vec_sql = (
            "SELECT TOP @k c.id, c.product_id, c.product_title_translated, c.brand, "
            "c.claims_translated, c.ingredients_translated, c.allergens_translated, "
            "c.pack_size_translated, c.product_title, c.claims, c.ingredients, c.allergens, c.pack_size, "
            "VectorDistance(c.e, @emb) AS score "
            "FROM c ORDER BY VectorDistance(c.e, @emb)"
        )
        async for doc in food_container.query_items(
            query=vec_sql,
            parameters=[
                {"name": "@k", "value": vec_k},
                {"name": "@emb", "value": q_emb},
            ],
        ):
            if doc.get("id") not in seen_ids:
                doc.pop("_rid", None)
                doc.pop("_self", None)
                doc.pop("_etag", None)
                doc.pop("_attachments", None)
                doc.pop("_ts", None)
                source_chunks.append(doc)
                seen_ids.add(doc.get("id"))

        timings["source_fetch"] = time.time() - t0

        # Step 5: Build prompt and call LLM
        t0 = time.time()
        graph_context = self._build_graph_context(seed_entities, all_triples)
        source_text = self._build_source_text(source_chunks)

        prompt = GRAPHRAG_ANSWER_PROMPT.replace("{graph_context}", graph_context) \
                                       .replace("{source_chunks}", source_text) \
                                       .replace("{question}", question)

        resp = await self._llm.chat.completions.create(
            model=self._llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=self._max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        answer = resp.choices[0].message.content or ""
        timings["llm"] = time.time() - t0

        timings["total"] = time.time() - t_total

        return {
            "answer": answer,
            "entities_found": len(seed_entities),
            "triples_found": len(all_triples),
            "source_docs": len(source_chunks),
            "timings": timings,
        }

    def _build_graph_context(self, entities: list[dict], triples: list[dict]) -> str:
        """Format graph data for LLM prompt."""
        lines = []
        lines.append("ENTITIES:")
        for e in entities[:10]:
            lines.append(f"  - {e['name']} ({e.get('relation_count', 0)} relations)")

        lines.append("\nPRODUCT FACTS:")
        for t in triples:
            conf = t.get("confidence", "")
            conf_str = f" [conf={conf}]" if conf else ""
            lines.append(f"  ({t.get('subject','')}) --[{t.get('predicate','')}]--> ({t.get('object','')}){conf_str}")

        return "\n".join(lines)

    def _build_source_text(self, source_chunks: list[dict]) -> str:
        """Format source documents for LLM prompt — rich detail for creative synthesis."""
        if not source_chunks:
            return "(No source documents available)"
        lines = []
        for doc in source_chunks[:30]:
            pid = doc.get("product_id", doc.get("id", "?"))
            title = doc.get("product_title_translated") or doc.get("product_title", "")
            brand = doc.get("brand", "")
            claims = doc.get("claims_translated") or doc.get("claims", [])
            ingredients = doc.get("ingredients_translated") or doc.get("ingredients", "")
            allergens = doc.get("allergens_translated") or doc.get("allergens", "")
            pack_size = doc.get("pack_size_translated") or doc.get("pack_size", "")
            prep = doc.get("preparation_translated") or doc.get("preparation", "")
            nutrition = doc.get("nutrition_translated") or doc.get("nutrition", "")
            price = doc.get("price", "")

            parts = [f"[product_id: {pid}] {title}"]
            if brand:
                parts.append(f"  Brand: {brand}")
            if claims:
                parts.append(f"  Claims: {', '.join(claims) if isinstance(claims, list) else claims}")
            if ingredients:
                ingr_str = ingredients if isinstance(ingredients, str) else ", ".join(ingredients)
                parts.append(f"  Ingredients: {ingr_str[:500]}")
            if allergens:
                parts.append(f"  Allergens: {allergens}")
            if pack_size:
                parts.append(f"  Pack size: {pack_size}")
            if prep:
                prep_str = prep if isinstance(prep, str) else str(prep)
                parts.append(f"  Preparation: {prep_str[:200]}")
            if nutrition:
                nutr_str = nutrition if isinstance(nutrition, str) else str(nutrition)
                parts.append(f"  Nutrition: {nutr_str[:200]}")
            if price:
                parts.append(f"  Price: {price}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines)


# =============================================================================
# CLI: run benchmark with KG query
# =============================================================================

async def run_benchmark(config_path: str, questions_path: str | None = None):
    """Run benchmark questions through KG query engine."""
    cfg = load_config(config_path)
    qfile = questions_path or cfg.get("paths", {}).get("questions_file", "data/food.json")

    with open(qfile) as f:
        questions = json.load(f)

    print(f"KG-RAG Benchmark: {len(questions)} questions")
    print(f"Config: {config_path}")
    print("=" * 60)

    engine = KGQueryEngine(cfg)

    # Warm up embedding model
    from kg_builder import embed_sync
    embed_sync("warmup")

    # Run all questions in parallel
    wall_start = time.time()

    async def _answer_one(i, q):
        q_text = q.get("question_text", "")
        q_id = q.get("question_id", f"q{i}")
        result = await engine.answer(q_text)
        return q_id, q_text, q, result

    tasks = [_answer_one(i, q) for i, q in enumerate(questions)]
    raw_results = await asyncio.gather(*tasks)
    wall_time = time.time() - wall_start

    results = []
    total_time = 0.0
    for q_id, q_text, q, result in raw_results:
        total_time += result["timings"]["total"]
        print(f"\n[{q_id}] {q_text[:70]}...")
        print(f"  Time: {result['timings']['total']:.2f}s "
              f"(embed={result['timings'].get('embed', 0):.2f}s, "
              f"entities={result['timings'].get('entity_search', 0):.2f}s, "
              f"graph={result['timings'].get('graph_traversal', 0):.2f}s, "
              f"source={result['timings'].get('source_fetch', 0):.2f}s, "
              f"llm={result['timings'].get('llm', 0):.2f}s)")
        print(f"  Found: {result.get('entities_found', 0)} entities, "
              f"{result.get('triples_found', 0)} triples, "
              f"{result.get('source_docs', 0)} source docs")
        print(f"  Answer: {result['answer'][:150]}...")

        results.append({
            "question_id": q_id,
            "question_text": q_text,
            "answer": result["answer"],
            "ground_truth": q.get("answer", ""),
            "llm_model": cfg["llm"]["model"],
            "embed_model": "Qwen/Qwen3-Embedding-0.6B",
            "mode": "kg-rag",
            "timings": result["timings"],
            "entities_found": result.get("entities_found", 0),
            "triples_found": result.get("triples_found", 0),
        })

    await engine.close()

    print("\n" + "=" * 60)
    print(f"WALL TIME: {wall_time:.1f}s for {len(questions)} questions (parallel)")
    print(f"SUM of per-question times: {total_time:.1f}s "
          f"(avg {total_time / len(questions):.1f}s/question)")
    print("=" * 60)

    # Save results
    out_dir = cfg.get("paths", {}).get("output_root", "out_kg")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out_file = os.path.join(out_dir, f"kg_answers_{ts}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_file}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="KG-RAG Query Engine for Food")
    parser.add_argument("--config", default="config_kg.yaml")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--question", default=None, help="Single question to answer")
    args = parser.parse_args()

    if args.question:
        cfg = load_config(args.config)
        engine = KGQueryEngine(cfg)

        async def _single():
            result = await engine.answer(args.question)
            print(f"\nAnswer: {result['answer']}")
            print(f"\nTimings: {json.dumps(result['timings'], indent=2)}")
            await engine.close()

        asyncio.run(_single())
    else:
        asyncio.run(run_benchmark(args.config, args.questions))


if __name__ == "__main__":
    main()
