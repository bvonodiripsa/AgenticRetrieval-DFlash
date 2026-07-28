#!/usr/bin/env python
"""Online GI-RAG query engine for food products.

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

from prompts_gi_food import GRAPHRAG_ANSWER_PROMPT, FALLBACK_ANSWER_PROMPT
from gi_builder import EmbedClient, embed_sync, load_config
from retrieval import CosmosBackend, retrieve


def build_llm_call_kwargs(llm_cfg: dict, model: str) -> dict:
    """Return provider-specific kwargs for chat.completions.create.

    Chain-of-thought / reasoning is suppressed differently per model family:
      * Qwen models served by vLLM accept the `enable_thinking` chat-template
        kwarg, which turns their thinking mode off for fast, direct answers.
      * OpenAI-compatible reasoning models (e.g. GLM-5.2) don't take
        that flag. An optional `llm.reasoning.effort` setting is forwarded as
        `reasoning_effort` to trade reasoning depth for latency/cost when it is
        provided; otherwise the model uses its default reasoning behavior.
    """
    model_lower = (model or "").lower()
    reasoning_cfg = llm_cfg.get("reasoning") or {}
    effort = reasoning_cfg.get("effort")

    extra_body: dict[str, Any] = {}
    if "qwen" in model_lower:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if effort:
        extra_body["reasoning_effort"] = str(effort)

    return {"extra_body": extra_body} if extra_body else {}


# =============================================================================
# GI Query Engine
# =============================================================================

class GIQueryEngine:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._cosmos: CosmosClient | None = None
        self._cred: AzureCliCredential | None = None
        self._embedder = EmbedClient(cfg)

        llm_cfg = cfg.get("llm", {})
        # New upstream schema uses llm_endpoint / llm_model / llm_api_key; keep
        # the old endpoint / model / api_key names as fallbacks.
        self._llm = AsyncOpenAI(
            base_url=llm_cfg.get("llm_endpoint", llm_cfg.get("endpoint", "http://localhost:8000/v1")),
            api_key=(llm_cfg.get("llm_api_key") or llm_cfg.get("api_key")
                     or llm_cfg.get("azure_openai_key") or "dummy"),
            timeout=120.0,
            max_retries=int(llm_cfg.get("max_retries", 3)),
        )
        self._llm_model = llm_cfg.get("llm_model", llm_cfg.get("model", "Qwen/Qwen2.5-32B-Instruct"))
        # Answer token budget: project-specific query.max_answer_tokens first,
        # then upstream llm.max_completion_tokens, then a safe default.
        self._max_tokens = int(
            cfg.get("query", {}).get("max_answer_tokens")
            or llm_cfg.get("max_completion_tokens")
            or llm_cfg.get("max_tokens")
            or 1024
        )
        self._llm_call_kwargs = build_llm_call_kwargs(llm_cfg, self._llm_model)

        cosmos_cfg = cfg["cosmos"]
        self._db_name = cosmos_cfg["database_name"]
        self._gi_cfg = cfg.get("kg", {})
        self._triples_pk_field = self._gi_cfg.get("triples_partition_key_path", "/s").lstrip("/")
        self._query_cfg = cfg.get("query", {})
        self._backend = None

    async def _get_backend(self):
        """Retrieval backend selected by `index.mode` in the config.

        `local` serves the whole Graph Index from GPU memory; `cosmos` keeps
        the original remote queries. Both satisfy the same interface, so the
        rest of the pipeline is unaffected.
        """
        if self._backend is not None:
            return self._backend
        icfg = self._cfg.get("index", {})
        if str(icfg.get("mode", "cosmos")).lower() == "local":
            from gi_index import get_index
            from retrieval import LocalBackend
            index = get_index(
                icfg.get("snapshot_path", "data/local_index"),
                device=icfg.get("device", "cuda"),
                enable_bm25=bool(icfg.get("enable_bm25", True)),
            )
            self._backend = LocalBackend(index, reverse_edges=bool(icfg.get("reverse_edges", True)))
        else:
            cosmos = await self._get_cosmos()
            db = cosmos.get_database_client(self._db_name)
            self._backend = CosmosBackend(
                db.get_container_client(self._gi_cfg.get("entities_container", "entities")),
                db.get_container_client(self._gi_cfg.get("triples_container", "triples")),
                db.get_container_client("food"),
                self._triples_pk_field,
            )
        return self._backend

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
        ranker_http = getattr(self, "_ranker_http", None)
        if ranker_http is not None:
            await ranker_http.aclose()

    async def answer(self, question: str) -> dict[str, Any]:
        """Enhanced GI-RAG pipeline: embed -> entities -> graph -> vector augment -> LLM."""
        timings: dict[str, float] = {}
        t_total = time.time()

        # Step 1: Embed question
        t0 = time.time()
        q_emb = await self._embedder.embed(question)
        timings["embed"] = time.time() - t0

        # Steps 2-4: entity search, graph traversal and source fetch, against
        # whichever backend index.mode selects.
        result = await retrieve(await self._get_backend(), q_emb, self._cfg)
        timings.update(result.timings)
        seed_entities = result.seed_entities
        all_triples = result.triples
        source_chunks = result.source_chunks

        if not seed_entities:
            timings["total"] = time.time() - t_total
            return {
                "answer": "No relevant entities found in the graph index.",
                "entities": [],
                "triples": [],
                "timings": timings,
            }

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
            **self._llm_call_kwargs,
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
# CLI: run benchmark with GI query
# =============================================================================

async def run_benchmark(config_path: str, questions_path: str | None = None):
    """Run benchmark questions through GI query engine."""
    cfg = load_config(config_path)
    qfile = questions_path or cfg.get("paths", {}).get("questions_file", "data/food.json")

    with open(qfile) as f:
        questions = json.load(f)

    print(f"GI-RAG Benchmark: {len(questions)} questions")
    print(f"Config: {config_path}")
    print("=" * 60)

    engine = GIQueryEngine(cfg)

    # Warm up embedding model
    from gi_builder import embed_sync
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
            "mode": "gi-rag",
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
    out_dir = cfg.get("paths", {}).get("output_root", "out_gi")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    out_file = os.path.join(out_dir, f"gi_answers_{ts}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_file}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GI-RAG Query Engine for Food")
    parser.add_argument("--config", default="my.yaml")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--question", default=None, help="Single question to answer")
    args = parser.parse_args()

    if args.question:
        cfg = load_config(args.config)
        engine = GIQueryEngine(cfg)

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
