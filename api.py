#!/usr/bin/env python
"""
KG-RAG API + Web UI — three food backends in one app.

Backends:
  1. Original AgenticRetrieval  (decomposed RAG, no KG — from github.com/AzureCosmosDB/AgenticRetrieval)
  2. KG version                 (KG-RAG, Qwen2.5-32B query settings)
  3. DFlash                     (KG-RAG + Qwen3.5-27B + DFlash speculative decoding)

All use the food database (58K products). LLM served by vLLM on localhost:8000.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault(
    "AZURE_COSMOS_SEMANTIC_RERANKER_INFERENCE_ENDPOINT",
    "https://divdet.westus3.dbinference.azure.com",
)

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kg_builder import load_config, embed_sync
from kg_query import KGQueryEngine

_ROOT = Path(__file__).parent
log = logging.getLogger("food_dflash.api")

BACKENDS = {
    "original": {
        "config": "config_original_local.yaml",
        "label": "Original AgenticRetrieval",
        "description": "Multi-round decomposed RAG — vector + full-text search, no KG (GPT-4.1)",
        "badge_color": "#7c3aed",
    },
    "kg": {
        "config": "config_kg_oldqwen.yaml",
        "label": "KG-RAG",
        "description": "Knowledge graph RAG — 892K triples, full context window",
        "badge_color": "#2563eb",
    },
    "dflash": {
        "label": "KG-RAG + DFlash",
        "description": "Knowledge graph RAG — optimized context, DFlash speculative decoding",
        "badge_color": "#059669",
    },
}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    backend: str = Field(default="dflash")


def _load_questions_from_cfg(cfg: dict) -> list[dict]:
    qpath = _ROOT / cfg.get("paths", {}).get("questions_file", "data/food.json")
    if qpath.exists():
        data = json.loads(qpath.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return data
    return []


def _load_questions_from_path(path: str) -> list[dict]:
    p = Path(path)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return data
    return []


# ---------------------------------------------------------------------------
# Original AgenticRetrieval pipeline wrapper
# ---------------------------------------------------------------------------

_original_pipeline = None
_original_llm = None


async def _init_original():
    """Import and initialize the original AgenticRetrieval decomposed pipeline.

    Uses config_original_local.yaml which points LLM at the local vLLM
    endpoint (Qwen3.5-27B) and uses in-process embeddings — no Azure
    OpenAI or Ollama dependency required.
    """
    global _original_pipeline, _original_llm

    orig_root = Path("/home/azureuser/AgenticRetrieval")
    if not orig_root.exists():
        log.warning("Original AgenticRetrieval not found at %s", orig_root)
        return False

    local_cfg_path = _ROOT / "config_original_local.yaml"
    if not local_cfg_path.exists():
        log.warning("config_original_local.yaml not found")
        return False

    if str(orig_root) not in sys.path:
        sys.path.insert(0, str(orig_root))

    import dynamic_retriever as dr
    dr.load_config(local_cfg_path)

    from utils.cosmos_retriever import CombinedRetriever, RETRIEVAL_SOURCES
    retriever = CombinedRetriever(
        retrieval_sources=RETRIEVAL_SOURCES,
        k_diverse=dr.CONFIG["retrieval"]["k_diverse"],
        k_ranker=0,
        eta=dr.CONFIG["retrieval"]["eta"],
        rescale_power=dr.CONFIG["retrieval"]["rescale_power"],
        cosmos_az_login=True,
    )
    await retriever.initialize()

    pipeline_cfg = dr.CONFIG.get("pipeline", {})
    _original_llm = dr.LLMClient(azure_az_login=True)
    _original_pipeline = dr.DecomposedRAGPipeline(
        retriever,
        _original_llm,
        max_sub_q=pipeline_cfg.get("max_sub_questions", 5),
        num_rounds=pipeline_cfg.get("rounds", 2),
        subq_fanout_cap=pipeline_cfg.get("subq_fanout_cap", 3),
        subq_max_concurrency=pipeline_cfg.get("subq_max_concurrency", 2),
    )
    log.info("Original AgenticRetrieval pipeline initialized (local vLLM)")
    return True


async def _run_original_stream(question: str):
    """Run original decomposed pipeline and stream SSE events."""
    t0 = time.perf_counter()

    if _original_pipeline is None:
        yield _sse({"stage": "error", "message": "Original pipeline not initialized"})
        yield "data: [DONE]\n\n"
        return

    yield _sse({"stage": "progress", "message": "Starting decomposed RAG pipeline...", "_ts": _elapsed(t0)})
    yield _sse({"stage": "progress", "message": "Round 1: initial retrieval + preliminary answer...", "_ts": _elapsed(t0)})

    try:
        result = await _original_pipeline.run(question)
        t_total = time.perf_counter() - t0

        yield _sse({
            "stage": "progress",
            "message": f"Pipeline complete in {t_total:.1f}s",
            "_ts": _elapsed(t0),
        })

        rounds = result.get("rounds", [])
        n_chunks = len(result.get("initial_chunks", []))
        n_subs = sum(len(r.get("sub_questions", [])) for r in rounds)
        yield _sse({
            "stage": "stats",
            "seed_entities": n_chunks,
            "triples_found": 0,
            "source_chunks": n_chunks,
            "entity_names": [f"{len(rounds)} rounds", f"{n_subs} sub-questions"],
            "_ts": _elapsed(t0),
        })

        answer = result.get("final_answer", "")
        for i in range(0, len(answer), 20):
            yield _sse({"stage": "token", "text": answer[i:i+20]})

        yield _sse({
            "stage": "done",
            "_ts": _elapsed(t0),
            "timings": {"total": t_total, "llm": t_total},
        })

    except Exception as e:
        log.exception("original pipeline error: %s", e)
        yield _sse({"stage": "error", "message": str(e)})

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# KG-RAG streaming (shared by kg and dflash backends)
# ---------------------------------------------------------------------------

DFLASH_ANSWER_PROMPT = """You are a food product expert.

DATA:
{graph_context}
{source_chunks}

QUESTION: {question}

Recommend 8-10 products. For each: name, (product_id: XXXXX), one sentence on why it fits. Never say "no products match." End with top pick."""


RERANKER_INFERENCE_ENDPOINT = os.environ.get(
    "AZURE_COSMOS_SEMANTIC_RERANKER_INFERENCE_ENDPOINT",
    "https://divdet.westus3.dbinference.azure.com",
)
RERANKER_FETCH_K = 40
RERANKER_TOP_K = 25

_STOP_WORDS = frozenset(
    "i me my we our you your he she it they them a an the this that these those "
    "is am are was were be been being have has had do does did will would shall should "
    "can could may might must need dare ought to of in on at by for with about against "
    "between through during before after above below from up down out off over under "
    "again further then once here there when where why how all both each few more most "
    "other some such no nor not only own same so than too very and but or if while "
    "because until just also already always never still even much really very "
    "what which who whom whose search searching looking find finding want need "
    "please help me tell give show recommend suggest".split()
)


def _extract_keywords(question: str) -> list[str]:
    """Extract meaningful content words from a question, stripping stop words."""
    import re
    words = re.findall(r"[a-zA-Z]+", question.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]


async def _llm_expand_keywords(question: str, engine) -> list[str]:
    """Use LLM to expand a question into additional food-related search terms."""
    try:
        resp = await engine._llm.chat.completions.create(
            model=engine._llm_model,
            messages=[
                {"role": "system", "content": "Extract food-related search keywords from the user question. "
                 "Return ONLY a comma-separated list of 5-8 single-word or two-word search terms "
                 "that would help find relevant food products in a database. Include ingredient names, "
                 "product types, and nutrition-related terms. No explanations."},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=80,
            **engine._llm_call_kwargs,
        )
        raw = resp.choices[0].message.content.strip()
        terms = [t.strip().lower() for t in raw.split(",") if t.strip()]
        return terms
    except Exception as e:
        log.warning("LLM keyword expansion failed: %s", e)
        return []


def _build_fulltext_sql(keyword: str) -> str:
    """Build a simple FullTextContains query for a single keyword across multiple fields."""
    return (
        "SELECT TOP @k c.id, c.product_id, c.product_title_translated, c.brand, "
        "c.claims_translated, c.ingredients_translated, c.allergens_translated, "
        "c.pack_size_translated, c.product_title, c.claims, c.ingredients, c.allergens, c.pack_size "
        "FROM c WHERE FullTextContains(c.product_title_translated, @kw) "
        "OR FullTextContains(c.ingredients_translated, @kw) "
        "OR FullTextContains(c.claims_translated, @kw)"
    )


async def _semantic_rerank(food_ctr, question: str, docs: list[dict], top_k: int = 10) -> list[dict]:
    """Rerank food documents using Cosmos DB Semantic Reranker. Falls back to original order on failure."""
    if not docs:
        return docs
    import json as _json
    doc_strings = []
    for doc in docs:
        parts = []
        title = doc.get("product_title_translated") or doc.get("product_title", "")
        if title:
            parts.append(title)
        brand = doc.get("brand", "")
        if brand:
            parts.append(f"Brand: {brand}")
        claims = doc.get("claims_translated") or doc.get("claims", "")
        if claims:
            parts.append(f"Claims: {', '.join(claims) if isinstance(claims, list) else claims}")
        ingredients = doc.get("ingredients_translated") or doc.get("ingredients", "")
        if ingredients:
            ingr = ingredients if isinstance(ingredients, str) else ", ".join(ingredients)
            parts.append(f"Ingredients: {ingr[:300]}")
        pack_size = doc.get("pack_size_translated") or doc.get("pack_size", "")
        if pack_size:
            parts.append(f"Pack size: {pack_size}")
        doc_strings.append(" | ".join(parts) if parts else _json.dumps(doc)[:500])

    try:
        result = await food_ctr.semantic_rerank(
            context=question,
            documents=doc_strings,
            options={"return_documents": False, "top_k": min(top_k, len(docs)), "sort": True},
        )
        scores = result.get("Scores", [])
        if scores:
            reranked = [docs[s["index"]] for s in scores if s["index"] < len(docs)]
            return reranked
    except Exception as e:
        log.warning("Semantic reranker failed (falling back to vector order): %s", e)

    return docs[:top_k]


async def _stream_dflash_sse(question: str, engine: KGQueryEngine):
    """DFlash path: full KG retrieval (same as KG) + non-streaming LLM with speculative decoding."""
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    try:
        yield _sse({"stage": "progress", "message": "Embedding question...", "_ts": _elapsed(t0)})

        t_embed = time.perf_counter()
        q_emb = await engine._embedder.embed(question)
        timings["embed"] = time.perf_counter() - t_embed

        yield _sse({"stage": "progress", "message": f"Embedded in {timings['embed']:.2f}s", "_ts": _elapsed(t0)})

        cosmos = await engine._get_cosmos()
        db = cosmos.get_database_client(engine._db_name)
        entities_ctr = db.get_container_client(engine._kg_cfg.get("entities_container", "kg_entities_food"))
        triples_ctr = db.get_container_client(engine._kg_cfg.get("triples_container", "kg_triples_food"))
        food_ctr = db.get_container_client("food")

        cfg = engine._query_cfg
        seed_k = int(cfg.get("seed_entities_k", 20))
        max_hops = int(cfg.get("max_hops", 2))
        max_triples = int(cfg.get("max_triples", 150))
        max_source = int(cfg.get("max_source_chunks", 30))
        vec_k = int(cfg.get("vector_augment_k", 15))

        # --- Step 1: Entity search + keyword expansion (parallel) ---
        yield _sse({"stage": "progress", "message": "Searching entity index + expanding keywords...", "_ts": _elapsed(t0)})

        async def _entity_search():
            r = []
            async for item in entities_ctr.query_items(
                query=("SELECT TOP @k c.n AS name, c.t AS description, c.r AS relation_count, c.d AS source_chunks, "
                       "VectorDistance(c.e, @emb) AS score FROM c ORDER BY VectorDistance(c.e, @emb)"),
                parameters=[{"name": "@k", "value": seed_k}, {"name": "@emb", "value": q_emb}]):
                r.append(item)
            return r

        basic_kw = _extract_keywords(question)
        t_es = time.perf_counter()
        seed_entities, llm_keywords = await asyncio.gather(
            _entity_search(), _llm_expand_keywords(question, engine)
        )
        timings["entity_search"] = time.perf_counter() - t_es

        if not seed_entities:
            yield _sse({"stage": "progress", "message": "No entities found.", "_ts": _elapsed(t0)})
            yield _sse({"stage": "token", "text": "No relevant entities found in the knowledge graph."})
            timings["total"] = time.perf_counter() - t0
            yield _sse({"stage": "done", "_ts": _elapsed(t0), "timings": timings})
            yield "data: [DONE]\n\n"
            return

        entity_names = [e["name"] for e in seed_entities[:10]]
        yield _sse({"stage": "progress",
                     "message": f"Found {len(seed_entities)} entities in {timings['entity_search']:.2f}s: {', '.join(entity_names[:5])}",
                     "_ts": _elapsed(t0)})

        # --- Step 2: Graph traversal + vector triples + food search + keyword search (parallel) ---
        yield _sse({"stage": "progress", "message": "Graph traversal + vector search + keyword search...", "_ts": _elapsed(t0)})

        async def _graph_traversal():
            """PK-based hop traversal like KG path."""
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
                    pk = name.lower()[:100]
                    r = []
                    async for triple in triples_ctr.query_items(
                        query=("SELECT c.id, c.s AS subject, c.p AS predicate, c.o AS object, "
                               "c.f AS confidence, c.d AS source_chunks FROM c WHERE c.s = @pk"),
                        parameters=[{"name": "@pk", "value": pk}],
                    ):
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
                       "c.claims_translated, c.ingredients_translated, c.allergens_translated, "
                       "c.pack_size_translated, c.product_title, c.claims, c.ingredients, c.allergens, c.pack_size, "
                       "VectorDistance(c.e, @emb) AS score FROM c ORDER BY VectorDistance(c.e, @emb)"),
                parameters=[{"name": "@k", "value": vec_k}, {"name": "@emb", "value": q_emb}]):
                for k in ("_rid", "_self", "_etag", "_attachments", "_ts"):
                    doc.pop(k, None)
                r.append(doc)
            return r

        async def _food_fulltext(keyword: str):
            r = []
            try:
                sql = _build_fulltext_sql(keyword)
                async for doc in food_ctr.query_items(
                    query=sql,
                    parameters=[{"name": "@k", "value": 10}, {"name": "@kw", "value": keyword}],
                ):
                    for k in ("_rid", "_self", "_etag", "_attachments", "_ts", "e"):
                        doc.pop(k, None)
                    r.append(doc)
            except Exception as e:
                log.warning("Fulltext search for '%s' failed: %s", keyword, e)
            return r

        all_kw = list(set(basic_kw[:5] + (llm_keywords or [])[:6]))
        ft_tasks = [_food_fulltext(kw) for kw in all_kw]
        log.info("Keywords basic=%s llm=%s combined=%s", basic_kw[:5], llm_keywords, all_kw)

        t_graph = time.perf_counter()
        graph_results = await asyncio.gather(
            _graph_traversal(), _triple_vec(), _food_vec(), *ft_tasks
        )
        pk_triples = graph_results[0]
        vec_triples = graph_results[1]
        vec_food = graph_results[2]
        ft_results = graph_results[3:]

        # Deduplicate triples
        all_triples_raw = pk_triples + vec_triples
        seen_keys: set[str] = set()
        all_triples = []
        for t in all_triples_raw:
            key = f"{t.get('subject','')}|{t.get('predicate','')}|{t.get('object','')}"
            if key not in seen_keys:
                seen_keys.add(key)
                all_triples.append(t)
        all_triples = all_triples[:max_triples]
        timings["graph_traversal"] = time.perf_counter() - t_graph

        yield _sse({"stage": "progress",
                     "message": f"Graph: {len(pk_triples)} PK + {len(vec_triples)} vec = {len(all_triples)} unique triples in {timings['graph_traversal']:.2f}s",
                     "_ts": _elapsed(t0)})

        # --- Step 3: Fetch source documents from KG references + merge with vector/keyword results ---
        yield _sse({"stage": "progress", "message": "Fetching source documents...", "_ts": _elapsed(t0)})

        t_src = time.perf_counter()
        source_chunk_ids: set[str] = set()
        for t in all_triples:
            for cid in t.get("source_chunks", []):
                source_chunk_ids.add(cid)
        for e in seed_entities[:5]:
            for cid in e.get("source_chunks", []):
                source_chunk_ids.add(cid)

        source_ids = list(source_chunk_ids)[:max_source]
        source_chunks: list[dict] = []

        if source_ids:
            for batch_start in range(0, len(source_ids), 20):
                batch = source_ids[batch_start:batch_start + 20]
                ids_param = ", ".join(f'"{sid}"' for sid in batch)
                async for doc in food_ctr.query_items(query=f"SELECT * FROM c WHERE c.id IN ({ids_param})"):
                    for k in ("e", "_rid", "_self", "_etag", "_attachments", "_ts"):
                        doc.pop(k, None)
                    source_chunks.append(doc)

        # Merge vector + keyword results
        seen_ids = {doc.get("id") for doc in source_chunks}
        for doc in vec_food:
            if doc.get("id") not in seen_ids:
                source_chunks.append(doc)
                seen_ids.add(doc.get("id"))
        for batch in ft_results:
            for doc in batch:
                if doc.get("id") not in seen_ids:
                    source_chunks.append(doc)
                    seen_ids.add(doc.get("id"))

        timings["source_fetch"] = time.perf_counter() - t_src

        yield _sse({"stage": "progress",
                     "message": f"Sources: {len(source_ids)} from KG + {len(vec_food)} vector + keyword = {len(source_chunks)} total in {timings['source_fetch']:.2f}s",
                     "_ts": _elapsed(t0)})

        # --- Step 4: Rerank ---
        t_rerank = time.perf_counter()
        source_chunks = await _semantic_rerank(food_ctr, question, source_chunks, top_k=RERANKER_TOP_K)
        timings["rerank"] = time.perf_counter() - t_rerank

        yield _sse({
            "stage": "stats",
            "seed_entities": len(seed_entities),
            "triples_found": len(all_triples),
            "source_chunks": len(source_chunks),
            "entity_names": [e["name"] for e in seed_entities[:8]],
            "_ts": _elapsed(t0),
        })

        # --- Step 5: Build prompt + non-streaming LLM call (DFlash speculative decoding) ---
        yield _sse({"stage": "progress",
                     "message": f"Retrieval done in {time.perf_counter() - t0:.1f}s — calling LLM (DFlash)...",
                     "_ts": _elapsed(t0)})

        graph_context = engine._build_graph_context(seed_entities, all_triples)
        source_text = engine._build_source_text(source_chunks)
        prompt = DFLASH_ANSWER_PROMPT.replace("{graph_context}", graph_context) \
                                      .replace("{source_chunks}", source_text) \
                                      .replace("{question}", question)

        t_llm = time.perf_counter()
        resp = await engine._llm.chat.completions.create(
            model=engine._llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful food product expert. Always recommend products."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=engine._max_tokens,
            **engine._llm_call_kwargs,
        )
        answer = resp.choices[0].message.content if resp.choices else ""
        timings["llm"] = time.perf_counter() - t_llm
        timings["total"] = time.perf_counter() - t0

        yield _sse({"stage": "progress",
                     "message": f"Answer ready in {timings['total']:.1f}s (LLM: {timings['llm']:.1f}s) — streaming...",
                     "_ts": str(timings["total"])})

        chunk_size = 80
        for i in range(0, len(answer), chunk_size):
            yield _sse({"stage": "token", "text": answer[i:i + chunk_size]})

        yield _sse({"stage": "done", "_ts": _elapsed(t0), "timings": timings})

    except Exception as e:
        log.exception("dflash stream error: %s", e)
        yield _sse({"stage": "error", "message": str(e)})

    yield "data: [DONE]\n\n"


async def _stream_kg_sse(question: str, engine: KGQueryEngine, backend_id: str = "kg"):
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    try:
        yield _sse({"stage": "progress", "message": "Embedding question...", "_ts": _elapsed(t0)})

        t_embed = time.perf_counter()
        q_emb = await engine._embedder.embed(question)
        timings["embed"] = time.perf_counter() - t_embed

        yield _sse({"stage": "progress", "message": f"Embedded in {timings['embed']:.2f}s", "_ts": _elapsed(t0)})
        yield _sse({"stage": "progress", "message": "Searching entity index...", "_ts": _elapsed(t0)})

        cosmos = await engine._get_cosmos()
        db = cosmos.get_database_client(engine._db_name)

        entities_container = db.get_container_client(
            engine._kg_cfg.get("entities_container", "kg_entities_food")
        )
        seed_k = int(engine._query_cfg.get("seed_entities_k", 20))

        t_es = time.perf_counter()
        sql = (
            "SELECT TOP @k c.n AS name, c.t AS description, c.r AS relation_count, c.d AS source_chunks, "
            "VectorDistance(c.e, @emb) AS score "
            "FROM c ORDER BY VectorDistance(c.e, @emb)"
        )
        seed_entities = []
        async for item in entities_container.query_items(
            query=sql,
            parameters=[{"name": "@k", "value": seed_k}, {"name": "@emb", "value": q_emb}],
        ):
            seed_entities.append(item)
        timings["entity_search"] = time.perf_counter() - t_es

        if not seed_entities:
            yield _sse({"stage": "progress", "message": "No entities found.", "_ts": _elapsed(t0)})
            yield _sse({"stage": "token", "text": "No relevant entities found in the knowledge graph."})
            timings["total"] = time.perf_counter() - t0
            yield _sse({"stage": "done", "_ts": _elapsed(t0), "timings": timings})
            yield "data: [DONE]\n\n"
            return

        entity_names = [e["name"] for e in seed_entities[:10]]
        yield _sse({"stage": "progress", "message": f"Found {len(seed_entities)} entities in {timings['entity_search']:.2f}s", "_ts": _elapsed(t0)})
        yield _sse({"stage": "progress", "message": "Graph traversal...", "_ts": _elapsed(t0)})

        t_graph = time.perf_counter()
        triples_container = db.get_container_client(
            engine._kg_cfg.get("triples_container", "kg_triples_food")
        )
        max_hops = int(engine._query_cfg.get("max_hops", 2))
        max_triples = int(engine._query_cfg.get("max_triples", 150))

        all_triples: list[dict] = []
        visited_entities: set[str] = set()

        async def _fetch_triples(name: str):
            pk = name.lower()[:100]
            results = []
            async for triple in triples_container.query_items(
                query=("SELECT c.id, c.s AS subject, c.p AS predicate, c.o AS object, "
                       "c.f AS confidence, c.d AS source_chunks FROM c WHERE c.s = @pk"),
                parameters=[{"name": "@pk", "value": pk}],
            ):
                results.append(triple)
            return results

        for hop in range(max_hops):
            if not entity_names:
                break
            batch = [n for n in entity_names if n not in visited_entities]
            if not batch:
                break
            for n in batch[:10]:
                visited_entities.add(n)
            results = await asyncio.gather(*[_fetch_triples(n) for n in batch[:10]])
            for r in results:
                all_triples.extend(r)
            if hop == 0 and len(all_triples) < max_triples:
                entity_names = list({t["object"] for t in all_triples
                                    if t["object"] not in visited_entities})[:5]

        triple_sql = (
            "SELECT TOP @k c.s AS subject, c.p AS predicate, c.o AS object, c.f AS confidence, c.d AS source_chunks, "
            "VectorDistance(c.e, @emb) AS score "
            "FROM c ORDER BY VectorDistance(c.e, @emb)"
        )
        async for triple in triples_container.query_items(
            query=triple_sql,
            parameters=[{"name": "@k", "value": 30}, {"name": "@emb", "value": q_emb}],
        ):
            all_triples.append(triple)

        seen_keys: set[str] = set()
        unique = []
        for t in all_triples:
            key = f"{t.get('subject','')}|{t.get('predicate','')}|{t.get('object','')}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique.append(t)
        all_triples = unique[:max_triples]
        timings["graph_traversal"] = time.perf_counter() - t_graph

        yield _sse({"stage": "progress", "message": f"Graph: {len(all_triples)} triples in {timings['graph_traversal']:.2f}s", "_ts": _elapsed(t0)})
        yield _sse({"stage": "progress", "message": "Fetching source documents...", "_ts": _elapsed(t0)})

        t_src = time.perf_counter()
        source_chunk_ids: set[str] = set()
        for t in all_triples:
            for cid in t.get("source_chunks", []):
                source_chunk_ids.add(cid)
        for e in seed_entities[:5]:
            for cid in e.get("source_chunks", []):
                source_chunk_ids.add(cid)

        max_source = int(engine._query_cfg.get("max_source_chunks", 30))
        source_ids = list(source_chunk_ids)[:max_source]
        source_chunks: list[dict] = []
        food_container = db.get_container_client("food")

        if source_ids:
            for batch_start in range(0, len(source_ids), 20):
                batch = source_ids[batch_start:batch_start + 20]
                ids_param = ", ".join(f'"{sid}"' for sid in batch)
                async for doc in food_container.query_items(query=f"SELECT * FROM c WHERE c.id IN ({ids_param})"):
                    for k in ("e", "_rid", "_self", "_etag", "_attachments", "_ts"):
                        doc.pop(k, None)
                    source_chunks.append(doc)

        vec_k = int(engine._query_cfg.get("vector_augment_k", 15))
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
            parameters=[{"name": "@k", "value": vec_k}, {"name": "@emb", "value": q_emb}],
        ):
            if doc.get("id") not in seen_ids:
                for k in ("_rid", "_self", "_etag", "_attachments", "_ts"):
                    doc.pop(k, None)
                source_chunks.append(doc)
                seen_ids.add(doc.get("id"))

        timings["source_fetch"] = time.perf_counter() - t_src

        yield _sse({
            "stage": "stats",
            "seed_entities": len(seed_entities),
            "triples_found": len(all_triples),
            "source_chunks": len(source_chunks),
            "entity_names": [e["name"] for e in seed_entities[:8]],
            "_ts": _elapsed(t0),
        })
        yield _sse({"stage": "progress", "message": f"Retrieval done in {time.perf_counter() - t0:.1f}s — calling LLM...", "_ts": _elapsed(t0)})

        t_llm = time.perf_counter()
        from prompts_kg_food import GRAPHRAG_ANSWER_PROMPT
        graph_context = engine._build_graph_context(seed_entities, all_triples)
        source_text = engine._build_source_text(source_chunks)
        template = DFLASH_ANSWER_PROMPT if backend_id == "dflash" else GRAPHRAG_ANSWER_PROMPT
        prompt = template.replace("{graph_context}", graph_context) \
                         .replace("{source_chunks}", source_text) \
                         .replace("{question}", question)

        resp = await engine._llm.chat.completions.create(
            model=engine._llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=engine._max_tokens,
            **engine._llm_call_kwargs,
        )
        answer = resp.choices[0].message.content if resp.choices else ""
        timings["llm"] = time.perf_counter() - t_llm
        timings["total"] = time.perf_counter() - t0

        yield _sse({"stage": "progress",
                     "message": f"Answer ready in {timings['total']:.1f}s (LLM: {timings['llm']:.1f}s) — streaming...",
                     "_ts": str(timings["total"])})

        chunk_size = 80
        for i in range(0, len(answer), chunk_size):
            yield _sse({"stage": "token", "text": answer[i:i + chunk_size]})

        yield _sse({"stage": "done", "_ts": _elapsed(t0), "timings": timings})

    except Exception as e:
        log.exception("stream error: %s", e)
        yield _sse({"stage": "error", "message": str(e)})

    yield "data: [DONE]\n\n"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

def _elapsed(t0: float) -> float:
    return round(time.perf_counter() - t0, 2)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Warming up embedding model...")
    embed_sync("warmup")

    kg_engines: dict[str, KGQueryEngine] = {}
    questions: dict[str, list[dict]] = {}

    main_cfg = os.environ.get("KG_CONFIG")
    if not main_cfg:
        raise RuntimeError(
            "KG_CONFIG is not set. Start the app with "
            "'python api.py --config <path-to-config.yaml>'."
        )

    backend_configs = {
        "kg": _ROOT / BACKENDS["kg"]["config"],
        "dflash": Path(main_cfg),
    }
    for bid, cfg_path in backend_configs.items():
        if cfg_path.exists():
            cfg = load_config(str(cfg_path))
            kg_engines[bid] = KGQueryEngine(cfg)
            questions[bid] = _load_questions_from_cfg(cfg)
            log.info("Backend %s loaded from %s: %d questions", bid, cfg_path, len(questions[bid]))

    # Original AgenticRetrieval
    try:
        ok = await _init_original()
        if ok:
            questions["original"] = _load_questions_from_path(
                "/home/azureuser/AgenticRetrieval/data/food.json"
            )
            log.info("Backend original loaded: %d questions", len(questions["original"]))
    except Exception as e:
        log.warning("Could not initialize original pipeline: %s", e)

    app.state.kg_engines = kg_engines
    app.state.questions = questions

    yield

    for engine in kg_engines.values():
        await engine.close()
    if _original_llm:
        try:
            await _original_llm.close()
        except Exception:
            pass


app = FastAPI(title="Food KG-RAG (Multi-Backend)", version="2.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(_ROOT / "static" / "index.html")

app.mount("/static", StaticFiles(directory=str(_ROOT / "static")), name="static")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/v1/backends")
async def get_backends():
    available = []
    for bid, binfo in BACKENDS.items():
        is_ready = bid in app.state.questions and (
            bid == "original" and _original_pipeline is not None
            or bid in app.state.kg_engines
        )
        if is_ready:
            available.append({
                "id": bid,
                "label": binfo["label"],
                "description": binfo["description"],
                "badge_color": binfo["badge_color"],
                "question_count": len(app.state.questions.get(bid, [])),
            })
    return JSONResponse(content=available)

@app.get("/v1/questions")
async def get_questions(backend: str = "dflash"):
    return JSONResponse(content=app.state.questions.get(backend, []))

@app.post("/v1/ask/stream")
async def ask_stream(body: AskRequest):
    if body.backend == "original":
        gen = _run_original_stream(body.question)
    elif body.backend == "dflash":
        engine = app.state.kg_engines.get("dflash")
        if not engine:
            engine = next(iter(app.state.kg_engines.values()))
        gen = _stream_dflash_sse(body.question, engine)
    else:
        engine = app.state.kg_engines.get(body.backend)
        if not engine:
            engine = next(iter(app.state.kg_engines.values()))
        gen = _stream_kg_sse(body.question, engine, backend_id=body.backend)

    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

async def _dflash_answer(question: str, engine: KGQueryEngine) -> dict:
    """Non-streaming DFlash: full KG retrieval + non-streaming LLM, returns result dict."""
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    q_emb = await engine._embedder.embed(question)
    timings["embed"] = time.perf_counter() - t0

    cosmos = await engine._get_cosmos()
    db = cosmos.get_database_client(engine._db_name)
    entities_ctr = db.get_container_client(engine._kg_cfg.get("entities_container", "kg_entities_food"))
    triples_ctr = db.get_container_client(engine._kg_cfg.get("triples_container", "kg_triples_food"))
    food_ctr = db.get_container_client("food")

    cfg = engine._query_cfg
    seed_k = int(cfg.get("seed_entities_k", 20))
    max_hops = int(cfg.get("max_hops", 2))
    max_triples = int(cfg.get("max_triples", 150))
    max_source = int(cfg.get("max_source_chunks", 30))
    vec_k = int(cfg.get("vector_augment_k", 15))

    # Entity search + keyword expansion
    async def _es():
        r = []
        async for item in entities_ctr.query_items(
            query=("SELECT TOP @k c.n AS name, c.t AS description, c.r AS relation_count, c.d AS source_chunks, "
                   "VectorDistance(c.e, @emb) AS score FROM c ORDER BY VectorDistance(c.e, @emb)"),
            parameters=[{"name": "@k", "value": seed_k}, {"name": "@emb", "value": q_emb}]):
            r.append(item)
        return r

    basic_kw = _extract_keywords(question)
    t_es = time.perf_counter()
    seed_entities, llm_keywords = await asyncio.gather(
        _es(), _llm_expand_keywords(question, engine)
    )
    timings["entity_search"] = time.perf_counter() - t_es

    if not seed_entities:
        timings["total"] = time.perf_counter() - t0
        return {"answer": "No relevant entities found.", "timings": timings}

    entity_names = [e["name"] for e in seed_entities[:10]]

    # Graph traversal + vector + keyword (parallel)
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
                pk = name.lower()[:100]
                r = []
                async for triple in triples_ctr.query_items(
                    query=("SELECT c.id, c.s AS subject, c.p AS predicate, c.o AS object, "
                           "c.f AS confidence, c.d AS source_chunks FROM c WHERE c.s = @pk"),
                    parameters=[{"name": "@pk", "value": pk}]):
                    r.append(triple)
                return r
            results = await asyncio.gather(*[_fetch_pk(n) for n in batch[:10]])
            for r in results:
                all_t.extend(r)
            if hop == 0 and len(all_t) < max_triples:
                names = list({t["object"] for t in all_t if t["object"] not in visited})[:5]
        return all_t

    async def _tv():
        r = []
        async for t in triples_ctr.query_items(
            query=("SELECT TOP @k c.s AS subject, c.p AS predicate, c.o AS object, c.f AS confidence, c.d AS source_chunks, "
                   "VectorDistance(c.e, @emb) AS score FROM c ORDER BY VectorDistance(c.e, @emb)"),
            parameters=[{"name": "@k", "value": 30}, {"name": "@emb", "value": q_emb}]):
            r.append(t)
        return r

    async def _fv():
        r = []
        async for doc in food_ctr.query_items(
            query=("SELECT TOP @k c.id, c.product_id, c.product_title_translated, c.brand, "
                   "c.claims_translated, c.ingredients_translated, c.allergens_translated, "
                   "c.pack_size_translated, c.product_title, c.claims, c.ingredients, c.allergens, c.pack_size, "
                   "VectorDistance(c.e, @emb) AS score FROM c ORDER BY VectorDistance(c.e, @emb)"),
            parameters=[{"name": "@k", "value": vec_k}, {"name": "@emb", "value": q_emb}]):
            for k in ("_rid", "_self", "_etag", "_attachments", "_ts"):
                doc.pop(k, None)
            r.append(doc)
        return r

    async def _ft(keyword: str):
        r = []
        try:
            sql = _build_fulltext_sql(keyword)
            async for doc in food_ctr.query_items(
                query=sql, parameters=[{"name": "@k", "value": 10}, {"name": "@kw", "value": keyword}]):
                for k in ("_rid", "_self", "_etag", "_attachments", "_ts", "e"):
                    doc.pop(k, None)
                r.append(doc)
        except Exception:
            pass
        return r

    all_kw = list(set(basic_kw[:5] + (llm_keywords or [])[:6]))
    ft_tasks = [_ft(kw) for kw in all_kw]

    t_graph = time.perf_counter()
    graph_results = await asyncio.gather(_graph_traversal(), _tv(), _fv(), *ft_tasks)
    pk_triples = graph_results[0]
    vec_triples = graph_results[1]
    vec_food = graph_results[2]
    ft_results = graph_results[3:]

    seen_keys: set[str] = set()
    all_triples = []
    for t in pk_triples + vec_triples:
        key = f"{t.get('subject','')}|{t.get('predicate','')}|{t.get('object','')}"
        if key not in seen_keys:
            seen_keys.add(key)
            all_triples.append(t)
    all_triples = all_triples[:max_triples]
    timings["graph_traversal"] = time.perf_counter() - t_graph

    # Source chunk fetch + merge
    t_src = time.perf_counter()
    source_chunk_ids: set[str] = set()
    for t in all_triples:
        for cid in t.get("source_chunks", []):
            source_chunk_ids.add(cid)
    for e in seed_entities[:5]:
        for cid in e.get("source_chunks", []):
            source_chunk_ids.add(cid)

    source_ids = list(source_chunk_ids)[:max_source]
    source_chunks: list[dict] = []
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
    for batch in ft_results:
        for doc in batch:
            if doc.get("id") not in seen_ids:
                source_chunks.append(doc)
                seen_ids.add(doc.get("id"))
    timings["source_fetch"] = time.perf_counter() - t_src

    source_chunks = await _semantic_rerank(food_ctr, question, source_chunks, top_k=RERANKER_TOP_K)

    graph_context = engine._build_graph_context(seed_entities, all_triples)
    source_text = engine._build_source_text(source_chunks)
    prompt = DFLASH_ANSWER_PROMPT.replace("{graph_context}", graph_context) \
                                  .replace("{source_chunks}", source_text) \
                                  .replace("{question}", question)

    t_llm = time.perf_counter()
    resp = await engine._llm.chat.completions.create(
        model=engine._llm_model,
        messages=[
            {"role": "system", "content": "You are a helpful food product expert. Always recommend products."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=engine._max_tokens,
        **engine._llm_call_kwargs,
    )
    timings["llm"] = time.perf_counter() - t_llm
    timings["total"] = time.perf_counter() - t0

    answer = resp.choices[0].message.content if resp.choices else ""
    return {"answer": answer, "timings": timings}


@app.post("/v1/ask")
async def ask(body: AskRequest):
    t0 = time.perf_counter()

    if body.backend == "original":
        if _original_pipeline is None:
            return JSONResponse(status_code=503, content={"error": "Original pipeline not initialized"})
        result = await _original_pipeline.run(body.question)
        wall = time.perf_counter() - t0
        return JSONResponse(content={
            "answer": result.get("final_answer", ""),
            "timings": {"total": wall, "llm": wall},
            "http_wall_s": round(wall, 4),
        })

    if body.backend == "dflash":
        engine = app.state.kg_engines.get("dflash")
        if not engine:
            engine = next(iter(app.state.kg_engines.values()))
        result = await _dflash_answer(body.question, engine)
        result["http_wall_s"] = round(time.perf_counter() - t0, 4)
        return JSONResponse(content=result)

    engine = app.state.kg_engines.get(body.backend)
    if not engine:
        engine = app.state.kg_engines.get("dflash", next(iter(app.state.kg_engines.values())))
    result = await engine.answer(body.question)
    wall = time.perf_counter() - t0
    result["http_wall_s"] = round(wall, 4)
    return JSONResponse(content=result)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Food KG-RAG multi-backend API")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the KG backend YAML config (drives the primary model backend).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080).")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        parser.error(f"Config file not found: {cfg_path}")
    os.environ["KG_CONFIG"] = str(cfg_path.resolve())

    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)
