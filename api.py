#!/usr/bin/env python
"""
KG-RAG API + Web UI — single knowledge-graph + LLM backend.

Pipeline: entity/triple vector search + graph traversal + LLM keyword expansion +
semantic rerank, then a single LLM answer call (speculative decoding when the
configured model/endpoint supports it).

Config is a single YAML file (default: my.yaml; override with --config).
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

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kg_builder import load_config
from kg_query import KGQueryEngine

_ROOT = Path(__file__).parent
log = logging.getLogger("food_dflash.api")

BACKENDS = {
    "kg": {
        "label": "KG-RAG",
        "description": "Knowledge graph RAG + LLM (speculative decoding when supported)",
        "badge_color": "#059669",
    },
}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # Retained for API compatibility; there is a single backend now.
    backend: str = Field(default="kg")


def _load_questions_from_cfg(cfg: dict) -> list[dict]:
    paths = cfg.get("paths", {})
    # Upstream schema uses `questions_path`; keep `questions_file` fallback.
    qpath = _ROOT / paths.get("questions_path", paths.get("questions_file", "data/food.json"))
    if qpath.exists():
        data = json.loads(qpath.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return data
    return []


# ---------------------------------------------------------------------------
# KG-RAG streaming (single KG + LLM backend)
# ---------------------------------------------------------------------------

DFLASH_ANSWER_PROMPT = """You are a food product expert.

DATA:
{graph_context}
{source_chunks}

QUESTION: {question}

Recommend 8-10 products. For each: name, (product_id: XXXXX), one sentence on why it fits. Never say "no products match." End with top pick."""


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
        raw = (resp.choices[0].message.content or "").strip()
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


_RERANK_URL_SUFFIX = "dbinference.azure.com:443/inference/semanticReranking"


async def _rerank_token(engine, scope: str) -> str | None:
    """Acquire (and cache on the engine) a bearer token for the reranker service."""
    now = time.time()
    if getattr(engine, "_ranker_token", None) and now < getattr(engine, "_ranker_token_exp", 0) - 60:
        return engine._ranker_token
    try:
        await engine._get_cosmos()  # ensures engine._cred is set for RBAC configs
        cred = getattr(engine, "_cred", None)
        if cred is None:
            from azure.identity.aio import AzureCliCredential
            cred = AzureCliCredential()
        tok = await cred.get_token(scope)
        engine._ranker_token = tok.token
        engine._ranker_token_exp = tok.expires_on
        return tok.token
    except Exception as e:
        log.warning("Reranker token acquisition failed: %s", e)
        return None


async def _semantic_rerank(engine, question: str, docs: list[dict]) -> list[dict]:
    """Rerank food docs via the Cosmos semantic-reranker HTTP endpoint (ranker.* config).

    Mirrors the upstream CombinedRetriever behaviour: every candidate is scored
    by the ranker and the top ``ranker.k_ranker`` are kept. Falls back to the
    existing order when the ranker is disabled/unconfigured, on any error, or
    when there are already <= k_ranker candidates.
    """
    if not docs:
        return docs

    ranker = engine._cfg.get("ranker", {})
    account = str(ranker.get("account_name", "")).strip()
    region = str(ranker.get("region", "")).strip()
    k_ranker = int(ranker.get("k_ranker", 0) or 0)
    if not ranker.get("use_ranker", True) or not account or not region or k_ranker <= 0:
        return docs
    # Nothing to trim if we already have <= k_ranker candidates.
    if len(docs) <= k_ranker:
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

    # The ranker rejects payloads containing empty strings.
    if any(not (isinstance(s, str) and s.strip()) for s in doc_strings):
        return docs

    scope = str(ranker.get("token_scope", "https://dbinference.azure.com/.default")).strip()
    token = await _rerank_token(engine, scope)
    if not token:
        return docs

    url_suffix = str(ranker.get("url_suffix", _RERANK_URL_SUFFIX)).strip()
    url = f"https://{account}.{region}.{url_suffix}"
    body = {
        "query": question,
        "documents": doc_strings,
        "return_documents": False,
        "top_k": k_ranker,
        "batch_size": int(ranker.get("batch_size", 32)),
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        import httpx
        client = getattr(engine, "_ranker_http", None)
        if client is None:
            client = httpx.AsyncClient(timeout=30)
            engine._ranker_http = client
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        scores = resp.json().get("Scores", [])
        if scores:
            return [docs[s["index"]] for s in scores if s["index"] < len(docs)]
    except Exception as e:
        log.warning("Semantic reranker failed (falling back to vector order): %s", e)

    return docs


async def _identify_missing_containers(engine) -> list[str]:
    """Return configured KG/food containers that don't exist (queried from Cosmos)."""
    missing: list[str] = []
    try:
        cosmos = await engine._get_cosmos()
        db = cosmos.get_database_client(engine._db_name)
        kg = engine._kg_cfg
        for n in (kg.get("entities_container", "entities"),
                  kg.get("triples_container", "triples"),
                  "food"):
            try:
                await db.get_container_client(n).read()
            except Exception as e:
                if "NotFound" in str(e) or "404" in str(e):
                    missing.append(n)
    except Exception:
        pass
    return missing


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
                    pk = name
                    r = []
                    async for triple in triples_ctr.query_items(
                        query=f"SELECT c.s AS subject, c.p AS predicate, c.o AS object, c.f AS confidence, c.d AS source_chunks FROM c WHERE c.{engine._triples_pk_field} = @pk",
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
        source_chunks = await _semantic_rerank(engine, question, source_chunks)
        timings["rerank"] = time.perf_counter() - t_rerank

        yield _sse({
            "stage": "stats",
            "seed_entities": len(seed_entities),
            "triples_found": len(all_triples),
            "source_chunks": len(source_chunks),
            "entity_names": [e["name"] for e in seed_entities[:8]],
            "_ts": _elapsed(t0),
        })

        # --- Step 5: Build prompt + non-streaming LLM call ---
        yield _sse({"stage": "progress",
                     "message": f"Retrieval done in {time.perf_counter() - t0:.1f}s — calling LLM ({engine._llm_model})...",
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
        msg = str(e)
        if "NotFound" in msg or "404" in msg:
            missing = await _identify_missing_containers(engine)
            if missing:
                msg = (
                    f"Cosmos DB container(s) not found in database '{engine._db_name}': "
                    f"{', '.join(missing)}. Check kg.triples_container / kg.entities_container "
                    f"in your config, or (re)build the KG."
                )
        yield _sse({"stage": "error", "message": msg})

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
            pk = name
            results = []
            async for triple in triples_container.query_items(
                query=f"SELECT c.s AS subject, c.p AS predicate, c.o AS object, c.f AS confidence, c.d AS source_chunks FROM c WHERE c.{engine._triples_pk_field} = @pk",
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
    main_cfg = os.environ.get("KG_CONFIG", str(_ROOT / "my.yaml"))
    cfg_path = Path(main_cfg)
    if not cfg_path.exists():
        raise RuntimeError(
            f"Config not found: {cfg_path}. Provide --config or create my.yaml."
        )

    cfg = load_config(str(cfg_path))

    # Cosmos DB Semantic Reranker endpoint comes from config; an explicit env
    # var wins. The azure-cosmos SDK reads this env var at rerank time.
    reranker_endpoint = cfg.get("cosmos", {}).get("semantic_reranker_endpoint")
    if reranker_endpoint:
        os.environ.setdefault(
            "AZURE_COSMOS_SEMANTIC_RERANKER_INFERENCE_ENDPOINT", str(reranker_endpoint)
        )

    engine = KGQueryEngine(cfg)
    questions = _load_questions_from_cfg(cfg)
    log.info("Backend loaded from %s: %d questions", cfg_path, len(questions))

    # Warm up the embedder (loads the in-process model, or checks the HTTP
    # embedding endpoint). Best-effort — don't fail startup if it errors.
    log.info("Warming up embedder...")
    try:
        await engine._embedder.embed("warmup")
    except Exception as e:
        log.warning("Embedding warmup failed: %s", e)

    app.state.engine = engine
    app.state.questions = questions

    yield

    await engine.close()


app = FastAPI(title="Food KG-RAG", version="2.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(_ROOT / "static" / "index.html")

app.mount("/static", StaticFiles(directory=str(_ROOT / "static")), name="static")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/v1/backends")
async def get_backends():
    bid, binfo = next(iter(BACKENDS.items()))
    return JSONResponse(content=[{
        "id": bid,
        "label": binfo["label"],
        "description": binfo["description"],
        "badge_color": binfo["badge_color"],
        "question_count": len(app.state.questions),
    }])

@app.get("/v1/questions")
async def get_questions(backend: str = "kg"):
    return JSONResponse(content=app.state.questions)

@app.post("/v1/ask/stream")
async def ask_stream(body: AskRequest):
    gen = _stream_dflash_sse(body.question, app.state.engine)

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
                pk = name
                r = []
                async for triple in triples_ctr.query_items(
                    query=f"SELECT c.s AS subject, c.p AS predicate, c.o AS object, c.f AS confidence, c.d AS source_chunks FROM c WHERE c.{engine._triples_pk_field} = @pk",
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

    source_chunks = await _semantic_rerank(engine, question, source_chunks)

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
    result = await _dflash_answer(body.question, app.state.engine)
    result["http_wall_s"] = round(time.perf_counter() - t0, 4)
    return JSONResponse(content=result)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Food KG-RAG API (single KG + LLM backend)")
    parser.add_argument(
        "--config",
        default="my.yaml",
        help="Path to the YAML config (default: my.yaml).",
    )
    parser.add_argument("--host", default="localhost", help="Host to bind (default: localhost).")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080).")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        parser.error(f"Config file not found: {cfg_path}")
    os.environ["KG_CONFIG"] = str(cfg_path.resolve())

    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)
