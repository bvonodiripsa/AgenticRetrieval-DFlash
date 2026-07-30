#!/usr/bin/env python
"""
GI-RAG API + Web UI — single graph-index + LLM backend.

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

from gi_builder import load_config
from gi_query import GIQueryEngine
from retrieval import retrieve

_ROOT = Path(__file__).parent
log = logging.getLogger("food_dflash.api")

BACKENDS = {
    "gi": {
        "label": "GI-RAG",
        "description": "Graph Index RAG + LLM (speculative decoding when supported)",
        "badge_color": "#059669",
    },
}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # Retained for API compatibility; there is a single backend now.
    backend: str = Field(default="gi")


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
# GI-RAG streaming (single Graph Index + LLM backend)
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
    """Return configured GI/food containers that don't exist (queried from Cosmos)."""
    missing: list[str] = []
    try:
        cosmos = await engine._get_cosmos()
        db = cosmos.get_database_client(engine._db_name)
        gi = engine._gi_cfg
        for n in (gi.get("entities_container", "entities"),
                  gi.get("triples_container", "triples"),
                  "food"):
            try:
                await db.get_container_client(n).read()
            except Exception as e:
                if "NotFound" in str(e) or "404" in str(e):
                    missing.append(n)
    except Exception:
        pass
    return missing


async def _stream_dflash_sse(question: str, engine: GIQueryEngine):
    """DFlash path: GI retrieval (local index or Cosmos, via retrieval.py) + real
    token-by-token LLM streaming with speculative decoding.

    Retrieval used to be duplicated here as hardcoded Cosmos queries, separate
    from `_dflash_answer`'s copy. Both now call the same `retrieve()` against
    whichever backend `index.mode` selects, so this path gets the local-index
    speedup for free and there is exactly one place that implements the
    five-stage pipeline. The LLM call is also switched from
    "wait for the full completion, then fake-chunk it" to `stream=True`, so
    time-to-first-token drops from the full generation time to roughly one
    speculative-decoding step.
    """
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    try:
        yield _sse({"stage": "progress", "message": "Embedding question...", "_ts": _elapsed(t0)})

        t_embed = time.perf_counter()
        q_emb = await engine._embedder.embed(question)
        timings["embed"] = time.perf_counter() - t_embed
        yield _sse({"stage": "progress", "message": f"Embedded in {timings['embed']:.2f}s", "_ts": _elapsed(t0)})

        # Keyword expansion is an LLM call; start it now so it overlaps retrieval.
        basic_kw = _extract_keywords(question)
        kw_task = asyncio.create_task(_llm_expand_keywords(question, engine))

        yield _sse({"stage": "progress", "message": "Retrieving (entity search + graph traversal + sources)...",
                     "_ts": _elapsed(t0)})

        backend = await engine._get_backend()
        t_retr = time.perf_counter()
        result = await retrieve(backend, q_emb, engine._cfg)
        timings.update(result.timings)
        seed_entities = result.seed_entities
        all_triples = result.triples
        source_chunks = result.source_chunks

        if not seed_entities:
            kw_task.cancel()
            yield _sse({"stage": "progress", "message": "No entities found.", "_ts": _elapsed(t0)})
            yield _sse({"stage": "token", "text": "No relevant entities found in the graph index."})
            timings["total"] = time.perf_counter() - t0
            yield _sse({"stage": "done", "_ts": _elapsed(t0), "timings": timings})
            yield "data: [DONE]\n\n"
            return

        entity_names = [e["name"] for e in seed_entities[:8]]
        yield _sse({"stage": "progress",
                     "message": f"Retrieved {len(seed_entities)} entities, {len(all_triples)} triples, "
                                f"{len(source_chunks)} sources in {time.perf_counter() - t_retr:.2f}s "
                                f"({result.stats.get('pk_triples', 0)} PK + {result.stats.get('vec_triples', 0)} vec): "
                                f"{', '.join(entity_names[:5])}",
                     "_ts": _elapsed(t0)})

        # --- Keyword-expanded full-text search, merged into source_chunks ---
        llm_keywords = await kw_task
        all_kw = list(set(basic_kw[:5] + (llm_keywords or [])[:6]))
        log.info("Keywords basic=%s llm=%s combined=%s", basic_kw[:5], llm_keywords, all_kw)

        t_ft = time.perf_counter()
        seen_ids = {doc.get("id") for doc in source_chunks}
        for doc in await backend.fulltext_food(all_kw, 10):
            if doc.get("id") not in seen_ids:
                source_chunks.append(doc)
                seen_ids.add(doc.get("id"))
        timings["source_fetch"] += time.perf_counter() - t_ft

        # --- Rerank ---
        t_rerank = time.perf_counter()
        source_chunks = await _semantic_rerank(engine, question, source_chunks)
        timings["rerank"] = time.perf_counter() - t_rerank

        yield _sse({
            "stage": "stats",
            "seed_entities": len(seed_entities),
            "triples_found": len(all_triples),
            "source_chunks": len(source_chunks),
            "entity_names": entity_names,
            "_ts": _elapsed(t0),
        })

        # --- Build prompt + streaming LLM call ---
        yield _sse({"stage": "progress",
                     "message": f"Retrieval done in {time.perf_counter() - t0:.1f}s — calling LLM ({engine._llm_model})...",
                     "_ts": _elapsed(t0)})

        graph_context = engine._build_graph_context(seed_entities, all_triples)
        source_text = engine._build_source_text(source_chunks)
        prompt = DFLASH_ANSWER_PROMPT.replace("{graph_context}", graph_context) \
                                      .replace("{source_chunks}", source_text) \
                                      .replace("{question}", question)

        t_llm = time.perf_counter()
        first_token_at: float | None = None
        stream = await engine._llm.chat.completions.create(
            model=engine._llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful food product expert. Always recommend products."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=engine._max_tokens,
            stream=True,
            **engine._llm_call_kwargs,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
                timings["ttft"] = first_token_at - t_llm
            yield _sse({"stage": "token", "text": delta})

        timings["llm"] = time.perf_counter() - t_llm
        timings["total"] = time.perf_counter() - t0

        yield _sse({"stage": "done", "_ts": _elapsed(t0), "timings": timings})

    except Exception as e:
        log.exception("dflash stream error: %s", e)
        msg = str(e)
        if "NotFound" in msg or "404" in msg:
            missing = await _identify_missing_containers(engine)
            if missing:
                msg = (
                    f"Cosmos DB container(s) not found in database '{engine._db_name}': "
                    f"{', '.join(missing)}. Check gi.triples_container / gi.entities_container "
                    f"in your config, or (re)build the graph index."
                )
        yield _sse({"stage": "error", "message": msg})

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
    main_cfg = os.environ.get("GI_CONFIG", str(_ROOT / "my.yaml"))
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

    engine = GIQueryEngine(cfg)
    questions = _load_questions_from_cfg(cfg)
    log.info("Backend loaded from %s: %d questions", cfg_path, len(questions))

    # Warm up the embedder and the retrieval backend concurrently. Best-effort
    # — don't fail startup if either errors, so an unreachable Cosmos or a
    # missing snapshot doesn't take the whole app down; failures surface on
    # first request instead. Warming the backend here specifically matters
    # for index.mode: local — loading the snapshot (numpy -> GPU, BM25 build)
    # takes a few seconds, and without this it happens inside whichever
    # request arrives first instead of before the app starts accepting them.
    log.info("Warming up embedder + retrieval backend...")
    results = await asyncio.gather(
        engine._embedder.embed("warmup"),
        engine._get_backend(),
        return_exceptions=True,
    )
    for label, result in zip(("embedder", "backend"), results):
        if isinstance(result, Exception):
            log.warning("%s warmup failed: %s", label, result)

    app.state.engine = engine
    app.state.questions = questions

    yield

    await engine.close()


app = FastAPI(title="Food GI-RAG", version="2.0.0", lifespan=lifespan)


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
async def get_questions(backend: str = "gi"):
    return JSONResponse(content=app.state.questions)

@app.post("/v1/ask/stream")
async def ask_stream(body: AskRequest):
    gen = _stream_dflash_sse(body.question, app.state.engine)

    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

async def _dflash_answer(question: str, engine: GIQueryEngine) -> dict:
    """Non-streaming DFlash: full GI retrieval + non-streaming LLM, returns result dict."""
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    q_emb = await engine._embedder.embed(question)
    timings["embed"] = time.perf_counter() - t0

    # Keyword expansion is an LLM call, so start it now and let it run
    # alongside retrieval; its results are only needed for the full-text merge.
    basic_kw = _extract_keywords(question)
    kw_task = asyncio.create_task(_llm_expand_keywords(question, engine))

    backend = await engine._get_backend()
    result = await retrieve(backend, q_emb, engine._cfg)
    timings.update(result.timings)
    seed_entities = result.seed_entities
    all_triples = result.triples
    source_chunks = result.source_chunks

    if not seed_entities:
        kw_task.cancel()
        timings["total"] = time.perf_counter() - t0
        return {"answer": "No relevant entities found.", "timings": timings}

    llm_keywords = await kw_task
    all_kw = list(set(basic_kw[:5] + (llm_keywords or [])[:6]))
    log.info("Keywords basic=%s llm=%s combined=%s", basic_kw[:5], llm_keywords, all_kw)

    t_ft = time.perf_counter()
    seen_ids = {doc.get("id") for doc in source_chunks}
    for doc in await backend.fulltext_food(all_kw, 10):
        if doc.get("id") not in seen_ids:
            source_chunks.append(doc)
            seen_ids.add(doc.get("id"))
    timings["source_fetch"] += time.perf_counter() - t_ft

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

    parser = argparse.ArgumentParser(description="Food GI-RAG API (single Graph Index + LLM backend)")
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
    os.environ["GI_CONFIG"] = str(cfg_path.resolve())

    uvicorn.run("api:app", host=args.host, port=args.port, reload=False)
