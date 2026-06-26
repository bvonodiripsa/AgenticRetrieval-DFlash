#!/usr/bin/env python
"""
KG-RAG API + Web UI for Food Knowledge Graph with DFlash.

Online query flow:
  embed question → entity vector search → graph traversal → single LLM call

LLM: vLLM (Qwen3.5-27B + DFlash speculative decoding) on localhost:8000.
Embeddings: in-process Qwen3-Embedding-0.6B.
Data: Azure Cosmos DB ("food" database).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kg_builder import load_config, embed_sync
from kg_query import KGQueryEngine

_ROOT = Path(__file__).parent
_CONFIG_PATH = os.getenv("CONFIG_PATH", str(_ROOT / "config_kg_dflash.yaml"))
log = logging.getLogger("food_dflash.api")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


def _load_questions() -> list[dict]:
    cfg = load_config(_CONFIG_PATH)
    qpath = _ROOT / cfg.get("paths", {}).get("questions_file", "data/food.json")
    if qpath.exists():
        data = json.loads(qpath.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return data
    return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(_CONFIG_PATH)

    log.info("Warming up embedding model...")
    embed_sync("warmup")

    engine = KGQueryEngine(cfg)
    app.state.engine = engine
    app.state.questions = _load_questions()
    app.state.cfg = cfg

    yield

    await engine.close()


app = FastAPI(
    title="Food KG-RAG API (DFlash)",
    description="Knowledge-graph augmented generation for food products with DFlash acceleration.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(_ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=str(_ROOT / "static")), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/questions")
async def get_questions():
    return JSONResponse(content=app.state.questions)


# ---------------------------------------------------------------------------
# SSE streaming endpoint — yields progress events, then tokens, then done
# ---------------------------------------------------------------------------

async def _stream_answer_sse(question: str, engine: KGQueryEngine):
    """Stream KG-RAG answer via SSE with progress events."""
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    try:
        yield _sse({"stage": "progress", "message": "Embedding question...", "_ts": _elapsed(t0)})

        from kg_builder import EmbedClient
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
        timings["entity_search"] = time.perf_counter() - t_es

        if not seed_entities:
            yield _sse({"stage": "progress", "message": "No entities found.", "_ts": _elapsed(t0)})
            yield _sse({"stage": "token", "text": "No relevant entities found in the knowledge graph."})
            timings["total"] = time.perf_counter() - t0
            yield _sse({"stage": "done", "_ts": _elapsed(t0), "timings": timings})
            yield "data: [DONE]\n\n"
            return

        entity_names = [e["name"] for e in seed_entities[:10]]
        yield _sse({
            "stage": "progress",
            "message": f"Found {len(seed_entities)} entities in {timings['entity_search']:.2f}s",
            "_ts": _elapsed(t0),
        })
        yield _sse({"stage": "progress", "message": "Graph traversal...", "_ts": _elapsed(t0)})

        # Graph traversal
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
                query="SELECT * FROM c WHERE c.pk = @pk",
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
            "SELECT TOP @k c.subject, c.predicate, c.object, c.confidence, c.source_chunks, "
            "VectorDistance(c.embedding, @emb) AS score "
            "FROM c ORDER BY VectorDistance(c.embedding, @emb)"
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

        yield _sse({
            "stage": "progress",
            "message": f"Graph: {len(all_triples)} triples in {timings['graph_traversal']:.2f}s",
            "_ts": _elapsed(t0),
        })
        yield _sse({"stage": "progress", "message": "Fetching source documents...", "_ts": _elapsed(t0)})

        # Source fetch
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

        retrieval_time = time.perf_counter() - t0
        yield _sse({
            "stage": "progress",
            "message": f"Retrieval done in {retrieval_time:.1f}s — streaming answer...",
            "_ts": _elapsed(t0),
        })

        # LLM streaming
        t_llm = time.perf_counter()
        from prompts_kg_food import GRAPHRAG_ANSWER_PROMPT
        graph_context = engine._build_graph_context(seed_entities, all_triples)
        source_text = engine._build_source_text(source_chunks)
        prompt = GRAPHRAG_ANSWER_PROMPT.replace("{graph_context}", graph_context) \
                                       .replace("{source_chunks}", source_text) \
                                       .replace("{question}", question)

        stream = await engine._llm.chat.completions.create(
            model=engine._llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=engine._max_tokens,
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield _sse({"stage": "token", "text": delta.content})

        timings["llm"] = time.perf_counter() - t_llm
        timings["total"] = time.perf_counter() - t0

        yield _sse({
            "stage": "done",
            "_ts": _elapsed(t0),
            "timings": timings,
        })

    except Exception as e:
        log.exception("stream error: %s", e)
        yield _sse({"stage": "error", "message": str(e)})

    yield "data: [DONE]\n\n"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _elapsed(t0: float) -> float:
    return round(time.perf_counter() - t0, 2)


@app.post("/v1/ask/stream")
async def ask_stream(body: AskRequest):
    engine: KGQueryEngine = app.state.engine
    return StreamingResponse(
        _stream_answer_sse(body.question, engine),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/ask")
async def ask(body: AskRequest):
    t0 = time.perf_counter()
    engine: KGQueryEngine = app.state.engine
    result = await engine.answer(body.question)
    wall = time.perf_counter() - t0
    result["http_wall_s"] = round(wall, 4)
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=False)
