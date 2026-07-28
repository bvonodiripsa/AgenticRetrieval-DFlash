"""Single implementation of GI retrieval, shared by the CLI and the web API.

The same five-stage pipeline previously existed in four near-identical copies
(`gi_query.answer`, and three functions in `api.py`). It now lives here once,
behind a backend interface with two implementations:

    CosmosBackend  — the original remote queries
    LocalBackend   — in-process GPU index (see gi_index.py)

Both return identically shaped dicts, so prompt construction downstream is
unchanged and the two can be compared directly.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    seed_entities: list[dict[str, Any]] = field(default_factory=list)
    triples: list[dict[str, Any]] = field(default_factory=list)
    source_chunks: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- Cosmos

_ENTITY_SQL = (
    "SELECT TOP @k c.n AS name, c.t AS description, c.r AS relation_count, "
    "c.d AS source_chunks, VectorDistance(c.e, @emb) AS score "
    "FROM c ORDER BY VectorDistance(c.e, @emb)"
)
_TRIPLE_VEC_SQL = (
    "SELECT TOP @k c.s AS subject, c.p AS predicate, c.o AS object, c.f AS confidence, "
    "c.d AS source_chunks, VectorDistance(c.e, @emb) AS score "
    "FROM c ORDER BY VectorDistance(c.e, @emb)"
)
_FOOD_VEC_SQL = (
    "SELECT TOP @k c.id, c.product_id, c.product_title_translated, c.brand, "
    "c.claims_translated, c.ingredients_translated, c.allergens_translated, "
    "c.pack_size_translated, c.product_title, c.claims, c.ingredients, c.allergens, "
    "c.pack_size, VectorDistance(c.e, @emb) AS score "
    "FROM c ORDER BY VectorDistance(c.e, @emb)"
)
_FT_SQL = (
    "SELECT TOP @k c.id, c.product_id, c.product_title_translated, c.brand, "
    "c.claims_translated, c.ingredients_translated, c.allergens_translated, "
    "c.pack_size_translated, c.product_title, c.claims, c.ingredients, c.allergens, c.pack_size "
    "FROM c WHERE FullTextContains(c.product_title_translated, @kw) "
    "OR FullTextContains(c.ingredients_translated, @kw) "
    "OR FullTextContains(c.claims_translated, @kw)"
)
_STRIP = ("e", "_rid", "_self", "_etag", "_attachments", "_ts")


class CosmosBackend:
    """Original behaviour: one network round trip per query."""

    def __init__(self, entities_ctr, triples_ctr, food_ctr, triples_pk_field: str = "s"):
        self._entities, self._triples, self._food = entities_ctr, triples_ctr, food_ctr
        self._pk = triples_pk_field

    async def _collect(self, ctr, query, params=None, strip=False):
        out = []
        async for doc in ctr.query_items(query=query, parameters=params or []):
            if strip:
                for k in _STRIP:
                    doc.pop(k, None)
            out.append(doc)
        return out

    async def search_entities(self, q_emb, k):
        return await self._collect(self._entities, _ENTITY_SQL,
                                   [{"name": "@k", "value": k}, {"name": "@emb", "value": q_emb}])

    async def search_triples(self, q_emb, k):
        return await self._collect(self._triples, _TRIPLE_VEC_SQL,
                                   [{"name": "@k", "value": k}, {"name": "@emb", "value": q_emb}])

    async def search_food(self, q_emb, k):
        return await self._collect(self._food, _FOOD_VEC_SQL,
                                   [{"name": "@k", "value": k}, {"name": "@emb", "value": q_emb}], strip=True)

    async def triples_for(self, names, reverse=False):
        # `reverse` would need `WHERE c.o = @pk`, a cross-partition scan over
        # 1.59M documents, so it is not offered remotely.
        sql = (f"SELECT c.s AS subject, c.p AS predicate, c.o AS object, c.f AS confidence, "
               f"c.d AS source_chunks FROM c WHERE c.{self._pk} = @pk")
        batches = await asyncio.gather(*[
            self._collect(self._triples, sql, [{"name": "@pk", "value": n}]) for n in names
        ])
        return [t for b in batches for t in b]

    async def docs_by_id(self, ids):
        out = []
        ids = list(ids)
        for start in range(0, len(ids), 20):
            batch = ids[start:start + 20]
            in_list = ", ".join(f'"{i}"' for i in batch)
            out += await self._collect(self._food, f"SELECT * FROM c WHERE c.id IN ({in_list})", strip=True)
        return out

    async def fulltext_food(self, keywords, k):
        batches = await asyncio.gather(*[
            self._collect(self._food, _FT_SQL, [{"name": "@k", "value": k}, {"name": "@kw", "value": kw}])
            for kw in keywords
        ], return_exceptions=True)
        return [d for b in batches if isinstance(b, list) for d in b]


# ---------------------------------------------------------------------- local


class LocalBackend:
    """In-process GPU index. Same interface, no network."""

    def __init__(self, index, reverse_edges: bool = False):
        self._ix = index
        self.reverse_edges = reverse_edges

    async def search_entities(self, q_emb, k):
        return self._ix.search_entities(q_emb, k)

    async def search_triples(self, q_emb, k):
        return self._ix.search_triples(q_emb, k)

    async def search_food(self, q_emb, k):
        return self._ix.search_food(q_emb, k)

    async def triples_for(self, names, reverse=None):
        rev = self.reverse_edges if reverse is None else reverse
        out = []
        for n in names:
            out += self._ix.triples_for(n, reverse=rev)
        return out

    async def docs_by_id(self, ids):
        return self._ix.docs_by_id(ids)

    async def fulltext_food(self, keywords, k):
        out = []
        for kw in keywords:
            out += self._ix.fulltext_food(kw, k)
        return out


# ------------------------------------------------------------------ pipeline


async def retrieve(backend, q_emb, cfg: dict, *, keywords: list[str] | None = None) -> RetrievalResult:
    """Stages 2-4 of the GI-RAG pipeline. Embedding is the caller's job."""
    q = cfg.get("query", {})
    seed_k = int(q.get("seed_entities_k", 10))
    max_hops = int(q.get("max_hops", 1))
    max_triples = int(q.get("max_triples", 40))
    max_source = int(q.get("max_source_chunks", 15))
    vec_k = int(q.get("vector_augment_k", 12))

    res = RetrievalResult()

    # --- entity search -----------------------------------------------------
    t0 = time.perf_counter()
    seed_entities = await backend.search_entities(q_emb, seed_k)
    res.timings["entity_search"] = time.perf_counter() - t0
    res.seed_entities = seed_entities
    if not seed_entities:
        return res

    # --- graph traversal + triple vector search ----------------------------
    t0 = time.perf_counter()
    names = [e["name"] for e in seed_entities[:10]]
    visited: set[str] = set()
    pk_triples: list[dict] = []
    for hop in range(max_hops):
        batch = [n for n in names if n not in visited][:10]
        if not batch:
            break
        visited.update(batch)
        pk_triples += await backend.triples_for(batch)
        if hop == 0 and len(pk_triples) < max_triples:
            names = [t["object"] for t in pk_triples if t["object"] not in visited][:5]

    vec_triples = await backend.search_triples(q_emb, 30)

    seen: set[str] = set()
    merged = []
    for t in pk_triples + vec_triples:
        key = f"{t.get('subject','')}|{t.get('predicate','')}|{t.get('object','')}"
        if key not in seen:
            seen.add(key)
            merged.append(t)
    res.triples = merged[:max_triples]
    res.timings["graph_traversal"] = time.perf_counter() - t0
    res.stats["pk_triples"] = len(pk_triples)
    res.stats["vec_triples"] = len(vec_triples)

    # --- source documents --------------------------------------------------
    t0 = time.perf_counter()
    chunk_ids: set[str] = set()
    for t in res.triples:
        chunk_ids.update(t.get("source_chunks") or [])
    for e in seed_entities[:5]:
        chunk_ids.update(e.get("source_chunks") or [])
    source_ids = list(chunk_ids)[:max_source]

    source_chunks = await backend.docs_by_id(source_ids) if source_ids else []
    seen_ids = {d.get("id") for d in source_chunks}

    for doc in await backend.search_food(q_emb, vec_k):
        if doc.get("id") not in seen_ids:
            source_chunks.append(doc)
            seen_ids.add(doc.get("id"))

    if keywords:
        for doc in await backend.fulltext_food(keywords, 10):
            if doc.get("id") not in seen_ids:
                source_chunks.append(doc)
                seen_ids.add(doc.get("id"))

    res.source_chunks = source_chunks
    res.timings["source_fetch"] = time.perf_counter() - t0
    res.stats["source_ids"] = len(source_ids)
    return res
