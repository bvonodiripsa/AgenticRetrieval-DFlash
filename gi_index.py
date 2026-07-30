"""In-process Graph Index: GPU vector search, CSR graph traversal, BM25 keyword search.

Drop-in replacement for the Cosmos DB round trips in the retrieval pipeline.
Vectors live on the GPU (~3.8 GB in fp16 for the full corpus); payloads stay in
host memory as Arrow columns and are materialised only for the handful of rows
each query actually returns.

Search is exact — every vector is compared on every query. At this corpus size
that costs ~21 ms for the 1.59M triples, so an approximate index (CAGRA,
Vamana) would save time the pipeline cannot use while giving up recall.

    index = LocalGraphIndex("data/local_index")
    hits  = index.search_entities(q_emb, k=10)
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

_TOKEN = re.compile(r"[a-z0-9]+")

# Payload column -> key expected by the prompt builders, per container.
_ENTITY_MAP = {"n": "name", "t": "description", "r": "relation_count", "d": "source_chunks"}
_TRIPLE_MAP = {"s": "subject", "p": "predicate", "o": "object", "f": "confidence", "d": "source_chunks"}

# Food fields that BM25 indexes, mirroring the fullTextIndexes in my.yaml.
_FT_FIELDS = ("product_title_translated", "ingredients_translated", "claims_translated",
              "product_title", "brand")


class _Table:
    """Arrow-backed payload with GPU-resident vectors."""

    def __init__(self, path: str, name: str, torch, device: str):
        self.name = name
        vecs = np.load(os.path.join(path, f"{name}.vecs.npy"))
        self.vecs = torch.from_numpy(vecs).to(device)
        self.table = pq.read_table(os.path.join(path, f"{name}.payload.parquet"))
        self.cols = {c: self.table.column(c) for c in self.table.column_names}
        self.n = len(self.table)
        if self.n != self.vecs.shape[0]:
            raise ValueError(f"{name}: {self.n} payload rows vs {self.vecs.shape[0]} vectors")

    def row(self, i: int, mapping: dict[str, str] | None) -> dict[str, Any]:
        if mapping is None:
            return {c: self.cols[c][i].as_py() for c in self.cols}
        return {out: self.cols[src][i].as_py() for src, out in mapping.items() if src in self.cols}

    @property
    def nbytes(self) -> int:
        return self.vecs.numel() * self.vecs.element_size()


class LocalGraphIndex:
    def __init__(self, path: str = "data/local_index", device: str = "cuda",
                 enable_bm25: bool = True, verbose: bool = True):
        import torch  # deferred so importing this module stays cheap
        self._torch = torch
        self.path = path
        self.device = device if torch.cuda.is_available() else "cpu"
        t0 = time.perf_counter()

        self.manifest = json.load(open(os.path.join(path, "manifest.json")))
        self.entities = _Table(path, "entities", torch, self.device)
        self.triples = _Table(path, "triples", torch, self.device)
        self.food = _Table(path, "food", torch, self.device)

        csr = np.load(os.path.join(path, "triples.csr.npz"))
        self._fwd_indptr, self._fwd_indices = csr["fwd_indptr"], csr["fwd_indices"]
        self._rev_indptr, self._rev_indices = csr["rev_indptr"], csr["rev_indices"]
        self._vocab: dict[str, int] = json.load(open(os.path.join(path, "triples.vocab.json")))

        self._food_by_id = {v.as_py(): i for i, v in enumerate(self.food.cols["id"])}

        self._bm25 = None
        if enable_bm25:
            self._build_bm25()

        if verbose:
            gb = sum(t.nbytes for t in (self.entities, self.triples, self.food)) / 1e9
            print(f"[gi_index] loaded in {time.perf_counter()-t0:.1f}s on {self.device}: "
                  f"{self.entities.n:,} entities, {self.triples.n:,} triples, "
                  f"{self.food.n:,} food docs, {gb:.2f} GB of vectors", flush=True)

    # ---------------------------------------------------------------- vectors

    def _search(self, tbl: _Table, q: Iterable[float], k: int, mapping):
        torch = self._torch
        qv = torch.as_tensor(np.asarray(q, dtype=np.float32), device=self.device)
        qv = qv / qv.norm().clamp_min(1e-12)
        # Vectors were L2-normalised at export, so a dot product is cosine
        # similarity and Cosmos' VectorDistance ordering is 1 - this.
        scores = (qv.to(tbl.vecs.dtype) @ tbl.vecs.T).float()
        k = min(k, tbl.n)
        vals, idx = torch.topk(scores, k)
        out = []
        for score, i in zip(vals.tolist(), idx.tolist()):
            row = tbl.row(i, mapping)
            # Cosmos VectorDistance with a cosine policy reports similarity,
            # highest first. Match that so downstream code sees one convention.
            row["score"] = score
            out.append(row)
        return out

    def search_entities(self, q_emb, k: int = 10):
        return self._search(self.entities, q_emb, k, _ENTITY_MAP)

    def search_triples(self, q_emb, k: int = 30):
        return self._search(self.triples, q_emb, k, _TRIPLE_MAP)

    def search_food(self, q_emb, k: int = 12):
        return self._search(self.food, q_emb, k, None)

    # ------------------------------------------------------------------ graph

    def triples_for(self, name: str, reverse: bool = False) -> list[dict[str, Any]]:
        """Triples touching `name`.

        Forward edges mirror what Cosmos can serve, since `s` is the partition
        key. Reverse edges are the ones a cross-partition scan makes
        impractical remotely, and they are what makes ingredient-level seed
        entities reachable at all.
        """
        node = self._vocab.get(name)
        if node is None:
            return []
        rows = list(self._fwd_indices[self._fwd_indptr[node]:self._fwd_indptr[node + 1]])
        if reverse:
            rows += list(self._rev_indices[self._rev_indptr[node]:self._rev_indptr[node + 1]])
        return [self.triples.row(int(i), _TRIPLE_MAP) for i in rows]

    def docs_by_id(self, ids: Iterable[str]) -> list[dict[str, Any]]:
        out = []
        for cid in ids:
            i = self._food_by_id.get(cid)
            if i is not None:
                out.append(self.food.row(i, None))
        return out

    # ------------------------------------------------------------------ BM25

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        cols = [self.food.cols[f] for f in _FT_FIELDS if f in self.food.cols]
        docs = []
        for i in range(self.food.n):
            parts = []
            for c in cols:
                v = c[i].as_py()
                if v:
                    parts.append(v if isinstance(v, str) else " ".join(map(str, v)))
            docs.append(_TOKEN.findall(" ".join(parts).lower()))
        self._bm25 = BM25Okapi(docs)

    def fulltext_food(self, keyword: str, k: int = 10) -> list[dict[str, Any]]:
        """Local stand-in for Cosmos FullTextContains over the food container."""
        if self._bm25 is None:
            return []
        toks = _TOKEN.findall(keyword.lower())
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        k = min(k, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [self.food.row(int(i), None) for i in idx if scores[i] > 0]


_INSTANCE: LocalGraphIndex | None = None


def get_index(path: str = "data/local_index", **kw) -> LocalGraphIndex:
    """Process-wide singleton so the 3.8 GB of vectors is loaded once."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LocalGraphIndex(path, **kw)
    return _INSTANCE
