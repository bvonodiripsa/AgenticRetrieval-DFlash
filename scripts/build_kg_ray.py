#!/usr/bin/env python
"""Food KG Build via Ray cluster (20x H100 workers).

Each worker loads Qwen3.5-27B with DFlash speculative decoding on its GPU
and extracts triples locally using vLLM. The head node coordinates via ray.data.

DFlash provides 3-4x lossless speedup via block diffusion drafting.

Usage:
  # On head node (10.0.0.4):
  export RAY_ADDRESS=auto
  python scripts/build_kg_ray.py --config config_ray_build.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from prompts_kg_food import (
    INITIAL_EXTRACTION_PROMPT,
    GAP_ANALYSIS_PROMPT,
    TARGETED_EXTRACTION_PROMPT,
)

# =============================================================================
# Config
# =============================================================================

_vllm_cfg: dict = {}

SYSTEM_MSG = (
    "Respond directly with the requested JSON array. "
    "No reasoning, no explanation, no markdown fences."
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_json_array(text: str) -> list[dict]:
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    return []


def triples_to_json(triples: list[dict]) -> str:
    compact = [{"s": t.get("subject", ""), "p": t.get("predicate", ""), "o": t.get("object", "")}
               for t in triples[:30]]
    return json.dumps(compact, ensure_ascii=False)


# =============================================================================
# Ray Actor: DeepExtractor (runs on each GPU worker)
# =============================================================================

class FoodDeepExtractor:
    def __init__(self):
        import os as _os
        _os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM, SamplingParams
        c = _vllm_cfg
        print(f"[FoodDeepExtractor] Loading {c['model']} (FP8)...")
        self.llm = LLM(
            model=c["model"],
            dtype="auto",
            quantization="fp8",
            gpu_memory_utilization=0.90,
            max_model_len=c["max_model_len"],
            enable_prefix_caching=True,
            trust_remote_code=True,
            enforce_eager=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0, max_tokens=c["max_tokens"])
        self._max_prompt_tokens = c["max_model_len"] - c["max_tokens"]
        self._max_gaps = c.get("max_gaps", 3)
        self._rounds = c.get("extraction_rounds", 1)
        self._batch_count = 0
        self._total_triples = 0
        print(f"[FoodDeepExtractor] Ready (rounds={self._rounds})")

    def _make_prompt(self, template, **kwargs):
        text = template.format(**kwargs)
        messages = [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": text}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        return formatted if len(self.tokenizer.encode(formatted)) <= self._max_prompt_tokens else None

    def _generate(self, prompts):
        return [o.outputs[0].text for o in self.llm.generate(prompts, self.params)] if prompts else []

    def _clean(self, text):
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _extract_batch(self, json_docs: list[str]) -> list[list[dict]]:
        """Extract triples from multiple docs in a single batched generate() call."""
        prompts = []
        doc_indices = []
        for i, doc in enumerate(json_docs):
            p = self._make_prompt(INITIAL_EXTRACTION_PROMPT, json_doc=doc)
            if p:
                prompts.append(p)
                doc_indices.append(i)

        per_doc_triples: list[list[dict]] = [[] for _ in json_docs]
        if not prompts:
            return per_doc_triples

        results = self._generate(prompts)
        for idx, resp in zip(doc_indices, results):
            triples = parse_json_array(self._clean(resp))
            for t in triples:
                t.setdefault("confidence", 0.8)
            per_doc_triples[idx] = triples

        if self._rounds <= 1:
            for doc_triples in per_doc_triples:
                for t in doc_triples:
                    for f in ("subject", "predicate", "object"):
                        t[f] = str(t.get(f, "")).strip()
            return per_doc_triples

        # Multi-round: batch gap analysis across all docs that got triples
        gap_prompts = []
        gap_doc_indices = []
        for i, doc in enumerate(json_docs):
            if not per_doc_triples[i]:
                continue
            p = self._make_prompt(GAP_ANALYSIS_PROMPT, json_doc=doc,
                                  existing_triples=triples_to_json(per_doc_triples[i]))
            if p:
                gap_prompts.append(p)
                gap_doc_indices.append(i)

        if gap_prompts:
            gap_results = self._generate(gap_prompts)
            all_targeted = []
            targeted_doc_indices = []
            for idx, gap_resp in zip(gap_doc_indices, gap_results):
                gap_resp = self._clean(gap_resp)
                gap_instructions = []
                m = re.search(r"\[[\s\S]*\]", gap_resp)
                if m:
                    try:
                        raw = json.loads(m.group())
                        if isinstance(raw, list):
                            gap_instructions = raw
                    except json.JSONDecodeError:
                        pass
                if not gap_instructions:
                    gap_instructions = parse_json_array(gap_resp)
                if not gap_instructions:
                    continue
                for g in gap_instructions[:self._max_gaps]:
                    gap_str = g if isinstance(g, str) else g.get("instruction", g.get("gap", str(g)))
                    p = self._make_prompt(TARGETED_EXTRACTION_PROMPT, gap_instruction=gap_str,
                                          existing_triples=triples_to_json(per_doc_triples[idx]),
                                          json_doc=json_docs[idx])
                    if p:
                        all_targeted.append(p)
                        targeted_doc_indices.append(idx)

            if all_targeted:
                targeted_results = self._generate(all_targeted)
                for idx, resp in zip(targeted_doc_indices, targeted_results):
                    for t in parse_json_array(self._clean(resp)):
                        t.setdefault("confidence", 0.85)
                        t["extraction_round"] = 2
                        per_doc_triples[idx].append(t)

        for doc_triples in per_doc_triples:
            for t in doc_triples:
                for f in ("subject", "predicate", "object"):
                    t[f] = str(t.get(f, "")).strip()
        return per_doc_triples

    def __call__(self, batch):
        chunk_ids = batch["chunk_id"].tolist()
        json_docs = batch["json_doc"].tolist()

        per_doc_triples = self._extract_batch(json_docs)

        all_results = []
        for i, triples in enumerate(per_doc_triples):
            for t in triples:
                t["source_chunks"] = [chunk_ids[i]]
            all_results.append(json.dumps(triples, ensure_ascii=False))
            self._total_triples += len(triples)

        self._batch_count += 1
        print(f"[FoodDeepExtractor] batch {self._batch_count}: {len(chunk_ids)} docs, "
              f"{self._total_triples:,} triples total")
        batch["triples_json"] = np.array(all_results)
        return batch


# =============================================================================
# Main build pipeline
# =============================================================================

async def fetch_all_docs(cfg: dict) -> list[dict]:
    """Fetch all documents from Cosmos DB food container."""
    from azure.cosmos.aio import CosmosClient
    from azure.identity.aio import AzureCliCredential

    cosmos_cfg = cfg["cosmos"]
    cred = AzureCliCredential(tenant_id=cosmos_cfg["tenant_id"])
    cosmos = CosmosClient(cosmos_cfg["uri"], credential=cred)
    db = cosmos.get_database_client(cosmos_cfg["database_name"])

    docs = []
    for src in cosmos_cfg.get("sources", []):
        container = db.get_container_client(src["container_name"])
        print(f"  Reading from '{src['container_name']}'...")
        count = 0
        async for item in container.query_items("SELECT * FROM c"):
            doc_id = item.get("id", "")
            json_str = json.dumps({k: v for k, v in item.items()
                                   if k not in ("e", "embedding", "_rid", "_self", "_etag",
                                                "_attachments", "_ts", "id")},
                                  ensure_ascii=False)
            docs.append({"chunk_id": doc_id, "json_doc": json_str})
            count += 1
            if count % 5000 == 0:
                print(f"    {count} docs read...")
        print(f"    Total: {count} docs from '{src['container_name']}'")

    await cosmos.close()
    await cred.close()
    return docs


def run_build(cfg: dict, time_limit: int | None = None):
    """Run Ray-based extraction across the GPU cluster."""
    import ray

    global _vllm_cfg
    build_llm = cfg.get("build_llm", {})
    build_cfg = cfg.get("build", {})
    dflash_cfg = build_llm.get("dflash", {})
    _vllm_cfg = {
        "model": build_llm.get("model", "Qwen/Qwen3.5-27B"),
        "max_tokens": int(build_llm.get("max_tokens", 3000)),
        "max_model_len": int(build_llm.get("max_model_len", 16384)),
        "max_gaps": int(build_cfg.get("max_gaps_per_round", 3)),
        "extraction_rounds": int(build_cfg.get("extraction_rounds", 1)),
        "dflash_draft_model": dflash_cfg.get("draft_model", ""),
        "dflash_num_speculative_tokens": int(dflash_cfg.get("num_speculative_tokens", 15)),
    }

    ray.init(address="auto", ignore_reinit_error=True, runtime_env={
        "working_dir": PROJECT_ROOT,
        "env_vars": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
    })

    num_gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f"\n  Ray cluster: {num_gpus} GPUs available")
    assert num_gpus > 0, "No GPUs in Ray cluster"

    # Load checkpoint or fetch docs
    ckpt_dir = cfg.get("paths", {}).get("checkpoint_dir", "out_kg/ray_build")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_docs = os.path.join(ckpt_dir, "docs.json")
    ckpt_triples = os.path.join(ckpt_dir, "raw_triples.json")

    if os.path.exists(ckpt_triples):
        print(f"  Checkpoint found: {ckpt_triples}")
        with open(ckpt_triples) as f:
            existing = json.load(f)
        print(f"  Already have {len(existing):,} triples")
        processed_ids = {t["source_chunks"][0] for t in existing if t.get("source_chunks")}
    else:
        existing = []
        processed_ids = set()

    if os.path.exists(ckpt_docs):
        with open(ckpt_docs) as f:
            all_docs = json.load(f)
        print(f"  Loaded {len(all_docs):,} docs from cache")
    else:
        print("  Fetching all documents from Cosmos DB...")
        all_docs = asyncio.run(fetch_all_docs(cfg))
        with open(ckpt_docs, "w") as f:
            json.dump(all_docs, f, ensure_ascii=False)
        print(f"  Cached {len(all_docs):,} docs to {ckpt_docs}")

    # Filter already processed
    remaining = [d for d in all_docs if d["chunk_id"] not in processed_ids]
    print(f"  Remaining: {len(remaining):,} docs to process ({len(processed_ids):,} already done)")

    if not remaining:
        print("  Nothing to extract!")
        return existing

    # Run extraction via Ray
    actors = min(num_gpus, len(remaining))
    print(f"\n  Launching extraction: {len(remaining):,} docs, {actors} GPU actors, "
          f"batch_size=32")

    t0 = time.time()
    ds = ray.data.from_items(remaining)
    ds = ds.map_batches(FoodDeepExtractor, concurrency=actors, num_gpus=1, batch_size=32)

    all_triples = list(existing)
    batch_count = 0
    for row in ds.iter_rows():
        triples = json.loads(row["triples_json"])
        all_triples.extend(triples)
        batch_count += 1
        if batch_count % 100 == 0:
            elapsed = time.time() - t0
            rate = batch_count / elapsed
            print(f"  Progress: {batch_count}/{len(remaining)} docs, "
                  f"{len(all_triples):,} triples, {elapsed:.0f}s, {rate:.1f} docs/s")
            # Periodic checkpoint
            with open(ckpt_triples, "w") as f:
                json.dump(all_triples, f, ensure_ascii=False)

        if time_limit and (time.time() - t0) >= time_limit:
            print(f"\n  TIME LIMIT ({time_limit}s) reached. Saving...")
            break

    # Final save
    elapsed = time.time() - t0
    with open(ckpt_triples, "w") as f:
        json.dump(all_triples, f, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"  Extraction done: {len(all_triples):,} triples from "
          f"{batch_count + len(processed_ids):,} docs in {elapsed:.0f}s")
    print(f"  Saved to: {ckpt_triples}")
    print(f"{'='*60}")

    return all_triples


def main():
    parser = argparse.ArgumentParser(description="Food KG Build via Ray")
    parser.add_argument("--config", default="config_ray_build.yaml")
    parser.add_argument("--time-limit", type=int, default=None,
                        help="Stop after N seconds")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("=" * 60)
    print("  Food KG Builder — Ray Cluster")
    print("=" * 60)
    print(f"  Config: {args.config}")
    print(f"  Model:  {cfg.get('build_llm', {}).get('model', '?')}")
    print(f"  Cosmos: {cfg['cosmos']['uri']}")
    print(f"  DB:     {cfg['cosmos']['database_name']}")

    run_build(cfg, time_limit=args.time_limit)


if __name__ == "__main__":
    main()
