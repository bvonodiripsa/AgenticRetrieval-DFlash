#!/usr/bin/env python
"""Quick validation: run KG extraction on a small batch to compare Qwen3.5-27B + DFlash
against the Qwen2.5-32B baseline.

Usage:
    # Start vLLM with DFlash first (see PROGRESS.md), then:
    python test_dflash.py --config my.yaml --num-docs 20
"""
import argparse
import asyncio
import json
import re
import sys
import time

import yaml
from openai import AsyncOpenAI

from prompts_kg_food import INITIAL_EXTRACTION_PROMPT

SYSTEM_MSG = (
    "Respond directly with the requested JSON array. "
    "No reasoning, no explanation, no markdown fences."
)


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


async def fetch_docs(cfg: dict, limit: int) -> list[dict]:
    from azure.cosmos.aio import CosmosClient
    from azure.identity.aio import AzureCliCredential

    cosmos_cfg = cfg["cosmos"]
    cred = AzureCliCredential(tenant_id=cosmos_cfg["tenant_id"])
    cosmos = CosmosClient(cosmos_cfg["uri"], credential=cred)
    db = cosmos.get_database_client(cosmos_cfg["database_name"])
    src = cosmos_cfg["sources"][0]
    container = db.get_container_client(src["container_name"])

    docs = []
    async for item in container.query_items(
        f"SELECT TOP {limit} * FROM c", enable_cross_partition_query=True
    ):
        json_str = json.dumps(
            {k: v for k, v in item.items()
             if k not in ("e", "embedding", "_rid", "_self", "_etag",
                          "_attachments", "_ts", "id")},
            ensure_ascii=False,
        )
        docs.append({"id": item.get("id", ""), "json_doc": json_str})
    await cosmos.close()
    await cred.close()
    return docs


async def extract_triples(client: AsyncOpenAI, model: str, doc: dict) -> tuple[list, float]:
    prompt = INITIAL_EXTRACTION_PROMPT.format(json_doc=doc["json_doc"])
    t0 = time.time()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    elapsed = time.time() - t0
    text = resp.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    triples = parse_json_array(text)
    return triples, elapsed


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="my.yaml")
    parser.add_argument("--num-docs", type=int, default=20)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    llm_cfg = cfg.get("llm", {})
    model = llm_cfg.get("model", "Qwen/Qwen3.5-27B")
    client = AsyncOpenAI(
        base_url=llm_cfg.get("endpoint", "http://localhost:8000/v1"),
        api_key=llm_cfg.get("api_key") or "dummy",
        timeout=300.0,
    )

    print("=" * 60)
    print(f"  DFlash Validation Test")
    print("=" * 60)
    print(f"  Model:    {model}")
    print(f"  Endpoint: {llm_cfg.get('endpoint')}")
    print(f"  Docs:     {args.num_docs}")

    print(f"\n  Fetching {args.num_docs} docs from Cosmos DB...")
    docs = await fetch_docs(cfg, args.num_docs)
    print(f"  Got {len(docs)} docs")

    print(f"\n  Extracting triples...")
    total_triples = 0
    total_time = 0.0
    parse_ok = 0
    parse_fail = 0

    for i, doc in enumerate(docs):
        triples, elapsed = await extract_triples(client, model, doc)
        total_triples += len(triples)
        total_time += elapsed
        if triples:
            parse_ok += 1
        else:
            parse_fail += 1

        valid = all(t.get("subject") and t.get("predicate") and t.get("object") for t in triples)
        status = f"{len(triples)} triples" + ("" if valid else " [SCHEMA WARN]")
        print(f"    [{i+1}/{len(docs)}] {doc['id'][:30]:30s} {elapsed:.1f}s  {status}")

    print(f"\n{'=' * 60}")
    print(f"  Results")
    print(f"{'=' * 60}")
    print(f"  Documents:      {len(docs)}")
    print(f"  Total triples:  {total_triples}")
    print(f"  Avg per doc:    {total_triples / max(len(docs), 1):.1f}")
    print(f"  Parse success:  {parse_ok}/{len(docs)} ({100*parse_ok/max(len(docs),1):.0f}%)")
    print(f"  Total time:     {total_time:.1f}s")
    print(f"  Avg per doc:    {total_time / max(len(docs), 1):.2f}s")
    print(f"  Throughput:     {len(docs) / max(total_time, 0.1):.1f} docs/s")
    print(f"{'=' * 60}")

    if parse_fail > 0:
        print(f"\n  WARNING: {parse_fail} docs returned no triples — check prompts/model compat")


if __name__ == "__main__":
    asyncio.run(main())
