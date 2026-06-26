#!/usr/bin/env python
"""Copy ALL documents from the source food database to food-dflash.

Documents already have embeddings, so no re-embedding is needed.
Existing documents in the destination are skipped (resume-safe).

Usage:
    python scripts/upload_all_to_dflash.py                        # full copy
    python scripts/upload_all_to_dflash.py --dry-run              # count only
    python scripts/upload_all_to_dflash.py --concurrency 30       # faster
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

import yaml
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential

COSMOS_INTERNAL_FIELDS = {"_rid", "_self", "_etag", "_attachments", "_ts"}

CONCURRENCY = 10
PROGRESS_EVERY = 500
MAX_RETRIES = 8


def load_config(path: str = "config_kg_dflash.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cosmos = cfg.get("cosmos", {})
    cosmos["uri"] = os.getenv("COSMOS_ENDPOINT", cosmos.get("uri", ""))
    return cfg


async def fetch_existing_ids(container) -> set[str]:
    """Pre-fetch IDs already in the destination to enable resume."""
    ids: set[str] = set()
    async for row in container.query_items("SELECT c.id FROM c"):
        ids.add(row["id"])
    return ids


async def run(args):
    cfg = load_config(args.config)
    cosmos_cfg = cfg["cosmos"]
    src_cfg = cfg.get("source_cosmos", cosmos_cfg)
    tenant_id = cosmos_cfg.get("tenant_id", "")

    src_cred = AzureCliCredential(tenant_id=src_cfg.get("tenant_id", tenant_id))
    dst_cred = AzureCliCredential(tenant_id=tenant_id)
    src_cosmos = CosmosClient(src_cfg["uri"], credential=src_cred)
    dst_cosmos = CosmosClient(cosmos_cfg["uri"], credential=dst_cred)

    src_container = (src_cosmos
                     .get_database_client(src_cfg["database_name"])
                     .get_container_client("food"))
    dst_container = (dst_cosmos
                     .get_database_client(cosmos_cfg["database_name"])
                     .get_container_client("food"))

    # Count source docs
    src_count = 0
    async for v in src_container.query_items("SELECT VALUE COUNT(1) FROM c"):
        src_count = v
    print(f"Source '{src_cfg['database_name']}/food': {src_count:,} documents")

    # Fetch existing destination IDs for resume
    print("Fetching existing IDs in destination (for resume)...")
    existing_ids = await fetch_existing_ids(dst_container)
    print(f"Destination '{cosmos_cfg['database_name']}/food': {len(existing_ids):,} documents already present")

    if args.dry_run:
        print(f"\nDRY RUN — would copy up to {src_count - len(existing_ids):,} new documents")
        await _cleanup(src_cosmos, dst_cosmos, src_cred, dst_cred)
        return

    # Stream all docs from source
    print(f"\nStreaming documents from source (concurrency={args.concurrency})...")
    sem = asyncio.Semaphore(args.concurrency)
    copied = 0
    skipped = 0
    errors = 0
    total_seen = 0
    t0 = time.time()

    async def upsert_one(doc: dict):
        nonlocal copied, errors
        async with sem:
            for attempt in range(MAX_RETRIES):
                try:
                    await dst_container.upsert_item(doc)
                    copied += 1
                    return
                except Exception as e:
                    err_str = str(e)
                    if "TooManyRequests" in err_str or "429" in err_str:
                        wait = min(2 ** attempt * 0.5, 30)
                        await asyncio.sleep(wait)
                        continue
                    errors += 1
                    if errors <= 10:
                        print(f"  Error upserting {doc.get('id', '?')}: {e}")
                    return
            errors += 1
            if errors <= 10:
                print(f"  Error upserting {doc.get('id', '?')}: max retries exceeded (429)")

    tasks: list[asyncio.Task] = []

    async for item in src_container.query_items("SELECT * FROM c"):
        total_seen += 1
        doc_id = item.get("id", "")

        if doc_id in existing_ids:
            skipped += 1
            if total_seen % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate = copied / max(elapsed, 0.1)
                print(f"  {total_seen:,} read | {copied:,} copied | {skipped:,} skipped | "
                      f"{errors:,} errors | {rate:.0f} upserts/s | {elapsed:.0f}s")
            continue

        clean = {k: v for k, v in item.items() if k not in COSMOS_INTERNAL_FIELDS}
        tasks.append(asyncio.create_task(upsert_one(clean)))

        # Periodically await completed tasks to bound memory
        if len(tasks) >= args.concurrency * 4:
            done = [t for t in tasks if t.done()]
            tasks = [t for t in tasks if not t.done()]
            if not done:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                done = [t for t in tasks if t.done()]
                tasks = [t for t in tasks if not t.done()]

        if total_seen % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            rate = copied / max(elapsed, 0.1)
            print(f"  {total_seen:,} read | {copied:,} copied | {skipped:,} skipped | "
                  f"{errors:,} errors | {rate:.0f} upserts/s | {elapsed:.0f}s")

    # Wait for remaining tasks
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Read:    {total_seen:,}")
    print(f"  Copied:  {copied:,}")
    print(f"  Skipped: {skipped:,} (already existed)")
    print(f"  Errors:  {errors:,}")
    print(f"  Rate:    {copied / max(elapsed, 0.1):.0f} upserts/s")

    # Verify final count
    dst_count = 0
    async for v in dst_container.query_items("SELECT VALUE COUNT(1) FROM c"):
        dst_count = v
    print(f"\nFinal destination count: {dst_count:,} / {src_count:,}")
    if dst_count == src_count:
        print("All documents copied successfully!")
    else:
        print(f"  Gap: {src_count - dst_count:,} documents missing")

    await _cleanup(src_cosmos, dst_cosmos, src_cred, dst_cred)


async def _cleanup(*clients):
    for c in clients:
        try:
            await c.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Copy all docs from food → food-dflash")
    parser.add_argument("--config", default="config_kg_dflash.yaml")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
