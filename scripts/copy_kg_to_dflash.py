#!/usr/bin/env python
"""Copy KG triples and entities from food DB to food-dflash.

Usage:
    python scripts/copy_kg_to_dflash.py
    python scripts/copy_kg_to_dflash.py --dry-run
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

import yaml
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential

COSMOS_INTERNAL_FIELDS = {"_rid", "_self", "_etag", "_attachments", "_ts"}
CONCURRENCY = 100
MAX_RETRIES = 8


def load_config(path: str = "config_kg_dflash.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def copy_container(src_container, dst_container, name: str, dry_run: bool = False):
    """Copy all docs from src to dst, skipping existing."""
    src_count = 0
    async for v in src_container.query_items("SELECT VALUE COUNT(1) FROM c"):
        src_count = v
    print(f"  Source {name}: {src_count:,} docs")

    existing = set()
    async for row in dst_container.query_items("SELECT c.id FROM c"):
        existing.add(row["id"])
    print(f"  Destination {name}: {len(existing):,} already present")

    if dry_run:
        print(f"  DRY RUN — would copy up to {src_count - len(existing):,} docs")
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    copied = skipped = errors = 0
    t0 = time.time()

    async def upsert(doc):
        nonlocal copied, errors
        async with sem:
            for attempt in range(MAX_RETRIES):
                try:
                    await dst_container.upsert_item(doc)
                    copied += 1
                    return
                except Exception as e:
                    if "429" in str(e) or "TooManyRequests" in str(e):
                        await asyncio.sleep(min(2 ** attempt * 0.5, 30))
                        continue
                    errors += 1
                    if errors <= 5:
                        print(f"    Error: {doc.get('id', '?')}: {e}")
                    return
            errors += 1

    tasks = []
    total = 0
    async for item in src_container.query_items("SELECT * FROM c"):
        total += 1
        if item.get("id", "") in existing:
            skipped += 1
        else:
            clean = {k: v for k, v in item.items() if k not in COSMOS_INTERNAL_FIELDS}
            tasks.append(asyncio.create_task(upsert(clean)))

        if len(tasks) >= CONCURRENCY * 4:
            done = [t for t in tasks if t.done()]
            tasks = [t for t in tasks if not t.done()]
            if not done and tasks:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = [t for t in tasks if not t.done()]

        if total % 5000 == 0:
            elapsed = time.time() - t0
            rate = copied / max(elapsed, 0.1)
            print(f"    {total:,} read | {copied:,} copied | {skipped:,} skipped | {rate:.0f}/s")

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - t0
    print(f"  Done {name}: {copied:,} copied, {skipped:,} skipped, {errors:,} errors in {elapsed:.1f}s")
    return copied


async def run(args):
    cfg = load_config()
    tenant_id = cfg["cosmos"]["tenant_id"]
    uri = cfg["cosmos"]["uri"]

    cred = AzureCliCredential(tenant_id=tenant_id)
    cosmos = CosmosClient(uri, credential=cred)

    src_db = cosmos.get_database_client("food")
    dst_db = cosmos.get_database_client("food-dflash")

    print("=" * 60)
    print("  Copy KG: food → food-dflash")
    print("=" * 60)

    for src_name, dst_name in [("kg_triples_food", "kg_triples"), ("kg_entities_food", "kg_entities")]:
        print(f"\n  Copying {src_name} → {dst_name}...")
        src_c = src_db.get_container_client(src_name)
        dst_c = dst_db.get_container_client(dst_name)
        await copy_container(src_c, dst_c, dst_name, dry_run="--dry-run" in sys.argv)

    # Verify
    print("\n  Verification:")
    for name in ["kg_triples", "kg_entities"]:
        c = dst_db.get_container_client(name)
        async for v in c.query_items("SELECT VALUE COUNT(1) FROM c"):
            print(f"    food-dflash/{name}: {v:,}")

    await cosmos.close()
    await cred.close()
    print("\n  Done!")


if __name__ == "__main__":
    asyncio.run(run(sys.argv))
