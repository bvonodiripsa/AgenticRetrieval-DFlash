#!/usr/bin/env python
"""Migrate Graph Index data from full-field-name containers to 1-letter-field containers.

Source:  kg_triples_food (pk=/pk)  →  triples (pk=/s)
         kg_entities_food (pk=/pk) →  entities (pk=/id)

Field mapping:
  Triples:  subject→s, predicate→p, object→o, confidence→f,
            confirmations→n, source_chunks→d, embedding→e
  Entities: name→n, description→t, relation_count→r,
            source_chunks→d, embedding→e

No re-embedding — vectors are copied as-is.

Usage:
    python scripts/migrate_gi_schema.py --config config_kg_dflash.yaml
    python scripts/migrate_gi_schema.py --config config_kg_dflash.yaml --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from azure.cosmos.aio import CosmosClient
from azure.cosmos import PartitionKey
from azure.identity.aio import AzureCliCredential


NEW_TRIPLES = "triples"
NEW_ENTITIES = "entities"
BATCH_CONCURRENCY = 40


async def create_container_if_missing(db, name: str, pk_path: str, dims: int = 1024):
    """Create a container with vector index on /e if it doesn't exist."""
    try:
        await db.get_container_client(name).read()
        print(f"  Container '{name}' already exists")
        return
    except Exception as e:
        if "NotFound" not in str(e) and "404" not in str(e):
            raise

    print(f"  Creating container '{name}' (pk={pk_path})...")

    vector_embedding_policy = {
        "vectorEmbeddings": [{
            "path": "/e",
            "dataType": "float32",
            "dimensions": dims,
            "distanceFunction": "cosine",
        }]
    }

    indexing_policy = {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": f"{pk_path}/?"}] if pk_path != "/id" else [],
        "excludedPaths": [{"path": "/*"}, {"path": '/"_etag"/?'}],
        "vectorIndexes": [{
            "path": "/e",
            "type": "diskANN",
            "quantizationByteSize": 192,
            "indexingSearchListSize": 100,
        }],
    }

    from azure.cosmos import ThroughputProperties
    throughput = ThroughputProperties(auto_scale_max_throughput=4000)

    await db.create_container(
        id=name,
        partition_key=PartitionKey(path=pk_path),
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy,
        offer_throughput=throughput,
    )
    print(f"  Created '{name}'")


def remap_triple(doc: dict) -> dict:
    return {
        "id": doc["id"],
        "s": doc.get("subject", ""),
        "p": doc.get("predicate", ""),
        "o": doc.get("object", ""),
        "f": doc.get("confidence", 0.8),
        "n": doc.get("confirmations", 1),
        "d": doc.get("source_chunks", []),
        "e": doc.get("embedding", []),
    }


def remap_entity(doc: dict) -> dict:
    return {
        "id": doc["id"],
        "n": doc.get("name", ""),
        "t": doc.get("description", ""),
        "r": doc.get("relation_count", 0),
        "d": doc.get("source_chunks", []),
        "e": doc.get("embedding", []),
    }


async def copy_container(
    db,
    src_name: str,
    dst_name: str,
    remap_fn,
    dry_run: bool = False,
):
    """Stream docs from src, remap fields, write concurrently to dst."""
    src = db.get_container_client(src_name)
    dst = db.get_container_client(dst_name)

    print(f"\n  Streaming from '{src_name}' → '{dst_name}'...", flush=True)
    t0 = time.time()
    sem = asyncio.Semaphore(BATCH_CONCURRENCY)
    read_count = 0
    written = 0
    errors = 0
    lock = asyncio.Lock()
    pending: list[asyncio.Task] = []
    sample_printed = False

    async def _upsert(remapped):
        nonlocal written, errors
        async with sem:
            for attempt in range(5):
                try:
                    await dst.upsert_item(remapped)
                    async with lock:
                        written += 1
                        if written % 2000 == 0:
                            elapsed = time.time() - t0
                            rate = written / max(elapsed, 0.1)
                            print(f"    {written:,} written ({rate:.0f}/s)", flush=True)
                    return
                except Exception as e:
                    if "429" in str(e) or "TooManyRequests" in str(e):
                        await asyncio.sleep(1.0 * (attempt + 1))
                    else:
                        async with lock:
                            errors += 1
                            if errors <= 5:
                                print(f"    ERROR on doc: {e}", flush=True)
                        return

    async for item in src.query_items(query="SELECT * FROM c", max_item_count=500):
        read_count += 1
        remapped = remap_fn(item)

        if not sample_printed:
            sample_no_vec = {k: v for k, v in remapped.items() if k != "e"}
            print(f"  Sample remapped: {json.dumps(sample_no_vec, indent=2, ensure_ascii=False)}", flush=True)
            sample_printed = True

        if read_count % 5000 == 0:
            print(f"    Read {read_count:,} ({time.time() - t0:.1f}s)", flush=True)

        if not dry_run:
            pending.append(asyncio.create_task(_upsert(remapped)))
            if len(pending) >= 500:
                await asyncio.gather(*pending)
                pending.clear()

    if pending:
        await asyncio.gather(*pending)

    elapsed = time.time() - t0
    if dry_run:
        print(f"  [DRY RUN] {read_count:,} docs read in {elapsed:.1f}s", flush=True)
    else:
        rate = written / max(elapsed, 0.1)
        print(f"  Done: {written:,} written, {errors} errors in {elapsed:.1f}s ({rate:.0f}/s)", flush=True)
    return read_count


async def main():
    parser = argparse.ArgumentParser(description="Migrate Graph Index to 1-letter field schema")
    parser.add_argument("--config", default="config_kg_dflash.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Read and remap but don't write")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cosmos_cfg = cfg["cosmos"]
    db_name = cosmos_cfg["database_name"]

    cred = AzureCliCredential(tenant_id=cosmos_cfg["tenant_id"])
    cosmos = CosmosClient(cosmos_cfg["uri"], credential=cred)
    db = cosmos.get_database_client(db_name)

    kg = cfg.get("kg", {})
    old_triples = kg.get("triples_container", "kg_triples_food")
    old_entities = kg.get("entities_container", "kg_entities_food")

    print("=" * 60)
    print("Graph Index Schema Migration")
    print(f"  Database:        {db_name}")
    print(f"  Source triples:  {old_triples} → {NEW_TRIPLES}")
    print(f"  Source entities: {old_entities} → {NEW_ENTITIES}")
    print(f"  Dry run:         {args.dry_run}")
    print("=" * 60)

    print("\nStep 1: Verify containers exist")
    try:
        await db.get_container_client(NEW_TRIPLES).read()
        print(f"  '{NEW_TRIPLES}' exists")
    except Exception:
        print(f"  '{NEW_TRIPLES}' not found — create it first via Azure CLI")
        return
    try:
        await db.get_container_client(NEW_ENTITIES).read()
        print(f"  '{NEW_ENTITIES}' exists")
    except Exception:
        print(f"  '{NEW_ENTITIES}' not found — create it first via Azure CLI")
        return

    print("\nStep 2: Copy triples")
    n_triples = await copy_container(db, old_triples, NEW_TRIPLES, remap_triple, args.dry_run)

    print("\nStep 3: Copy entities")
    n_entities = await copy_container(db, old_entities, NEW_ENTITIES, remap_entity, args.dry_run)

    print("\n" + "=" * 60)
    print(f"Migration complete: {n_triples:,} triples + {n_entities:,} entities")
    print("=" * 60)

    await cosmos.close()
    await cred.close()


if __name__ == "__main__":
    asyncio.run(main())
