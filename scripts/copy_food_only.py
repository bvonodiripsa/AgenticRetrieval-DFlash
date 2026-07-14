#!/usr/bin/env python
"""Copy food container from serverless divdet to provisioned divdet-provisioned."""

from __future__ import annotations

import asyncio
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential

SRC_URI = "https://divdet.documents.azure.com:443/"
DST_URI = "https://divdet-provisioned.documents.azure.com:443/"
TENANT_ID = "43083d15-7273-40c1-b7db-39efd9ccc17a"
DB_NAME = "food"
WRITE_CONCURRENCY = 100


async def main():
    cred = AzureCliCredential(tenant_id=TENANT_ID)
    src_cosmos = CosmosClient(SRC_URI, credential=cred)
    dst_cosmos = CosmosClient(DST_URI, credential=cred)

    src = src_cosmos.get_database_client(DB_NAME).get_container_client("food")
    dst = dst_cosmos.get_database_client(DB_NAME).get_container_client("food")

    print("Copying food: divdet → divdet-provisioned", flush=True)

    t0 = time.time()
    sem = asyncio.Semaphore(WRITE_CONCURRENCY)
    read_count = 0
    written = 0
    errors = 0
    lock = asyncio.Lock()
    pending: list[asyncio.Task] = []

    async def _upsert(doc):
        nonlocal written, errors
        for k in ("_rid", "_self", "_etag", "_attachments", "_ts"):
            doc.pop(k, None)
        async with sem:
            for attempt in range(10):
                try:
                    await dst.upsert_item(doc)
                    async with lock:
                        written += 1
                        if written % 2000 == 0:
                            elapsed = time.time() - t0
                            rate = written / max(elapsed, 0.1)
                            print(f"  {written:,} written ({rate:.0f}/s, {elapsed/60:.1f}m)", flush=True)
                    return
                except Exception as e:
                    if "429" in str(e) or "TooManyRequests" in str(e):
                        await asyncio.sleep(0.3 * (attempt + 1))
                    else:
                        async with lock:
                            errors += 1
                            if errors <= 5:
                                print(f"  ERROR: {e}", flush=True)
                        return

    async for item in src.query_items(query="SELECT * FROM c", max_item_count=1000):
        read_count += 1
        if read_count % 5000 == 0:
            print(f"  Read {read_count:,} ({time.time() - t0:.1f}s)", flush=True)

        pending.append(asyncio.create_task(_upsert(item)))
        if len(pending) >= 1000:
            await asyncio.gather(*pending)
            pending.clear()

    if pending:
        await asyncio.gather(*pending)

    elapsed = time.time() - t0
    rate = written / max(elapsed, 0.1)
    print(f"DONE: {written:,} written, {errors} errors in {elapsed/60:.1f}m ({rate:.0f}/s)", flush=True)

    await src_cosmos.close()
    await dst_cosmos.close()
    await cred.close()


if __name__ == "__main__":
    asyncio.run(main())
