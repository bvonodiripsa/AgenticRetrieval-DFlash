#!/usr/bin/env python
"""Copy food container from divdet-provisioned to divdet-sweden."""
import asyncio
import json
import random
import sys
import time

from azure.cosmos.aio import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError
from azure.identity.aio import AzureCliCredential

SRC_URI = "https://divdet-provisioned.documents.azure.com:443/"
DST_URI = "https://divdet-sweden.documents.azure.com:443/"
TENANT_ID = "43083d15-7273-40c1-b7db-39efd9ccc17a"
DB = "food"
CONTAINER = "food"
CONCURRENCY = 100
MAX_RETRIES = 10


async def main():
    cred = AzureCliCredential(tenant_id=TENANT_ID)
    src = CosmosClient(SRC_URI, credential=cred)
    dst = CosmosClient(DST_URI, credential=cred)

    src_c = src.get_database_client(DB).get_container_client(CONTAINER)
    dst_c = dst.get_database_client(DB).get_container_client(CONTAINER)

    print(f"Copying {CONTAINER} from divdet-provisioned -> divdet-sweden")
    print(f"Concurrency: {CONCURRENCY}")
    sys.stdout.flush()

    sem = asyncio.Semaphore(CONCURRENCY)
    uploaded = 0
    errors = 0
    throttles = 0
    t0 = time.time()
    pending = set()

    async def _upsert(doc):
        nonlocal uploaded, errors, throttles
        async with sem:
            for attempt in range(MAX_RETRIES):
                try:
                    await dst_c.upsert_item(doc)
                    uploaded += 1
                    return
                except CosmosHttpResponseError as e:
                    if e.status_code == 429:
                        throttles += 1
                        wait = max(float(e.headers.get("x-ms-retry-after-ms", 0)) / 1000,
                                   0.5 * (2 ** attempt)) + random.uniform(0, 0.3)
                        await asyncio.sleep(wait)
                    else:
                        errors += 1
                        if errors <= 5:
                            print(f"  ERR {e.status_code}: {str(e.message)[:120]}")
                        return

    read_count = 0
    async for doc in src_c.read_all_items():
        read_count += 1
        for k in ("_rid", "_self", "_etag", "_attachments", "_ts"):
            doc.pop(k, None)

        task = asyncio.create_task(_upsert(doc))
        pending.add(task)
        task.add_done_callback(pending.discard)

        if len(pending) >= CONCURRENCY * 2:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        if read_count % 2000 == 0:
            elapsed = time.time() - t0
            rate = uploaded / max(elapsed, 1)
            print(f"  read={read_count:,} uploaded={uploaded:,} rate={rate:.0f}/s "
                  f"throttles={throttles} errors={errors} elapsed={elapsed:.0f}s")
            sys.stdout.flush()

    if pending:
        await asyncio.gather(*pending)

    elapsed = time.time() - t0
    print(f"\nDONE: {uploaded:,} food docs in {elapsed:.0f}s "
          f"({uploaded/max(elapsed,1):.0f}/s, {errors} errors, {throttles} throttles)")

    await src.close()
    await dst.close()
    await cred.close()


if __name__ == "__main__":
    asyncio.run(main())
