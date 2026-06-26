#!/usr/bin/env python
"""Set up the food-dflash database: create containers and copy question-relevant
documents from the source food database.

Usage:
    python scripts/setup_food_dflash.py                         # full setup
    python scripts/setup_food_dflash.py --k-per-question 50     # more docs per question
    python scripts/setup_food_dflash.py --skip-copy             # only create containers
    python scripts/setup_food_dflash.py --dry-run               # show what would be copied
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time

import yaml
from azure.cosmos.aio import CosmosClient
from azure.identity.aio import AzureCliCredential

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kg_builder import EmbedClient, embed_sync, COSMOS_INTERNAL_FIELDS

VECTOR_EMBEDDING_POLICY = {
    "vectorEmbeddings": [
        {"path": "/e", "dataType": "float32", "dimensions": 1024, "distanceFunction": "cosine"}
    ]
}

KG_VECTOR_EMBEDDING_POLICY = {
    "vectorEmbeddings": [
        {"path": "/embedding", "dataType": "float32", "dimensions": 1024, "distanceFunction": "cosine"}
    ]
}

FOOD_INDEXING_POLICY = {
    "indexingMode": "consistent",
    "automatic": True,
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}, {"path": "/e/*"}],
    "fullTextIndexes": [
        {"path": "/product_title"}, {"path": "/product_title_translated"},
        {"path": "/brand"}, {"path": "/claims"}, {"path": "/claims_translated"},
        {"path": "/ingredients_translated"}, {"path": "/country_code"},
    ],
    "vectorIndexes": [
        {"path": "/e", "type": "diskANN", "quantizationByteSize": 192, "indexingSearchListSize": 100}
    ],
}

FOOD_FULL_TEXT_POLICY = {
    "defaultLanguage": "en-US",
    "fullTextPaths": [
        {"path": "/product_title", "language": "en-US"},
        {"path": "/product_title_translated", "language": "en-US"},
        {"path": "/brand", "language": "en-US"},
        {"path": "/claims", "language": "en-US"},
        {"path": "/claims_translated", "language": "en-US"},
        {"path": "/ingredients_translated", "language": "en-US"},
        {"path": "/country_code", "language": "en-US"},
    ],
}

KG_INDEXING_POLICY = {
    "indexingMode": "consistent",
    "automatic": True,
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}, {"path": "/embedding/*"}],
    "vectorIndexes": [
        {"path": "/embedding", "type": "diskANN", "quantizationByteSize": 192, "indexingSearchListSize": 100}
    ],
}


def load_dflash_config(path: str = "config_kg_dflash.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cosmos = cfg.get("cosmos", {})
    cosmos["uri"] = os.getenv("COSMOS_ENDPOINT", cosmos.get("uri", ""))
    cosmos["key"] = os.getenv("COSMOS_KEY", cosmos.get("key", ""))
    return cfg


def _az(args: list[str], check: bool = True) -> str:
    result = subprocess.run(["az"] + args, capture_output=True, text=True, timeout=120)
    if check and result.returncode != 0:
        raise RuntimeError(f"az CLI error: {result.stderr.strip()}")
    return result.stdout.strip()


def _write_temp_json(data: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    return path


def create_database_if_needed(account: str, rg: str, db_name: str):
    out = _az(["cosmosdb", "sql", "database", "show",
               "-a", account, "-g", rg, "-n", db_name, "-o", "json"], check=False)
    if out and '"name"' in out:
        print(f"  Database '{db_name}' already exists")
        return
    print(f"  Creating database '{db_name}'...")
    _az(["cosmosdb", "sql", "database", "create", "-a", account, "-g", rg, "-n", db_name])
    print(f"  Database '{db_name}' created")


def create_container_if_needed(
    account: str, rg: str, db_name: str, name: str, partition_key_path: str,
    idx_policy: dict | None = None,
    vec_emb_policy: dict | None = None,
    ft_policy: dict | None = None,
):
    out = _az(["cosmosdb", "sql", "container", "show",
               "-a", account, "-g", rg, "-d", db_name, "-n", name, "-o", "json"], check=False)
    if out and '"name"' in out:
        print(f"  Container '{name}' already exists")
        return

    print(f"  Creating container '{name}' (pk={partition_key_path})...")
    cmd = ["cosmosdb", "sql", "container", "create",
           "-a", account, "-g", rg, "-d", db_name, "-n", name, "-p", partition_key_path]

    tmp_files: list[str] = []
    try:
        if idx_policy:
            p = _write_temp_json(idx_policy); tmp_files.append(p)
            cmd += ["--idx", f"@{p}"]
        if vec_emb_policy:
            p = _write_temp_json(vec_emb_policy); tmp_files.append(p)
            cmd += ["--vector-embeddings", f"@{p}"]
        if ft_policy:
            p = _write_temp_json(ft_policy); tmp_files.append(p)
            cmd += ["--full-text-policy", f"@{p}"]
        _az(cmd)
        print(f"  Container '{name}' created")
    finally:
        for f in tmp_files:
            os.unlink(f)


async def copy_question_relevant_docs(
    src_cosmos: CosmosClient, dst_cosmos: CosmosClient, cfg: dict,
    questions: list[dict], k_per_question: int = 30, dry_run: bool = False,
):
    src_cfg = cfg.get("source_cosmos", cfg["cosmos"])
    dst_cfg = cfg["cosmos"]

    src_db = src_cosmos.get_database_client(src_cfg["database_name"])
    dst_db = dst_cosmos.get_database_client(dst_cfg["database_name"])
    src_container = src_db.get_container_client("food")
    dst_container = dst_db.get_container_client("food")

    embedder = EmbedClient(cfg)
    seen_ids: set[str] = set()
    all_docs: list[dict] = []

    print(f"\n  Retrieving top-{k_per_question} docs per question from source...")
    for qi, q in enumerate(questions):
        q_text = q.get("question_text", "")
        q_id = q.get("question_id", f"q{qi}")
        print(f"\n  [{q_id}] {q_text[:70]}...")

        q_emb = await embedder.embed(q_text)
        sql = ("SELECT TOP @k c, VectorDistance(c.e, @emb) AS score "
               "FROM c ORDER BY VectorDistance(c.e, @emb)")
        docs = []
        async for item in src_container.query_items(
            query=sql, parameters=[{"name": "@k", "value": k_per_question}, {"name": "@emb", "value": q_emb}],
        ):
            docs.append(item)

        new_count = 0
        for item in docs:
            doc = item.get("c") if isinstance(item.get("c"), dict) else item
            doc_id = doc.get("id", "")
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            all_docs.append({k: v for k, v in doc.items() if k not in COSMOS_INTERNAL_FIELDS})
            new_count += 1
        print(f"    Retrieved {len(docs)}, {new_count} new unique (total: {len(all_docs)})")

    print(f"\n  Total unique documents to copy: {len(all_docs)}")
    if dry_run:
        print("  DRY RUN — not copying documents")
        return len(all_docs)

    print(f"\n  Copying {len(all_docs)} documents to '{dst_cfg['database_name']}/food'...")
    t0 = time.time()
    copied = errors = 0
    for i, doc in enumerate(all_docs):
        try:
            await dst_container.upsert_item(doc)
            copied += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"    Error copying doc {doc.get('id', '?')}: {e}")
        if (i + 1) % 50 == 0 or i + 1 == len(all_docs):
            elapsed = time.time() - t0
            print(f"    {i + 1}/{len(all_docs)} ({copied} ok, {errors} errors, {copied / max(elapsed, 0.1):.0f}/s)")

    print(f"  Copy complete: {copied} docs in {time.time() - t0:.1f}s")
    return copied


async def setup(args):
    cfg = load_dflash_config(args.config)
    cosmos_cfg = cfg["cosmos"]
    src_cfg = cfg.get("source_cosmos", cosmos_cfg)
    kg_cfg = cfg.get("kg", {})
    account = cosmos_cfg.get("cosmos_account_name", "divdet")
    rg = cosmos_cfg.get("cosmos_resource_group", "ams-cosmos-db")

    print("=" * 60)
    print("  Food-DFlash Database Setup")
    print("=" * 60)
    print(f"  Source DB:  {src_cfg['database_name']}")
    print(f"  Target DB:  {cosmos_cfg['database_name']}")
    print(f"  Account:    {account}")
    print(f"  Questions:  {args.questions}")
    print(f"  k/question: {args.k_per_question}")
    print()

    print("STEP 1: Create database")
    create_database_if_needed(account, rg, cosmos_cfg["database_name"])

    print("\nSTEP 2: Create containers")
    triples_container = kg_cfg.get("triples_container", "kg_triples")
    entities_container = kg_cfg.get("entities_container", "kg_entities")

    create_container_if_needed(account, rg, cosmos_cfg["database_name"],
        "food", "/country_code", idx_policy=FOOD_INDEXING_POLICY,
        vec_emb_policy=VECTOR_EMBEDDING_POLICY, ft_policy=FOOD_FULL_TEXT_POLICY)
    create_container_if_needed(account, rg, cosmos_cfg["database_name"],
        triples_container, "/pk", idx_policy=KG_INDEXING_POLICY, vec_emb_policy=KG_VECTOR_EMBEDDING_POLICY)
    create_container_if_needed(account, rg, cosmos_cfg["database_name"],
        entities_container, "/pk", idx_policy=KG_INDEXING_POLICY, vec_emb_policy=KG_VECTOR_EMBEDDING_POLICY)

    tenant_id = cosmos_cfg.get("tenant_id", "")
    cred = AzureCliCredential(tenant_id=tenant_id)
    cosmos = CosmosClient(cosmos_cfg["uri"], credential=cred)

    if not args.skip_copy:
        print("\nSTEP 3: Copy question-relevant documents")
        with open(args.questions) as f:
            questions = json.load(f)
        print(f"  Loaded {len(questions)} questions")

        src_cred = AzureCliCredential(tenant_id=src_cfg.get("tenant_id", tenant_id))
        src_cosmos = CosmosClient(src_cfg["uri"], credential=src_cred)
        await copy_question_relevant_docs(src_cosmos, cosmos, cfg, questions,
            k_per_question=args.k_per_question, dry_run=args.dry_run)
        await src_cosmos.close()
        await src_cred.close()
    else:
        print("\nSTEP 3: Skipped (--skip-copy)")

    await cosmos.close()
    await cred.close()

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print(f"  Database:   {cosmos_cfg['database_name']}")
    print(f"  Containers: food, {triples_container}, {entities_container}")
    print()
    print("  Next steps:")
    print("    1. Build KG:  python kg_builder.py --config config_kg_dflash.yaml --question-driven --question-index all")
    print("    2. Test:      python test_food_dflash.py --config config_kg_dflash.yaml")
    print("    3. Benchmark: python kg_query.py --config config_kg_dflash.yaml")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Set up food-dflash database")
    parser.add_argument("--config", default="config_kg_dflash.yaml")
    parser.add_argument("--questions", default="data/food.json")
    parser.add_argument("--k-per-question", type=int, default=30)
    parser.add_argument("--skip-copy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(setup(args))


if __name__ == "__main__":
    main()
