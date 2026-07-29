#!/usr/bin/env python
"""Manual/cron entry point for snapshot_freshness.check_freshness.

Exits 1 if the snapshot is stale (or missing), 0 if it matches live Cosmos.
Same check the server runs at startup on the background auto-build/hot-swap
path; this just lets you run it standalone without starting the app.

Usage:
    python scripts/check_snapshot_freshness.py --config my.yaml --index data/local_index
"""
import argparse
import asyncio
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snapshot_freshness import check_freshness  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="my.yaml")
    ap.add_argument("--index", default="data/local_index")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    report = await check_freshness(cfg, args.index)

    print(f"Snapshot: {args.index}")
    print(f"Built at: {report.get('built_at')}")
    if report.get("reason"):
        print(f"Reason:   {report['reason']}")
    for name, stats in report["containers"].items():
        flag = "OK" if stats["count_match"] and not stats["stale_by_ts"] else "STALE"
        print(f"  {name:10s} [{flag}]  snapshot={stats['snapshot_count']:,}  "
              f"live={stats['live_count']:,}  newer_writes={stats['stale_by_ts']}")
    if report["unrecorded_containers"]:
        print(f"  Unrecorded (on disk, not in manifest counts): "
              f"{', '.join(report['unrecorded_containers'])}")

    print()
    if report["stale"]:
        print("STALE -- rebuild with scripts/build_local_index.py")
        print(json.dumps(report, indent=2))
        return 1
    print("Fresh.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
