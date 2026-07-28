# Progress and H100 handoff

Branch `feat/gpu-graph-index` @ `8f3a65a`. Written 2026-07-27 after a day on
the GB10 dev box. Purpose of this file: pick the work up tomorrow on the
2x H100 machine without rediscovering anything.

---

## What was done

The Graph Index moved out of Cosmos DB and into GPU memory. Retrieval was 41%
of the measured 6.40s end-to-end on H100, and it was issuing 14 network queries
per question (25 on the web path) for a corpus that is only 3.75 GB of vectors
in fp16.

Five things landed:

| Step | What | Where |
|---|---|---|
| 0 | One retrieval implementation, two backends | `retrieval.py` |
| 1 | Cosmos to local snapshot exporter | `scripts/build_local_index.py` |
| 2 | GPU vector search + CSR graph traversal | `gi_index.py` |
| 3 | `index.mode` config switch | `my.yaml`, `config.yaml.example` |
| 4 | Local BM25 replacing `FullTextContains` | `gi_index.py` |
| 5 | Embedder moved to GPU | `gi_builder.py` |

Cosmos remains the system of record. The snapshot is a read-only copy and must
be rebuilt whenever the Graph Index changes.

---

## Measured on GB10

Full detail in `results/baseline_gb10.md` and `results/optimization_gb10.md`.

| Stage | Cosmos (Sweden) | Local GPU |
|---|---|---|
| embed | 0.030s | 0.030s |
| entity_search | 6.677s | 0.004s |
| graph_traversal | 11.769s | 0.019s |
| source_fetch | 12.113s | 0.002s |
| **retrieval total** | **30.559s** | **0.025s** |

Embedder separately: 202ms on CPU, 30ms on GPU.

**Do not quote the 1,270x.** Most of it is cross-continental latency to Sweden
Central that the co-located H100 box does not pay. The number to publish is
whatever tomorrow's H100 run produces.

### Two findings worth keeping

**Exact search beats the Cosmos index.** Agreement was only 0.538 Jaccard, so
it got checked: every hit exact search found and diskANN missed scores strictly
higher (>= 0.6123) than every hit diskANN found and exact search missed
(<= 0.6118). Shared hits agree to four decimals, so the snapshot is faithful.
`quantizationByteSize: 192` compresses 4096-byte vectors 21x and loses real
neighbours. Local search is both faster and more accurate.

**Graph traversal was returning nothing.** Every baseline run reported
`0 PK + 30 vec`. Triple subjects are product names, but entity vector search
returns ingredient-level entities that appear only as objects, so forward-only
traversal matched zero triples. The Graph Index was behaving as a plain vector
index. CSR stores reverse adjacency for free; Cosmos cannot, because `o` is not
the partition key. `polysorbate 20` goes from 0 triples to 192, and the
pipeline from `0 PK + 30 vec -> 30 triples` to `12 PK + 30 vec -> 40 triples`.

---

## Expected on H100

Against the README's co-located baseline:

| | Baseline | Predicted |
|---|---|---|
| Retrieval | 2.94s | ~0.1s |
| LLM (DFlash) | 3.46s | 3.46s unchanged |
| **End to end** | **6.40s** | **~4.6s (1.8x)** |

It is not larger because `entity_search` on the web path is gated by the
`_llm_expand_keywords` LLM call, not by Cosmos — see "Open questions" below.

### Memory

The index needs ~4.5 GB (3.75 GB of vectors plus workspace and CUDA context).
`--gpu-memory-utilization 0.92` leaves 8% of 160 GB free, so **12.8 GB**. It
fits with room to spare and needs no vLLM config change. If more headroom is
wanted, dropping to `0.89` frees another 4.8 GB and costs ~10% of a KV cache
the README sizes at 50 GB.

Only the float vectors go on GPU. Triple strings, entity descriptions, food
document text and the CSR arrays live in host RAM, and the VM has 1.9 TB.
Because search is a single GEMV, pin the index to GPU 0 rather than sharding.

**Unresolved:** `README.md` says the VM is `ND96isr_H100_v5` (2x H100 80GB)
while `SETUP.md` and `BENCHMARKS.md` say `Standard_NC80adis_H100_v5`
(2x H100 NVL 96GB). Numbers above assume the smaller. Confirm with
`nvidia-smi` before sizing anything precisely.

---

## H100 runbook

### 1. Get the branch

```bash
git fetch origin && git checkout feat/gpu-graph-index
```

### 2. Dependencies

These were missing on GB10 and are not all in `requirements-web.txt`:

```bash
.venv/bin/pip install aiohttp transformers pyarrow pandas rank_bm25
sudo apt install -y python3.12-dev     # Python.h, required by triton for GPU embedding
```

### 3. Config

`my.yaml` is gitignored and does not travel. Copy the `index:` block out of
`config.yaml.example` into the H100's `my.yaml` and set `mode: "local"`. The
example ships `mode: "cosmos"` so a plain checkout changes nothing.

```yaml
index:
  mode: "local"
  snapshot_path: "data/local_index"
  device: "cuda"
  enable_bm25: true
  reverse_edges: true
```

### 4. Build the snapshot

```bash
python scripts/build_local_index.py --config my.yaml --out data/local_index
```

Took 9.7 min from the US pulling 1.83M docs at ~3,600 docs/s across 20
partitions. Co-located in Sweden Central it should be considerably faster.
Writes ~3.4 GB of `.npy`, ~73 MB of parquet, and `triples.csr.npz`. The
directory is gitignored.

### 5. Baseline, before switching

With `index.mode: cosmos`, capture the current numbers on this hardware so the
comparison is same-machine rather than against the README:

```bash
python benchmark_pipeline.py          # 3 runs, as done for GB10
```

### 6. Compare

```bash
python benchmark_compare.py --config my.yaml --runs 3 --device cuda
```

Prints Cosmos, local forward-only (parity with baseline) and local with reverse
edges, plus a correctness check. Watch for:

- `entity quality exact search wins ...` — anything else means the snapshot is
  stale or misaligned, not a recall tradeoff
- `N PK + 30 vec` where N > 0 on the reverse-edges row
- Jaccard well below 1.0 is expected and fine given the above

### 7. Web app

`api.py::_dflash_answer` is converted. The two streaming paths are not — see
below. Start vLLM per `SETUP.md` and confirm the local index loads alongside it
without OOM.

---

## Not done

**`_stream_dflash_sse` and `_stream_gi_sse` in `api.py`** still carry their own
copies of the retrieval logic and still query Cosmos directly. They interleave
SSE progress events between stages, so consolidating them changes the
granularity of what the web UI shows mid-query. That needs the UI exercised
against a running LLM to verify, which was not possible on GB10 (no LLM
served). Until they are converted, the streaming web paths get no speedup.

**No end-to-end LLM measurement on GB10.** Nothing was served on `:8000`, so
only retrieval stages were measured. Every optimization here is retrieval-side,
so this does not affect the result, but the 6.40s to 4.6s figure is a
projection until tomorrow.

**Snapshot freshness has no guard.** If the Graph Index is rebuilt in Cosmos,
the local snapshot silently goes stale. `manifest.json` records `built_at` but
nothing checks it.

---

## Open questions

**The keyword-expansion LLM call is now the retrieval bottleneck.** On the web
path `entity_search` is `asyncio.gather(_es(), _llm_expand_keywords(...))`, so
its 0.84s is gated by an LLM round trip, not by Cosmos. Local search makes the
vector half 4ms and the LLM half unchanged. Dropping that call would take the
projection from ~4.6s to ~3.7s, and local BM25 may make it unnecessary since it
existed to compensate for weak keyword matching. Worth A/B testing tomorrow.

**`max_hops` can now be raised.** It is pinned at 1 in `my.yaml` because each
extra hop cost another wave of 10 Cosmos queries. Against CSR a hop is
microseconds. Try 2 or 3 and see whether answer quality improves.

**Whether to keep Cosmos in the loop at all for reads.** Currently it stays as
system of record and the snapshot is rebuilt manually. An incremental update
path via the Cosmos change feed would remove the staleness problem.

---

## Environment notes

Things that cost time on GB10 and will likely recur:

- `aiohttp` is required by `azure-cosmos`' async transport but is not in
  `requirements-web.txt`. Failure is `ModuleNotFoundError: No module named
  'aiohttp'` at client construction.
- `AzureCliCredential` shells out to `az`, so `az` must be on `PATH`. If the
  CLI is in its own venv, symlink it: `ln -sf ~/.azcli-venv/bin/az ~/.local/bin/az`.
- `transformers` resolves to 5.x, which renamed `torch_dtype` to `dtype`. The
  old kwarg still works but warns.
- GPU embedding needs `python3.12-dev`; without `Python.h` triton cannot
  compile its CUDA helper and the model silently stays on CPU at 202ms.
- GitHub push from GB10 required an SSH key. The OAuth device flow fails with
  "you don't have permissions to access this resource" when the browser is
  signed into an NVIDIA Enterprise Managed User account.
