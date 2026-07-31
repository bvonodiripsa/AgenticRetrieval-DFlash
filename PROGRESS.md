# Progress and H100 handoff

Branch `feat/gpu-graph-index`. Written 2026-07-27 after a day on the GB10 dev
box; updated 2026-07-28 with a full day of real H100 validation (see
`results/h100_comparison.md`); updated again 2026-07-29 with a second GB10
pass (see "2026-07-29: second GB10 pass") and then **with the web UI put in
front of real users on H100** (see "2026-07-29: H100 web app + UI
duplicate-submission fix"); updated again 2026-07-30 with an embedding-pooling
bug fix found via a baseline-vs-DFlash quality comparison (see "2026-07-30:
embedding pooling fix").

## Where things actually stand (end of 2026-07-30)

- **Fixed a retrieval-quality bug that was silently degrading every vector
  search in the local index.** The in-process embedder mean-pooled
  `Qwen3-Embedding-0.6B`'s hidden states instead of using the last-token
  pooling the model is actually trained for, which let semantically unrelated
  documents (chicken donuts, a hair-gel product) score higher than the
  genuinely relevant match for a running-snack query. Fixed the pooling,
  re-embedded all 1.83M local-index vectors, and validated against all 10
  predefined questions plus a direct baseline comparison. Full writeup:
  `EMBEDDING_FIX.md`.

- **The web app is live on H100 and one real-usage bug got found and fixed:
  duplicate form submissions from the UI, not backend slowness.** A user
  pressing Enter and then also clicking "Ask" fired two concurrent
  `/v1/ask/stream` requests; retrieval blocks the single asyncio event loop
  (`torch.topk`, BM25), so the second request stalled behind the first
  instead of running in parallel — presenting as "first question 4s, second
  question waits ~20s then shows 3.3s." Fixed with a re-entrancy guard in
  `static/index.html`'s `doAsk()`. See "2026-07-29: H100 web app + UI
  duplicate-submission fix" below. Confirmed fixed by the user.
- **Graph traversal had a second, bigger bug than the one fixed on
  2026-07-27, found while investigating why `max_hops` seemed to do nothing.**
  With `reverse_edges: true`, seed entities found by vector search are
  frequently claims/tags ("post-workout", "high protein snack") rather than
  product names — they were discovered *as objects*. The hop-2 frontier logic
  only ever looked at `t["object"]`, which for an object-discovered seed is
  just the seed itself (already visited), so hop 2 silently found nothing,
  on every question, regardless of `max_hops`. Fixed to consider both triple
  endpoints as candidate frontier. Measured effect: **PK triples per query
  ~11 -> ~184 (16x)** for +0.3ms of `graph_traversal` (still ~20ms total,
  still dominated by the vector search, not the traversal). Validated
  end-to-end against the real streaming endpoint: the "vegan breakfast
  options" query used as a running example throughout this doc went from
  `0 PK + 30 vec` in the original baseline to `160 PK + 30 vec` today. See
  "2026-07-29: second GB10 pass" below.
- **The other two 2026-07-28 "Not done" items are also closed**: a
  background hot-swap so `index.mode: local` never blocks a request on
  loading the snapshot, and a snapshot-freshness check against live Cosmos.
  Both are local-only changes, fully testable without H100 or an LLM — see
  below.
- **H100 state as of the end of 2026-07-28**: retrieval optimization fully
  validated on real hardware (2.06s -> 0.003s), streaming endpoint fixed to
  use the local index and real token streaming (ttft ~0.48-0.50s). vLLM and
  `api.py` were shut down at the end of that session to avoid idle cost.
  2026-07-29 restarted both (picking up the second GB10 pass's traversal fix,
  hot-swap and freshness guard — all shared via `retrieval.py`), rebound
  `api.py` to `--host 0.0.0.0` so the public IP actually reaches it (was
  `127.0.0.1`-only), and put the web UI in front of a real user, which
  surfaced and fixed the duplicate-submission bug above.

- **Retrieval optimization: fully validated, real hardware, real load.**
  2.06s (Cosmos) -> 0.003s (local index), matching the GB10 prediction almost
  exactly. Same correctness signature (exact search beats Cosmos's diskANN),
  same traversal fix (`0 PK -> 12 PK` with reverse edges).
- **End-to-end impact is real but smaller than the naive comparison
  suggests, and the confound is now isolated.** Holding graph-traversal
  content constant (comparing Cosmos forward-only against local forward-only,
  same LLM input either way), the index alone is worth **~2.1s / ~1.55x**
  (5.98s -> 3.87s). Turning reverse edges back on costs ~0.75s back (richer
  prompt, longer answer) — a real, separate, correctness-vs-latency tradeoff,
  not a flaw in the index. Caveat: LLM decode itself has ~1.1s of run-to-run
  variance on nominally identical calls, so both of those numbers are
  single-sample estimates, not tight ones. **N=5+ repeats per config is the
  first thing to do tomorrow if these numbers need to be defensible.**
- **The production streaming endpoint (`/v1/ask/stream`) is now actually
  fixed and using the local index.** It had its own hardcoded Cosmos queries
  all day yesterday and today until this evening — none of the above ever
  reached real users regardless of `index.mode`. Also switched from fake
  streaming (buffer full completion, chunk the string) to real `stream=True`.
  Measured `ttft` (time-to-first-token) ~0.48-0.50s vs the ~4-5s the old fake
  path made users wait before seeing anything. Validated against the live
  H100 vLLM instance, not a script. Commits `d7241e8` / `c55cfe0`.
- **Two environment issues found and fixed on H100, unrelated to this
  branch's code:** vLLM crashed on first start (`Flashinfer allreduce
  workspace` assertion, root cause: missing `ninja` binary — `pip install
  ninja` fixes it), and `--gpu-memory-utilization 0.92` (documented value)
  OOM'd once the local index and GPU embedder loaded alongside it on this
  box's 96GB-per-GPU pair — dropped to `0.85`, fits with ~11.5GB/GPU to
  spare. Both are now documented in `SETUP.md` and `config.yaml.example`.

**Live state right now:** vLLM (port 8000) and `api.py` (port 8080, bound to
`0.0.0.0` so the web UI is reachable at `http://<h100-public-ip>:8080`) are
both still running on the H100 box, left up in case testing continues
tonight. **They cost real money idling** (~$30-40/hr for the VM per
`GI_AND_DFLASH.md`'s own cost note) — worth killing before walking away for
the night unless there's a reason to keep them warm.

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

## Measured on H100 (superseded the projection below)

Full detail in `results/h100_comparison.md`. Same-process, same-question,
warm-call, holding graph content constant so only retrieval speed differs:

| | Cosmos | Local index |
|---|---|---|
| Retrieval | 2.06-2.26s | 0.003-0.03s |
| LLM (DFlash) | 3.65s | 3.46s (same content) / 4.2-4.6s (w/ reverse edges) |
| **End to end** | **5.98s** | **3.87s (same content, ~1.55x) / 4.62s (w/ reverse edges)** |

The 1.8x projection below assumed the LLM stage would be unchanged. It isn't
quite unchanged once reverse edges are on — see the top-of-file summary. The
`entity_search`-gated-by-keyword-expansion effect predicted below is real but
smaller than expected: measured at ~0.15-0.27s unaccounted time on H100
(warm), not the ~0.84s estimated from GB10.

<details>
<summary>Original 2026-07-27 projection (kept for record)</summary>

Against the README's co-located baseline:

| | Baseline | Predicted |
|---|---|---|
| Retrieval | 2.94s | ~0.1s |
| LLM (DFlash) | 3.46s | 3.46s unchanged |
| **End to end** | **6.40s** | **~4.6s (1.8x)** |

It is not larger because `entity_search` on the web path is gated by the
`_llm_expand_keywords` LLM call, not by Cosmos — see "Open questions" below.
</details>

### Memory

The index needs ~4.5 GB (3.75 GB of vectors plus workspace and CUDA context).
`--gpu-memory-utilization 0.92` leaves 8% of 160 GB free, so **12.8 GB**. It
fits with room to spare and needs no vLLM config change. If more headroom is
wanted, dropping to `0.89` frees another 4.8 GB and costs ~10% of a KV cache
the README sizes at 50 GB.

Only the float vectors go on GPU. Triple strings, entity descriptions, food
document text and the CSR arrays live in host RAM, and the VM has 1.9 TB.
Because search is a single GEMV, pin the index to GPU 0 rather than sharding.

**Resolved:** `nvidia-smi` on the actual box confirms `SETUP.md`/`BENCHMARKS.md`
were right — 2x H100 NVL, 96GB each, hostname `ams-agentic-h100`. `README.md`'s
80GB figure was stale and has been corrected. This mattered in practice: `--gpu-memory-utilization 0.92`
claims proportionally more absolute memory on the bigger pair than the 12.8GB
headroom estimated above assumed, and the local index + GPU embedder OOM'd at
0.92 with only 11MiB free. Fixed by dropping to `0.85` (~11.5GB/GPU free) — see
top-of-file summary and `results/h100_comparison.md`.

---

## 2026-07-29: second GB10 pass

Picked up three items flagged "not done" on 2026-07-28, all deliberately
scoped to not need H100 or a live LLM:

### 1. Graph traversal frontier bug (the big one)

`retrieval.py::retrieve()`'s multi-hop loop had two bugs, found by adding
`scripts/sweep_max_hops.py` (retrieval-only, no LLM) and seeing `max_hops`
1 through 5 produce *identical* PK-triple counts — which should be
impossible if hops beyond 1 were doing anything at all.

- **Bug A:** the "compute next hop's frontier" step only ran when
  `hop == 0`. So `max_hops > 2` was a silent no-op regardless of data: by
  hop 2 the frontier (`names`) was stuck at whatever hop 0 computed, every
  name in it was already `visited`, `batch` came back empty, and the loop
  always broke there.
- **Bug B, the one that actually mattered:** even the one hop that *did*
  run (hop 0 -> hop 1) used `t["object"]` as the next frontier. That's only
  correct if the seed was matched as a triple's *subject*. With
  `reverse_edges: true` (the whole point of the local index, since Cosmos
  can't serve reverse edges), seed entities are routinely matched as an
  *object* — inspecting an actual run showed the seed entities were things
  like `"post-workout"` and `"high protein snack"` (claims/tags), and the
  hop-0 triples found were `<product> -[has_claim]-> <seed>`. The object
  *is* the seed itself; the new, useful node is the *subject* (the actual
  product). Every hop-2 attempt was discarding exactly the nodes worth
  exploring.

  Fix: collect both `t["subject"]` and `t["object"]` per triple, drop
  whatever's already visited, use the rest as the next frontier.

**Measured effect** (`scripts/sweep_max_hops.py`, 3 questions x 3 runs,
local index, GB10):

| max_hops | PK triples (avg) | graph_traversal |
|---|---|---|
| 1 (old default) | 11.3 | 13.3ms |
| 2 | **183.7** | 13.7ms |
| 3-5 | 183.7 (plateaus — `max_triples: 40` caps the traversal early) | ~13.6ms |

16x more grounding triples found per query for +0.3ms. `my.yaml` and
`config.yaml.example` now default to `max_hops: 2` (3+ currently plateaus
because of the `max_triples` cap, not because there's nothing further to
find — raising `max_triples` too would be the next experiment if `hops: 3+`
turns out to matter for answer quality).

Validated three ways: `scripts/sweep_max_hops.py` (aggregate), a targeted
inspection script that printed the actual seed entities / hop-0 triples /
would-be hop-1 frontier for one question (confirmed the claims-not-products
pattern directly, not just inferred it from counts), and the live streaming
endpoint (`/v1/ask/stream`) on the "vegan breakfast options" question:
`0 PK + 30 vec` (2026-07-27 baseline) -> `160 PK + 30 vec` (today).
`benchmark_compare.py --skip-cosmos` still shows the expected
`0 PK` on the forward-only row (parity with Cosmos, unchanged) and now
`193 PK` on the forward+reverse row (was ~11-12 before today).

### 2. Background hot-swap for `index.mode: local`

Previously `_get_backend()` always blocked on `LocalGraphIndex.__init__`
(numpy -> GPU, BM25 build over 58k food docs), whether or not the snapshot
already existed. Two different problems collapsed into "hot-swap":

- **Snapshot already on disk** (the common case): the load only takes
  ~2.5-3s, but it happened inside whichever request arrived first, not
  before the server started accepting traffic. Fixed by calling
  `engine._get_backend()` eagerly in `api.py`'s `lifespan()`, concurrently
  with the existing embedder warmup (`asyncio.gather`). Verified in the
  startup log: `[gi_index] loaded in ...` now prints *before*
  `Application startup complete`, so the cost is off the request path
  entirely.
- **Snapshot missing** (fresh deployment, or Cosmos rebuilt and nobody's
  re-exported yet): blocking here means blocking on a 7-10 minute Cosmos
  export. Fixed properly this time: `_get_backend()` returns a
  `CosmosBackend` immediately (construction is lazy, no network call yet)
  and kicks off `scripts/build_local_index.py`'s export (now refactored
  into an importable `build()` function) as a background `asyncio.Task`.
  Once it finishes, the local index load runs via `asyncio.to_thread` (so
  the blocking numpy/GPU work doesn't stall the event loop while
  Cosmos-backed requests are still being served) and swaps
  `self._backend` to `LocalBackend`. New config: `index.auto_build`
  (default `true`; set `false` to fail loudly instead of an unattended
  multi-minute export on first use).

  Deliberately *not* done: falling back to Cosmos when the snapshot is merely
  *stale* rather than *missing*. Reasoning: Cosmos's own cold-connection cost
  was measured earlier in this project at up to ~14s (`results/baseline_gb10.md`),
  worse than the ~3s local load it would be avoiding — so auto-falling-back-
  to-Cosmos would only be a net win for the "doesn't exist yet" case, which
  is the case this implements. Confirmed the mechanism directly (not just by
  reading the code): pointed `index.snapshot_path` at an empty scratch
  directory, called `_get_backend()`, confirmed it returned a `CosmosBackend`
  in 0.05s with a real background export task running (visible in the Cosmos
  SDK's request logs), cancelled the task after 8s once the mechanism was
  confirmed rather than waiting out the full rebuild (which would have just
  re-produced the existing, already-validated snapshot).

### 3. Snapshot freshness guard

New `snapshot_freshness.py`: `check_freshness(cfg, snapshot_path)` compares
`manifest.json`'s `built_at` and per-container `counts` against two cheap
live Cosmos aggregates — `SELECT VALUE COUNT(1) FROM c` and
`SELECT VALUE MAX(c._ts) FROM c`. Either a count mismatch or a newer write
than `built_at` marks the snapshot stale. `scripts/check_snapshot_freshness.py`
is a standalone CLI (exit 1 if stale, for cron/monitoring); `_get_backend()`
also fires it as a non-blocking background task whenever it loads an
existing local backend, logging a warning rather than doing anything
automatic — staleness here is a signal for an operator, not a rebuild
trigger, since the failure modes of "hallucinate outdated food data
silently" and "rebuild unprompted based on a maybe-buggy heuristic" both
seem worse than a log line.

Tested against the real GB10 snapshot and live Cosmos: correctly reports
"Fresh" (entities/triples counts match, no newer writes since
`built_at: 2026-07-27T17:09:15-0700`), and correctly flags `food` as
*unrecorded* — a real, pre-existing gap where the manifest's `counts` dict
only has `entities`/`triples` even though `food.vecs.npy` exists (the
export was apparently run in two passes on GB10). Also tested the stale
path with a deliberately-wrong count injected into a scratch copy of the
manifest: correctly flagged `STALE`, exit code 1.

---

## 2026-07-29: H100 web app + UI duplicate-submission fix

Pulled the branch (including the traversal fix and hot-swap/freshness guard
above) onto the H100 box, rebuilt the local snapshot, and put the web app in
front of a real user for manual testing instead of scripted benchmarks.

**"Site cannot be reached" on the public IP.** `api.py` was started bound to
`127.0.0.1` (localhost-only); the NSG rule (`Allow-Web-8080`) already allowed
inbound traffic from the test IP, but nothing was listening on the external
interface. Fixed by restarting with `--host 0.0.0.0`.

**"Very big delay before answering" on the second question, not the
first.** Reported pattern: select question 1, run — immediate, 4s total.
Select question 2, run — ~20s of nothing, then completes showing a 3.3s
`timings.total`. Clicking stop and re-running immediately during the delay
made it run right away. Investigated in this order:

1. Repeated direct `curl` against both `127.0.0.1` and the public IP,
   including after idle periods — consistently 3.7-4.7s, never ~20s. Ruled
   out cold starts (vLLM prefix cache, snapshot paging, etc).
2. Simulated a client abort mid-request — vLLM correctly cancelled the
   server-side generation, no orphaned request left running. Ruled out
   "previous request never actually finished."
3. Read `api.py`'s access logs: real user requests were arriving in **pairs
   of near-simultaneous connections** (e.g. ports `58936` and `58940` a few
   milliseconds apart) for what the user experienced as one click.
4. Traced to `static/index.html`: the "Ask" button disables itself while a
   request is in flight, but the Enter-key handler on the question textbox
   didn't check anything before calling `doAsk()`. Pressing Enter and then
   also clicking Ask (or double-pressing Enter) fired two concurrent
   `/v1/ask/stream` requests for the same question.
5. Retrieval performs blocking GPU/CPU work directly on the single asyncio
   event loop (`torch.topk` for vector search, BM25 fulltext) — two
   concurrent requests don't run in parallel, they serialize and stall each
   other. That reproduces exactly the reported shape: one request completes
   fast, the other's wall-clock time balloons while its actual
   (`timings.total`) processing time stays small, because most of the ~20s
   was spent waiting for the event loop, not computing.

**Fix** (`static/index.html`, commit `f778834`): added a re-entrancy guard
to `doAsk()` — it now returns immediately if a request is already in flight
(`abortCtrl` set), checked regardless of which trigger (button or Enter)
called it. No backend change needed; this was a pure UI bug. Verified live:
pulled the updated `index.html` onto H100 (no `api.py` restart required —
it's served fresh off disk), user hard-refreshed, confirmed fixed.

**Broader takeaway:** the retrieval path being synchronous-and-fast (sub-
10ms typically) is exactly what made this bug easy to hide — it never shows
up in single-request benchmarking (`benchmark_compare.py`, `sweep_max_hops.py`,
curl loops), only under concurrent load from a real UI. Worth keeping in mind
if/when the retrieval path is asked to serve genuinely concurrent users
(batching section in "Open questions" below) — the current implementation is
fine for one request at a time but will need `asyncio.to_thread` (already
used for the local index's initial load) or a thread/process pool around the
blocking numpy/BM25 calls to actually parallelize multiple in-flight
requests, not just avoid one UI bug.

---

## 2026-07-30: embedding pooling fix

Found while comparing DFlash against the `AgenticRetrieval` baseline
side-by-side (same UI, same question, two apps behind one nginx proxy on
H100 — see "2026-07-29" section below for how that was set up). Baseline
answer for "high-calorie protein snack for long-distance running that fits
in a running belt" included product `22219407`; DFlash's did not, and
DFlash's list included an obviously wrong item (a hair-gel product, which
the LLM itself flagged as likely mislabeled before including it anyway).

**Root cause: the in-process embedder pooled wrong.** `gi_builder.py`'s
`embed_sync`/`embed_batch_sync` mean-pooled `Qwen3-Embedding-0.6B`'s hidden
states. That model is decoder-only and contrastively fine-tuned to be read
off the *last* token's hidden state (with an instruction prefix on the query
side) — mean-pooling throws most of that fine-tuning away, because causal
attention means early tokens never see later context, so averaging them back
in dilutes the one token that saw the whole input. Confirmed directly: raw
cosine similarity ranked `22219407` **#800 of 58,233** food docs (score
0.42) against the query, behind things like "chicken donuts" and "scallop,
girolle and paris mushroom bites" (score 0.67-0.69) — a narrow, uniformly-high
similarity band across unrelated documents, the textbook signature of
degenerate mean-pooled sentence embeddings.

Bumping the retrieval budget first (`seed_entities_k` 10→20, `max_triples`
40→80, `max_source_chunks` 15→25, `vector_augment_k` 12→20) was tried and
ruled out: it didn't surface `22219407`, and in one run made the answer
*worse* (the hair-gel item got included with no self-correction that time).
This is what pointed at a ranking/embedding problem rather than a
budget/recall problem.

**Fix:** last-token pooling (`last_hidden_state[:, -1]` with left-padding) +
Qwen3-Embedding's recommended query instruction prefix, applied only to
question-side embeddings (`EmbedClient.embed(..., is_query=True)`; document
text stays plain). Re-embedded the full local snapshot (58,233 food docs,
179,560 entities, 1,593,678 triples — 1,831,471 vectors total) on the H100's
second GPU, ~19 minutes end to end at ~1,575-1,645 docs/s. Full technical
writeup, exact text templates per container, and the still-open Cosmos
staleness gap: `EMBEDDING_FIX.md`.

**Validated three ways:**
1. Isolated proof-of-concept on 5 known docs: mean pooling put irrelevant
   docs within 0.08-0.15 of the relevant doc's score; last-token pooling
   widened that to 0.18-0.23.
2. The specific reported case: `22219407` went from **rank #800 (0.42) to
   rank #2 (0.55)** in raw food-vector search, and from absent to **item #1
   and the "Top Pick"** in the live app's actual answer. No more hair-gel or
   off-topic items in the top-15.
3. All 10 questions in `data/food.json` re-run against the live, fixed app:
   all succeeded, no errors, answers stayed on-topic and grounded in real
   product IDs (gluten-free popcorn for a cinema snack, pulled pork for a BBQ,
   minced beef for the air-fryer recipe, etc.). Timing unaffected — still
   3.5-6.2s wall time per question, same range as before the fix.

**Not done / follow-up:** Cosmos DB's own `/e` fields were not re-embedded —
only the local snapshot was. Rebuilding `data/local_index` from Cosmos as-is
(`scripts/build_local_index.py`) would silently reintroduce the bug. See
`EMBEDDING_FIX.md`'s "What's still stale" section for the two ways to close
that gap.

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
  auto_build: true        # added 2026-07-29 — see "second GB10 pass"
  check_freshness: true    # added 2026-07-29
```

Also pull `query.max_hops: 2` from `config.yaml.example` (was `1`) — it does
nothing without the `retrieval.py` frontier fix also landing, which it does
as part of this same branch update.

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

`api.py::_dflash_answer` (non-streaming `/v1/ask`) and `_stream_dflash_sse`
(`/v1/ask/stream`, what the actual UI uses) are both converted as of
2026-07-28 evening. `_stream_gi_sse` was dead code (no route referenced it)
and got deleted rather than converted. Start vLLM per `SETUP.md` (remember
`--gpu-memory-utilization 0.85`, not the documented `0.92`, and make sure
`ninja` is on `PATH` — see environment notes) and confirm the local index
loads alongside it without OOM.

```bash
python api.py --config my.yaml --host localhost --port 8080
curl -N -X POST http://localhost:8080/v1/ask/stream -H 'Content-Type: application/json' \
  -d '{"question": "..."}'
```

Watch the `done` event's `timings.ttft` — should be well under 1s. If it
equals `timings.llm`, streaming silently broke and reverted to buffering.

Use `--host 0.0.0.0` instead of `localhost` if the app needs to be reachable
from outside the box (e.g. a real browser hitting the public IP) — the NSG
rule alone isn't enough if nothing is listening on the external interface.
`static/index.html` already has the duplicate-submission fix (commit
`f778834`); no action needed there, it's served fresh off disk.

---

## Not done

**No N-repeat measurement.** Every H100 number in this file and
`results/h100_comparison.md` is a single sample or a 2-3 run average, and the
LLM stage alone showed ~1.1s of spread across nominally identical warm calls.
The retrieval numbers are solid (sub-10ms is sub-10ms, noise doesn't matter
at that scale); the end-to-end and reverse-edges-cost numbers are not tight
enough to defend as anything but "roughly."

**~~Snapshot freshness has no guard.~~ Done 2026-07-29** — see
`snapshot_freshness.py` / "second GB10 pass" above.

**Concurrent requests serialize instead of running in parallel.**
`LocalGraphIndex`'s vector search (`torch.topk`) and BM25 fulltext run
synchronously on the single asyncio event loop, so two truly-concurrent
requests stall each other rather than overlapping — see "H100 web app + UI
duplicate-submission fix" above, where this is exactly what made a client-
side double-submit bug look like a 20s server hang. Not a problem for one
user clicking through the UI (fixed by the re-entrancy guard), but it will
matter the moment more than one user hits the app at once. Fix would be
`asyncio.to_thread` (or a small thread pool) around the blocking numpy/BM25
calls in `gi_index.py`, the same pattern already used for the initial
snapshot load in `gi_query.py::_build_and_swap`. Not yet done.

**Answer quality with reverse edges was never evaluated.** Reverse edges make
answers longer and slower (measured). Whether they're actually *better* — not
just different — was never checked. This is the open question that actually
matters for whether `reverse_edges: true` should stay the default. **More
important after today's frontier fix**: the H100 reverse-edges numbers in
this file were measured against the *buggy* traversal (~11-12 PK triples).
With the fix, the same questions now surface ~16x more PK triples before the
`max_triples: 40` cap trims them, which changes which 40 triples make it into
the prompt. The H100 reverse-edges-cost numbers should be re-measured against
the fixed traversal, not assumed unchanged.

---

## Open questions

**The keyword-expansion LLM call costs less than expected but isn't free.**
Measured on H100 (warm vLLM) at ~0.15-0.27s unaccounted time per
`_dflash_answer`/`_stream_dflash_sse` call — smaller than GB10's 0.84s
estimate because vLLM itself is fast and warm at this small task, but still a
second LLM round trip per question. Local BM25 may make it droppable now that
it's not compensating for weak Cosmos keyword matching. Worth A/B testing.

**~~`max_hops` can now be raised.~~ Done 2026-07-29** — raising it alone did
nothing until the frontier bug above was fixed; with the fix, `max_hops: 2`
is now the default and measured to matter (16x more PK triples). Whether it
should go to 3+ depends on also raising `max_triples` past 40, since that's
what's capping the benefit today — not yet tried.

**DFlash/LLM-stage tuning, not yet done (see chat for full reasoning):**
- Pull real acceptance-rate metrics from vLLM's `/metrics` before touching
  `--spec-tokens` — `GI_AND_DFLASH.md`'s "3-4 of 5 accepted" is a general
  claim, not measured on this model/prompt distribution.
- Tighten the output-length prompt (`DFLASH_ANSWER_PROMPT` in `api.py`
  currently asks for "8-10 products" with full descriptions). Output length
  is the single biggest lever on the LLM stage's wall-clock time and it's a
  one-line prompt edit, not an infra change — try 5 products / 1-2 sentence
  justifications and measure the LLM-stage delta.
- Check whether vLLM 0.23.0 now supports fp8 KV-cache (SETUP.md's
  troubleshooting notes say it was removed as unsupported with fp8
  checkpoints in an earlier version) — would claw back some of the memory
  headroom lost going from `0.92` to `0.85`.
- Batching (`--max-num-seqs`, continuous batching) is a throughput lever for
  concurrent users, not a single-request latency lever — don't expect it to
  move the numbers in this file, which are all single-request timings.

**Whether to keep Cosmos in the loop at all for reads.** Currently it stays as
system of record and the snapshot is rebuilt manually. An incremental update
path via the Cosmos change feed would remove the staleness problem.

**The "8-9 hour build" is two unrelated things, and only one is shortenable
the way it sounds.** Extraction (`gi_builder.py` steps 1-4: LLM triple
extraction, dedup, normalization, entity resolution) is fully in-memory —
`GI_AND_DFLASH.md` documents it as ~8h on 2x H100 at concurrency 20, and it
never touches Cosmos until the end. That part is not Cosmos-bound; it only
gets faster with more concurrency, a faster model, or fewer rounds.

Storage (step 5-6, `store_triples`/`store_entities` at `gi_builder.py:735-772`)
is a different story: a **serial** loop, one `upsert_item` at a time, over
~1.77M documents, into containers created with a **1000 RU/s autoscale
ceiling** (`gi_builder.py:400`). This is very likely eating real, unaccounted
wall-clock time beyond the documented 8h extraction figure. Strong supporting
evidence: five separate ad-hoc scripts already exist
(`fast_upload.py`, `fast_upload_gi.py`, `parallel_upload.py`, `shard_upload.py`,
`turbo_upload.py`) that independently reimplement a `Semaphore`-based
concurrent uploader (concurrency 30-300) with 429 retry/backoff — clear
evidence this was already hit and hand-patched multiple times, but never
folded back into the main pipeline's `store_triples`/`store_entities`.
`migrate_to_provisioned.py` exists for the same upstream reason (moved off a
serverless Cosmos account that couldn't sustain bulk vector writes).

**Fix, not yet done:** add `asyncio.Semaphore` concurrency to
`store_triples`/`store_entities` (pattern already proven in the scripts
above), and check whether `divdet-sweden`'s GI containers are still capped at
autoscale-1000. Needs a real extraction+upload run to validate the time saved
— an 8h+ commitment, so deliberately deferred rather than done blind.

**~~Separately, a background hot-swap for cold start.~~ Done 2026-07-29** —
see "second GB10 pass" above. The same swap-on-ready mechanism
(`_build_and_swap` in `gi_query.py`) is also what a future blue-green
full-rebuild (new extraction run swapped in without downtime) could reuse,
just triggered manually/on a schedule rather than only when the snapshot is
missing.

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

On H100 (`ams-agentic-h100`, 2x H100 NVL 96GB, box already had a `.venv`
symlinked to a sibling repo's venv with everything but `pyarrow`/`pandas`/
`rank_bm25` preinstalled — much less setup friction than GB10):

- **vLLM crash on first start**: `AssertionError: Flashinfer allreduce
  workspace must be initialized when using flashinfer`. Root cause, several
  layers down in the traceback: `ninja` binary missing, so FlashInfer's JIT
  compilation of its allreduce workspace and GDN prefill kernels silently
  fails (logged as a warning, not an error) and something downstream asserts
  on it later. Fix: `pip install ninja`, and make sure it's actually on
  `PATH` for the vLLM process (`pip install` puts the binary in
  `.venv/bin/ninja`, which isn't on `PATH` unless the venv is activated or
  you prepend it explicitly — a plain `.venv/bin/python -m pip install ninja`
  followed by `.venv/bin/vllm serve ...` without `PATH` adjustment will still
  fail the same way).
- **`git remote -v` on this box printed a live GitHub PAT in plaintext** in
  the origin URL. Flagged to revoke it; did not reuse it. If pushing from
  H100 directly is ever wanted, reconfigure the remote with SSH or a
  credential helper first, the same fix used on GB10.
- The repo checkout at `/home/azureuser/AgenticRetrieval-DFlash` already had
  `az` authenticated and a working `.venv` — if this box gets reimaged or a
  fresh clone is needed, budget time for all the GB10 environment notes above
  too, since none of that is guaranteed to carry over.
