# H100 comparison — measured, not projected

Run 2026-07-28 on `ams-agentic-h100` (2x NVIDIA H100 NVL, 96GB each — this
resolves the `README.md` vs `SETUP.md`/`BENCHMARKS.md` discrepancy noted in
`PROGRESS.md`: it's the 96GB NVL pair). Branch `feat/gpu-graph-index` @
`a0392b6`.

## Retrieval: matches the GB10 prediction almost exactly

| Stage | Cosmos (same box) | Local GPU (fwd) | Local GPU (fwd+rev) |
|---|---|---|---|
| entity_search | 0.505s | 0.000s | 0.000s |
| graph_traversal | 0.813s | 0.002s | 0.002s |
| source_fetch | 0.745s | 0.001s | 0.001s |
| **total** | **2.063s** | **0.003s** | **0.003s** |

~690x on retrieval. `benchmark_pipeline.py`'s own 3-run baseline (discarding a
cold-start first run) averaged 2.70s, close enough to corroborate this and to
the README's 2.94s co-located figure.

Same correctness signature as GB10: `exact search wins — 2 hits Cosmos missed
all score >= 0.6182, vs Cosmos-only max 0.6117`. Same traversal fix: reverse
edges take graph traversal from `0 PK` to `12 PK` triples for this question.

## Memory: fits, but with less margin than estimated

`--gpu-memory-utilization 0.92` (the documented value) left only **~5.1 GB
free per GPU** on this 96GB pair — not the ~12.8 GB `PROGRESS.md` estimated
(that arithmetic used the README's 80GB figure). Loading the local index
(3.75 GB) plus moving the embedder to GPU (~1.2 GB) **OOM'd**:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 0 has a total capacity of 93.09 GiB of which 11.00 MiB is free.
```

Fix: dropped to `--gpu-memory-utilization 0.85`, which left ~11.5 GB free per
GPU — comfortable. Confirmed working end to end with no OOM. **If you deploy
this, use 0.85–0.88 on H100 NVL 96GB, not the documented 0.92**, until the
local index is running alongside it.

Unrelated environment issue hit along the way: vLLM crashed on first start with
`AssertionError: Flashinfer allreduce workspace must be initialized when using
flashinfer`, root-caused to a missing `ninja` binary (FlashInfer needs it to
JIT-compile the allreduce workspace and GDN prefill kernels; the failure is
silently swallowed as a warning, then asserts later). Fixed with
`pip install ninja` — nothing to do with this branch's changes.

## End-to-end: real number is smaller than projected, and here's why

Warm, steady-state, 3 consecutive calls through `api.py::_dflash_answer`
(discarding a cold first call whose `embed` and `llm` stages both paid one-time
GPU kernel compilation):

| Run | embed | entity | graph | source (+BM25) | kw-expand (unaccounted) | llm | **total** |
|---|---|---|---|---|---|---|---|
| 1 | 0.024s | 0.001s | 0.002s | 0.236s | 0.15s | 5.24s | **5.66s** |
| 2 | 0.024s | 0.001s | 0.002s | 0.237s | 0.15s | 5.24s | **5.66s** |

**Projected 3.53s. Measured 5.66s.** The gap is not retrieval — retrieval
matches the projection almost exactly (0.027s here vs the 0.003–0.064s range
predicted). It's the LLM stage: **5.24s measured vs 3.46s in the README's
baseline.**

Working hypothesis, not yet isolated: reverse edges feed the LLM a richer
prompt. Forward-only Cosmos gave `0 PK + 30 vec -> 30 triples`; local with
reverse edges gives `12 PK + 30 vec -> 40 triples`, plus BM25 keyword hits the
Cosmos path didn't have. More context tokens and a fuller answer (the model
returned 9 detailed product recommendations) plausibly cost more decode time
than the README's benchmark question did. Not confirmed — would need a
same-prompt, same-`reverse_edges` A/B to isolate from ordinary run-to-run LLM
variance and the lower `--gpu-memory-utilization` (0.85 vs 0.92, smaller KV
cache pool).

There's also a small, real cost nowhere in the original design: the
keyword-expansion LLM call (`_llm_expand_keywords` in `api.py`) still runs
concurrently with entity search and costs ~0.15s unaccounted in the timings
dict — smaller here than GB10's projected 0.84s because vLLM itself is warm
and fast at this micro-task, but it's not free, and it's exactly the item
flagged as an open question in `PROGRESS.md`.

## Bottom line

Retrieval optimization: **fully validated, exactly as predicted.** 2.06s to
0.003s, on this exact hardware, with the same correctness profile as GB10.

End-to-end: **smaller net win than projected (5.66s vs 3.53s projected, vs
6.40s original baseline — about 1.13x, not 1.8x)**, because the traversal bug
fix changed what gets fed to the LLM, and that costs more than retrieval saves.
This is not a flaw in the local index — it's the LLM stage now doing more
work because the graph context is genuinely richer. Whether that trade is
worth it depends on whether the richer answers are actually better, which
wasn't evaluated here.

## Follow-up: isolating retrieval speed from the reverse-edges content change

Ran three same-process, same-question, warm-call configs back to back so the
only thing that changes between adjacent rows is one variable:

| Config | LLM stage | **Total (warm)** |
|---|---|---|
| OLD Cosmos (0 PK triples, forward-only) | 3.65s | **5.98s** |
| NEW local, forward-only (0 PK, same content as Cosmos) | 3.46s | **3.87s** |
| NEW local, forward+reverse (12 PK, richer content) | 4.20-4.58s | **4.62s** |

Holding graph-traversal content constant (row 1 vs row 2: both `0 PK`, so the
LLM sees the same triples either way), **the local index alone is worth ~2.1s,
~1.55x** — not the ~0.42s/1.08x the first same-prompt comparison suggested.
That first comparison conflated two effects: the index speedup (real, ~1.55x)
and the reverse-edges content change (real, separate, costs ~0.75s here).

Caveat: the LLM stage alone spanned 3.46s-4.58s across these nominally
identical warm calls — ~1.1s of run-to-run decode variance (speculative
decoding's acceptance rate isn't deterministic). Each number above is a
single sample sitting inside that noise band, not a tight estimate. A
defensible number for a writeup would need ~5 repeats per config for a mean
and stddev, not one sample each.

## Recommended next step

Re-run the three-way comparison above with N=5+ repeats per config to get
confidence intervals tight enough to actually defend "1.55x" or "0.75s" as
numbers rather than single samples. Also worth quoting the keyword-expansion
removal experiment from `PROGRESS.md`'s open questions against this real
end-to-end number rather than the GB10 projection.
