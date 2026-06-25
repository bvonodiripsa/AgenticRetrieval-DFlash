# QA_CLI

An interactive terminal app for **comparing retrieval strategies**. Ask a
question and choose how it gets answered — then run them head to head.

```
✨ Welcome ───────────────────────────────
 🔎 QA_CLI  ·  Tool-use · Agentic Retrieval · Vector Search

 🧭 strategy: ⚖️  Compare
 📄 config  : config.yaml

 Type a question and press Enter. Use quit or Ctrl+C to exit.
──────────────────────────────────────────
❓ Your question: How do tides work?
🔎 Retrieving & reasoning...
┌ 🤖 Tool-use ───┐ ┌ 🧩 Agentic Retrieval ┐ ┌ 🔎 Vector search ┐
│ Tides are ...   │ │ Tides result from ... │ │ The Moon's ...    │
└─────────────────┘ └───────────────────────┘ └───────────────────┘
 Tool-use: ⏱ 4.2s 🔁 3 rounds   Agentic Retrieval: 🧩 2 rounds ❔ 5 sub-questions   Vector search: ⏱ 1.1s 📄 12 docs
```

## Strategies

| Strategy | What it does |
| --- | --- |
| `tool-use` | The agentic function-calling loop (`process_question`): the LLM decides, turn by turn, what to search and when it has enough to answer. |
| `decomposed` | **Agentic Retrieval** — the decomposed multi-round RAG pipeline (`DecomposedRAGPipeline`): break the question into sub-questions, retrieve, and synthesize over several rounds. |
| `vector` | Single-shot baseline: embeds the question, runs one Cosmos DB vector (KNN) search per source via `tool_use_vec_search`, then asks the LLM to answer from those docs. No HyDE, diversity, reranking, or looping. |
| `compare` | Runs **all three** strategies for each question and shows the answers side by side. |

If you don't pass `--strategy`, QA_CLI shows an interactive menu at startup.

## Prerequisites

- A working Agentic Retrieval setup: an Azure Cosmos DB account with at least one
  container already populated by `cosmos_db_upload.py`, plus Azure OpenAI /
  Foundry deployments for the chat and embedding models.
- Python 3.10+.

## Install

From this folder, in your project virtual environment:

```bash
# Project dependencies (from the repo root)
pip install -r ../../requirements.txt

# QA_CLI's extra dependency (the rich terminal UI)
pip install -r requirements.txt
```

## Configure

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` and set at least:

- `llm.llm_endpoint`, `llm.llm_model` (+ `llm.llm_api_key` if `use_rbac_auth: false`)
- `embedding.embed_endpoint`, `embedding.embed_model`
- `cosmos.uri`, `cosmos.database_name`, and one entry under `cosmos.sources`
  matching the container you ingested (`container_name`, `embedding_field`,
  `retrieval.search_k`, `retrieval.fulltext_search_k`, `retrieval.fulltext_fields`)
- `ranker.use_ranker` — set to `false` if you do not have a semantic ranker
  resource, or fill in `ranker.region` / `ranker.account_name`.

Authentication follows the same rules as the root project: Cosmos DB and the
Azure OpenAI endpoints can use Entra ID RBAC (`use_rbac_auth: true`, requires
`az login`) or key-based auth.

## Run

```bash
python qa_cli.py                          # interactive strategy menu
python qa_cli.py --strategy tool-use      # agentic function-calling loop
python qa_cli.py --strategy decomposed    # Agentic Retrieval (multi-round)
python qa_cli.py --strategy vector        # plain vector search
python qa_cli.py --strategy compare       # run all three, side by side
python qa_cli.py --config my.yaml
python qa_cli.py --strategy vector --k 15 # override vector top-k
python qa_cli.py -v                        # verbose steps + full tracebacks
```

At the prompt, type a question and press Enter. Enter `quit` (or an empty line /
Ctrl+C) to exit.

### Saving research results (`-v` / `--verbose`)

With `--verbose`, QA_CLI also writes the **raw research result** of each answered
question as JSON into a `results/` subfolder next to the script. One file is
written per method per question (`<timestamp>_<method>.json`), so a `compare` run
produces three files (`tool-use`, `decomposed`, `vector`). Each file records the
question, a timestamp, and the method's full result payload (the decomposed
rounds / sub-questions, the tool-use trace, or the retrieved documents for the
vector baseline — embedding vectors are stripped to keep files readable). The
`results/` folder is git-ignored.

## How it works

1. Resolves the repo root (two levels up) and adds it to `sys.path`.
2. Calls `dynamic_retriever.load_config()` **before** importing
   `utils.cosmos_retriever` (that module reads the config at import time).
3. Picks the strategy from `--strategy` (or the startup menu):
   - **tool-use** — `init_tool_use_clients()` + `process_question()`.
   - **decomposed** (Agentic Retrieval) — `CombinedRetriever` +
     `DecomposedRAGPipeline.run_efficient()`.
   - **vector** — `init_tool_use_clients()` for the shared embedding/LLM clients,
     then `tool_use_embed()` → `tool_use_vec_search()` per source → one LLM call
     answering strictly from the retrieved context.
   - **compare** — runs all three for each question.
4. Renders the answer(s) with [rich](https://github.com/Textualize/rich).
