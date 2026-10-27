"""Offline regression tests: python -m unittest test_request_errors -v."""

import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from azure.core.exceptions import ClientAuthenticationError, ServiceRequestError
from openai import APIConnectionError

import api
from gi_builder import EmbedClient
from request_errors import RequestFailure, describe_failure, request_stage
from retrieval import RetrievalResult


CFG = {
    "embedding": {
        "embed_endpoint": "http://localhost:11434/api/embeddings",
        "embed_model": "qwen3-embedding:0.6b",
    },
    "llm": {"endpoint": "https://llm.example/v1", "model": "test-model"},
    "cosmos": {"uri": "https://cosmos.example", "database_name": "food"},
}


def http_error(status):
    request = httpx.Request("POST", "https://example.test")
    return httpx.HTTPStatusError(
        "sensitive provider response", request=request,
        response=httpx.Response(status, request=request),
    )


class FailureDescriptionTests(unittest.TestCase):
    def test_connection_identifies_ollama_and_model(self):
        error = describe_failure(httpx.ConnectError("All connection attempts failed"), "embedding", CFG)
        self.assertEqual(error.status_code, 503)
        self.assertEqual(error.details["code"], "connection_failed")
        self.assertIn("localhost:11434/api/embeddings", str(error))
        self.assertIn("qwen3-embedding:0.6b", str(error))
        self.assertIn("ollama serve", " ".join(error.details["recommendations"]))

    def test_common_http_errors(self):
        for status, code in ((401, "authentication_failed"), (403, "authentication_failed"),
                             (404, "not_found"), (429, "rate_limited"),
                             (500, "upstream_http_error")):
            with self.subTest(status=status):
                error = describe_failure(http_error(status), "embedding", CFG)
                self.assertEqual(error.details["code"], code)
                self.assertEqual(error.details["upstream_status"], status)
                self.assertTrue(error.details["recommendations"])
                self.assertNotIn("sensitive provider response", str(error))

    def test_missing_ollama_model(self):
        error = describe_failure(http_error(404), "embedding", CFG)
        self.assertIn("ollama pull qwen3-embedding:0.6b", " ".join(error.details["recommendations"]))

    def test_llm_404_is_not_misdiagnosed_as_cosmos(self):
        error = describe_failure(http_error(404), "answer generation", CFG)
        self.assertIn("LLM service", str(error))
        self.assertIn("test-model", str(error))
        self.assertNotIn("Cosmos", str(error))
        self.assertNotIn("ollama", " ".join(error.details["recommendations"]))

    def test_cosmos_auth_and_missing_resources(self):
        error = describe_failure(ClientAuthenticationError("expired"), "retrieval", CFG)
        self.assertIn("az login", " ".join(error.details["recommendations"]))
        error = describe_failure(http_error(404), "retrieval", CFG)
        self.assertIn("database: food", str(error))
        self.assertIn("kg.entities_container", " ".join(error.details["recommendations"]))

    def test_nested_llm_connection_error(self):
        error = APIConnectionError(request=httpx.Request("POST", "https://llm.example/v1"))
        error.__cause__ = httpx.ConnectError("All connection attempts failed")
        failure = describe_failure(error, "answer generation", CFG)
        self.assertEqual(failure.details["code"], "connection_failed")
        self.assertIn("llm.example/v1", str(failure))

    def test_timeout(self):
        error = describe_failure(httpx.ReadTimeout("timed out"), "embedding", CFG)
        self.assertEqual(error.status_code, 504)
        self.assertEqual(error.details["code"], "timeout")

    def test_invalid_embedding_response(self):
        error = describe_failure(ValueError("invalid embedding response"), "embedding", CFG)
        self.assertEqual(error.details["code"], "invalid_response")
        self.assertIn("embedding` array", " ".join(error.details["recommendations"]))

    def test_secrets_and_raw_errors_are_not_exposed(self):
        cfg = {**CFG, "embedding": {
            "endpoint": "https://user:password@example.test/embed?api_key=SECRET#TOKEN",
            "model": "model",
        }}
        payload = json.dumps(describe_failure(
            httpx.ConnectError("Bearer PRIVATE"), "embedding", cfg,
        ).details)
        for secret in ("user:", "password", "SECRET", "TOKEN", "PRIVATE"):
            self.assertNotIn(secret, payload)
        self.assertIn("https://example.test/embed", payload)

    def test_invalid_endpoint_does_not_break_error_handler(self):
        cfg = {**CFG, "embedding": {"endpoint": "http://localhost:invalid"}}
        error = describe_failure(ValueError("bad url"), "embedding", cfg)
        self.assertIn("invalid endpoint", str(error))

    def test_local_index_and_cosmos_fallback_are_distinguished(self):
        cfg = {**CFG, "index": {"mode": "local"}}
        error = describe_failure(FileNotFoundError("missing snapshot"), "retrieval", cfg)
        self.assertIn("local Graph Index", str(error))
        self.assertIn("index.snapshot_path", " ".join(error.details["recommendations"]))
        error = describe_failure(ServiceRequestError("connection refused"), "retrieval", cfg)
        self.assertIn("Cosmos DB", str(error))

    def test_context_retains_original_exception_and_failure(self):
        original = httpx.ConnectError("connection refused")
        with self.assertRaises(RequestFailure) as caught:
            with request_stage("embedding", CFG):
                raise original
        self.assertIs(caught.exception.__cause__, original)
        self.assertIs(describe_failure(caught.exception, "answer preparation", CFG), caught.exception)


class PipelineErrorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = SimpleNamespace(
            _cfg=CFG,
            _embedder=SimpleNamespace(embed=AsyncMock(return_value=[0.1])),
            _get_backend=AsyncMock(),
            _llm=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=AsyncMock(),
            ))),
            _llm_model="test-model", _max_tokens=10, _llm_call_kwargs={},
            _build_graph_context=lambda *args: "graph",
            _build_source_text=lambda *args: "sources",
        )
        self.result = RetrievalResult(
            seed_entities=[{"name": "food"}], timings={"source_fetch": 0.0},
        )
        self.engine._get_backend.return_value.fulltext_food = AsyncMock(return_value=[])

    async def stream_events(self):
        frames = [frame async for frame in api._stream_dflash_sse("question", self.engine)]
        self.assertEqual(frames[-1], "data: [DONE]\n\n")
        return [json.loads(frame[6:]) for frame in frames[:-1]]

    async def test_stream_reports_embedding_failure_with_actions(self):
        self.engine._embedder.embed.side_effect = httpx.ConnectError("All connection attempts failed")
        events = await self.stream_events()
        self.assertEqual(events[0]["message"], "Embedding question...")
        error = events[-1]
        self.assertEqual(error["stage"], "error")
        self.assertEqual(error["operation"], "embedding")
        self.assertIn("ollama serve", " ".join(error["recommendations"]))
        self.assertIn("_ts", error)
        self.engine._get_backend.assert_not_awaited()

    async def test_json_route_has_same_actionable_error(self):
        self.engine._embedder.embed.side_effect = httpx.ConnectError("connection refused")
        previous = getattr(api.app.state, "engine", None)
        api.app.state.engine = self.engine
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api.app), base_url="http://test",
            ) as client:
                response = await client.post("/v1/ask", json={"question": "question"})
        finally:
            api.app.state.engine = previous
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["detail"], data["message"])
        self.assertEqual(data["operation"], "embedding")
        self.assertTrue(data["recommendations"])

    async def test_retrieval_error_identifies_cosmos(self):
        self.engine._get_backend.side_effect = http_error(403)
        with patch.object(api, "_llm_expand_keywords", AsyncMock(return_value=[])):
            events = await self.stream_events()
        self.assertEqual(events[-1]["operation"], "retrieval")
        self.assertIn("az login", " ".join(events[-1]["recommendations"]))

    async def test_llm_stream_error_keeps_stage_even_after_partial_answer(self):
        async def chunks():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))])
            raise httpx.ReadTimeout("timed out")

        self.engine._llm.chat.completions.create.return_value = chunks()
        with patch.object(api, "retrieve", AsyncMock(return_value=self.result)), \
                patch.object(api, "_llm_expand_keywords", AsyncMock(return_value=[])), \
                patch.object(api, "_semantic_rerank", AsyncMock(return_value=[])):
            events = await self.stream_events()
        self.assertTrue(any(event.get("text") == "partial" for event in events))
        self.assertEqual(events[-1]["operation"], "answer generation")
        self.assertEqual(events[-1]["code"], "timeout")

    async def test_successful_stream_and_json_answer_unchanged(self):
        async def chunks():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="answer"))])

        with patch.object(api, "retrieve", AsyncMock(return_value=self.result)), \
                patch.object(api, "_llm_expand_keywords", AsyncMock(return_value=[])), \
                patch.object(api, "_semantic_rerank", AsyncMock(return_value=[])):
            self.engine._llm.chat.completions.create.return_value = chunks()
            events = await self.stream_events()
            self.assertEqual(events[-1]["stage"], "done")
            self.assertTrue(any(event.get("text") == "answer" for event in events))
            self.engine._llm.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
            )
            result = await api._dflash_answer("question", self.engine)
            self.assertEqual(result["answer"], "answer")
            self.assertIn("total", result["timings"])

    async def test_non_streaming_retrieval_failure_cleans_keyword_task(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def keywords(*args):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def backend():
            await started.wait()
            raise http_error(404)

        self.engine._get_backend.side_effect = backend
        with patch.object(api, "_llm_expand_keywords", keywords):
            with self.assertRaises(RequestFailure) as caught:
                await api._dflash_answer("question", self.engine)
        self.assertEqual(caught.exception.details["operation"], "retrieval")
        self.assertTrue(cancelled.is_set())

    async def test_embedding_recovers_after_service_starts_without_restart(self):
        requests = 0

        def handler(request):
            nonlocal requests
            requests += 1
            if requests == 1:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, json={"embedding": [0.1, 0.2]})

        embedder = EmbedClient({**CFG, "embedding": {
            **CFG["embedding"], "embed_dimensions": 2,
        }})
        embedder._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with self.assertRaises(httpx.ConnectError):
                await embedder.embed("question", is_query=True)
            self.assertEqual(await embedder.embed("question", is_query=True), [0.1, 0.2])
        finally:
            await embedder.close()


@unittest.skipUnless(shutil.which("node"), "Node.js is needed for UI regression tests")
class BrowserErrorTests(unittest.TestCase):
    def test_ui_displays_actions_and_detects_broken_streams(self):
        source = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
        helpers = source[source.index("function addProgress("):source.index("function showStats(")]
        stream = source[source.index("async function doStream("):source.index("async function doAsk(")]
        harness = r"""
const assert = require('node:assert/strict');
const messages = [];
const logEl = { appendChild(div) { messages.push(div.innerHTML); }, scrollHeight: 0 };
const document = { createElement() { return {}; } };
const startTime = performance.now();
const currentBackend = 'gi';
const abortCtrl = new AbortController();
const answerEl = {};
const answerModelBadge = { classList: { remove() {} } };
const renderMarkdown = text => text;
const esc = text => String(text).replaceAll('<', '&lt;').replaceAll('>', '&gt;');
function showStats() {}
"""
        scenarios = r"""
(async () => {
  const failure = {
    stage: 'error', message: 'Embedding failed at localhost:11434 <unsafe>',
    recommendations: ['Run ollama serve', 'Check embedding.embed_model'], _ts: 2.5,
  };
  global.fetch = async () => new Response('data: ' + JSON.stringify(failure) + '\n\ndata: [DONE]\n\n');
  await doStream('question');
  assert(messages.some(line => line.includes('localhost:11434')));
  assert(messages.some(line => line.includes('Run ollama serve')));
  assert(messages.some(line => line.includes('[2.5s]')));
  assert(!messages.some(line => line.includes('<unsafe>')));

  messages.length = 0;
  global.fetch = async () => new Response(JSON.stringify(failure), {status: 503});
  await doStream('question');
  assert(messages.some(line => line.includes('Run ollama serve')));

  messages.length = 0;
  global.fetch = async () => new Response('Internal Server Error', {status: 500});
  await doStream('question');
  assert(messages.some(line => line.includes('HTTP 500')));
  assert(messages.some(line => line.includes('run.ps1')));

  messages.length = 0;
  global.fetch = async () => { throw new TypeError('Failed to fetch'); };
  await doStream('question');
  assert(messages.some(line => line.includes('Browser-to-API')));
  assert(messages.some(line => line.includes('run.ps1')));

  messages.length = 0;
  global.fetch = async () => new Response('data: {"stage":"token","text":"partial"}\n\n');
  await doStream('question');
  assert.equal(answerEl.innerHTML, 'partial');
  assert(messages.some(line => line.includes('closed before the answer finished')));

  messages.length = 0;
  global.fetch = async () => new Response('data: {invalid}\n\n');
  await doStream('question');
  assert(messages.some(line => line.includes('invalid progress event')));

  const fallback = await readHttpError(new Response('null', {status: 500}));
  assert(fallback.message.includes('HTTP 500'));
  console.log('7 UI error scenarios passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
        result = subprocess.run(
            ["node", "-"], input=harness + helpers + stream + scenarios,
            text=True, capture_output=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
