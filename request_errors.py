"""Safe, actionable diagnostics for failures in the question pipeline."""

from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit

import httpx
from azure.core.exceptions import (
    ClientAuthenticationError, HttpResponseError, ServiceRequestError, ServiceResponseError,
)
from openai import APIConnectionError, APITimeoutError


def _safe_endpoint(value: str) -> str:
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return "(invalid endpoint; check configuration)"
    if not parts.hostname:
        return "(invalid endpoint; check configuration)"
    host = parts.hostname
    if ":" in host:
        host = f"[{host}]"
    if port:
        host += f":{port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


class RequestFailure(RuntimeError):
    def __init__(self, details: dict, status_code: int):
        super().__init__(details["message"])
        self.details = details
        self.status_code = status_code


def describe_failure(exc: Exception, operation: str, cfg: dict) -> RequestFailure:
    if isinstance(exc, RequestFailure):
        return exc

    causes = []
    current: BaseException | None = exc
    while current is not None and current not in causes:
        causes.append(current)
        current = current.__cause__ or current.__context__

    endpoint = ""
    model = ""
    if operation == "embedding":
        settings = cfg.get("embedding", {})
        endpoint = str(settings.get("embed_endpoint") or settings.get("endpoint") or "")
        model = str(settings.get("embed_model") or settings.get("model") or "Qwen/Qwen3-Embedding-0.6B")
        service = "embedding service" if endpoint else "in-process embedding model"
        config_key = "embedding.embed_endpoint / embedding.embed_model"
    elif operation == "answer generation":
        settings = cfg.get("llm", {})
        endpoint = str(settings.get("llm_endpoint") or settings.get("endpoint") or "http://localhost:8000/v1")
        model = str(settings.get("llm_model") or settings.get("model") or "Qwen/Qwen2.5-32B-Instruct")
        service = "LLM service"
        config_key = "llm.llm_endpoint / llm.endpoint and the configured model"
    elif operation in ("retrieval", "keyword search"):
        settings = cfg.get("cosmos", {})
        endpoint = str(settings.get("uri") or "")
        kg = cfg.get("kg", {})
        service = (
            f"Cosmos DB (database: {settings.get('database_name', 'not configured')}; "
            f"containers: {kg.get('entities_container', 'entities')}, "
            f"{kg.get('triples_container', 'triples')}, food)"
        )
        config_key = "cosmos.uri, cosmos.database_name, kg.entities_container and kg.triples_container"
        if cfg.get("index", {}).get("mode") == "local" and not any(
            isinstance(c, (HttpResponseError, ServiceRequestError, ServiceResponseError,
                           ClientAuthenticationError)) for c in causes
        ):
            service = "local Graph Index"
            endpoint = ""
            config_key = "index.snapshot_path, index.device and index.auto_build"
    else:
        service = "question pipeline"
        config_key = "the active YAML configuration"

    status = next((getattr(c, "status_code", None) for c in causes
                   if getattr(c, "status_code", None) is not None), None)
    status = status or next((c.response.status_code for c in causes
                             if isinstance(c, httpx.HTTPStatusError)), None)
    ollama = operation == "embedding" and (
        ":11434" in endpoint or "/api/embeddings" in endpoint
    )
    recommendations = []
    http_status = 502

    if any(isinstance(c, (httpx.TimeoutException, APITimeoutError, TimeoutError)) for c in causes):
        code, reason, http_status = "timeout", "the service did not respond before the timeout", 504
        recommendations.append("Check that the service is responsive; wait for model loading to finish and retry.")
    elif any(isinstance(c, (httpx.ConnectError, APIConnectionError, ConnectionError,
                            ServiceRequestError, ServiceResponseError)) for c in causes):
        code, reason, http_status = "connection_failed", "could not connect to the service", 503
        recommendations.append("Check the configured host/port, network or VPN, proxy and firewall settings.")
        if ollama:
            recommendations.insert(0, "Start the Ollama desktop app or run `ollama serve`, then retry the question.")
        elif operation == "answer generation":
            recommendations.insert(0, "If using a local LLM, start its server; for a hosted LLM, check provider availability.")
    elif status in (401, 403) or any(isinstance(c, ClientAuthenticationError) for c in causes):
        code, reason = "authentication_failed", "authentication failed or access was denied"
        if operation in ("retrieval", "keyword search"):
            recommendations.append("Run `az login --tenant <cosmos.tenant_id>` and verify Cosmos DB data-plane RBAC permissions.")
        else:
            recommendations.append("Check the configured service API key/token and permissions; do not share the key in logs.")
    elif status == 404:
        code, reason = "not_found", "the configured model, resource or API route was not found"
        if ollama:
            recommendations.append(f"Run `ollama list`; if the configured model is missing, run `ollama pull {model}`.")
        elif operation in ("retrieval", "keyword search"):
            recommendations.append("Verify the Cosmos database and food/entities/triples container names in the active config; build the Graph Index if missing.")
        else:
            recommendations.append("Verify the endpoint API path and that the configured model is available from this provider.")
    elif status == 429:
        code, reason, http_status = "rate_limited", "the service rate limit or quota was exceeded", 503
        recommendations.append("Wait and retry; check provider quota/billing and reduce concurrent requests.")
    elif status is not None:
        code, reason = "upstream_http_error", f"the service returned HTTP {status}"
        recommendations.append("Check the model name, request settings and provider/server logs; retry later for HTTP 5xx errors.")
    elif operation == "embedding" and isinstance(exc, ValueError):
        code, reason = "invalid_response", "the embedding response was invalid"
        recommendations.append("Use an Ollama-compatible embedding endpoint returning an `embedding` array and check the configured model.")
    else:
        code, reason, http_status = "pipeline_error", f"an unexpected {type(exc).__name__} occurred", 500
        recommendations.append("Check the API terminal traceback for the underlying error before retrying.")
        if operation == "embedding" and not endpoint:
            recommendations.append("Check local model dependencies/cache and available GPU/RAM, or configure the embedding HTTP endpoint.")
        elif operation in ("retrieval", "keyword search") and cfg.get("index", {}).get("mode") == "local":
            recommendations.append("Check index.snapshot_path and GPU/RAM; build the snapshot with scripts/build_local_index.py or select index.mode: cosmos.")

    recommendations.append(f"Verify {config_key} in the active config.")
    safe_endpoint = _safe_endpoint(endpoint) if endpoint else ""
    target = f" at {safe_endpoint}" if safe_endpoint else ""
    model_text = f" (model: {model})" if model else ""
    message = f"{operation.capitalize()} failed: {reason}. Service: {service}{target}{model_text}."
    return RequestFailure({
        "message": message,
        "operation": operation,
        "service": service,
        "endpoint": safe_endpoint,
        "model": model,
        "code": code,
        "upstream_status": status,
        "exception_type": type(exc).__name__,
        "recommendations": recommendations,
    }, http_status)


@contextmanager
def request_stage(operation: str, cfg: dict):
    try:
        yield
    except RequestFailure:
        raise
    except Exception as exc:
        raise describe_failure(exc, operation, cfg) from exc
