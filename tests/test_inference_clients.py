"""Both InferenceClient backends against a mocked HTTP transport (respx) --
valid replies, malformed JSON, and network failures, confirming every
failure path returns GenerationResult(text=None, ...) rather than raising
(the "never raises to the agent" contract -- see inference/base.py), and
that the circuit breaker actually stops issuing requests once open.
"""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from dominion.inference.llamacpp_client import LlamaCppInferenceClient
from dominion.inference.ollama_client import OllamaInferenceClient

OLLAMA_URL = "http://127.0.0.1:11434"
LLAMACPP_URL = "http://127.0.0.1:8080"


@pytest.mark.asyncio
@respx.mock
async def test_ollama_generate_success() -> None:
    respx.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(
            200,
            json={"response": "  2  ", "total_duration": 5_000_000_000, "load_duration": 1_000_000_000},
        )
    )
    client = OllamaInferenceClient(OLLAMA_URL)
    result = await client.generate("llama3.2:latest", "prompt", timeout=5.0)
    await client.aclose()
    assert result.text == "2"
    assert result.think_seconds == pytest.approx(4.0)


@pytest.mark.asyncio
@respx.mock
async def test_ollama_generate_malformed_json_falls_back() -> None:
    respx.post(f"{OLLAMA_URL}/api/generate").mock(
        return_value=httpx.Response(200, content=b"not json")
    )
    client = OllamaInferenceClient(OLLAMA_URL, max_retry_attempts=1)
    result = await client.generate("llama3.2:latest", "prompt", timeout=5.0)
    await client.aclose()
    assert result.text is None
    assert result.think_seconds is None


@pytest.mark.asyncio
@respx.mock
async def test_ollama_generate_network_error_retries_then_falls_back() -> None:
    route = respx.post(f"{OLLAMA_URL}/api/generate").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    client = OllamaInferenceClient(OLLAMA_URL, max_retry_attempts=3,
                                    circuit_breaker_failure_threshold=100)
    result = await client.generate("llama3.2:latest", "prompt", timeout=5.0)
    await client.aclose()
    assert result.text is None
    assert route.call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_ollama_circuit_breaker_stops_calling_once_open() -> None:
    route = respx.post(f"{OLLAMA_URL}/api/generate").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    client = OllamaInferenceClient(OLLAMA_URL, max_retry_attempts=1,
                                    circuit_breaker_failure_threshold=2,
                                    circuit_breaker_cooldown_seconds=60.0)
    await client.generate("m", "p", timeout=5.0)  # failure 1
    await client.generate("m", "p", timeout=5.0)  # failure 2 -> breaker opens
    calls_when_opened = route.call_count
    result = await client.generate("m", "p", timeout=5.0)  # should short-circuit, no network call
    await client.aclose()
    assert result.text is None
    assert route.call_count == calls_when_opened


@pytest.mark.asyncio
@respx.mock
async def test_llamacpp_generate_success_with_grammar() -> None:
    route = respx.post(f"{LLAMACPP_URL}/completion").mock(
        return_value=httpx.Response(
            200, json={"content": "B", "timings": {"predicted_ms": 250.0}}
        )
    )
    client = LlamaCppInferenceClient(LLAMACPP_URL)
    result = await client.generate(
        "qwen2.5-3b-instruct", "prompt", timeout=5.0, grammar='root ::= "A" | "B"')
    await client.aclose()
    assert result.text == "B"
    assert result.think_seconds == pytest.approx(0.25)
    sent_body = route.calls.last.request.content
    assert b"grammar" in sent_body


@pytest.mark.asyncio
@respx.mock
async def test_llamacpp_generate_sends_server_alias_not_registry_key() -> None:
    # Regression test for a real, previously-silent bug: llama-server's
    # router only recognizes the GGUF basename (e.g.
    # "Qwen2.5-3B-Instruct-Q4_K_M") as a model alias, never
    # LLAMACPP_MODELS' own registry key ("qwen2.5-3b-instruct") -- every
    # real llama.cpp show was sending the wrong string and getting a
    # silent HTTP 400, indistinguishable from any other fallback. This
    # mock actually inspects the sent body instead of just returning a
    # canned response regardless of it, which is exactly why the original
    # test above never caught this.
    route = respx.post(f"{LLAMACPP_URL}/completion").mock(
        return_value=httpx.Response(200, json={"content": "ok"})
    )
    client = LlamaCppInferenceClient(LLAMACPP_URL)
    result = await client.generate("qwen2.5-3b-instruct", "prompt", timeout=5.0)
    await client.aclose()
    assert result.text == "ok"
    sent_model = json.loads(route.calls.last.request.content)["model"]
    assert sent_model == "Qwen2.5-3B-Instruct-Q4_K_M"


@pytest.mark.asyncio
@respx.mock
async def test_llamacpp_generate_passes_through_an_already_real_alias() -> None:
    # A caller that already passes a real llama-server alias directly
    # (not a LLAMACPP_MODELS registry key) must keep working unchanged.
    route = respx.post(f"{LLAMACPP_URL}/completion").mock(
        return_value=httpx.Response(200, json={"content": "ok"})
    )
    client = LlamaCppInferenceClient(LLAMACPP_URL)
    await client.generate("Some-Custom-Model-Q4_K_M", "prompt", timeout=5.0)
    await client.aclose()
    sent_model = json.loads(route.calls.last.request.content)["model"]
    assert sent_model == "Some-Custom-Model-Q4_K_M"


@pytest.mark.asyncio
@respx.mock
async def test_llamacpp_generate_http_error_falls_back() -> None:
    respx.post(f"{LLAMACPP_URL}/completion").mock(return_value=httpx.Response(500))
    client = LlamaCppInferenceClient(LLAMACPP_URL, max_retry_attempts=1)
    result = await client.generate("model", "prompt", timeout=5.0)
    await client.aclose()
    assert result.text is None


def test_supports_grammar_flags() -> None:
    # Class-level attributes -- no instance (and so no live httpx.AsyncClient)
    # needs constructing just to check this.
    assert LlamaCppInferenceClient.supports_grammar is True
    assert OllamaInferenceClient.supports_grammar is False
