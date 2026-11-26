"""The OpenAI-compatible client against a fake provider (no key, no network)."""

import json

import httpx
import pytest

from app.llm.client import LLMUnavailable, OpenAICompatibleClient


def _completion(content=None, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


def _client(handler, **kw):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleClient("https://fake.test/v1", "key", "m", max_retries=0, http_client=http, **kw)


def test_json_mode_rejected_falls_back_and_is_remembered():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append("response_format" in body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        return httpx.Response(200, json=_completion('{"answer": "ok"}'))

    llm = _client(handler)
    assert llm.chat([{"role": "user", "content": "hi"}], json_mode=True).content == '{"answer": "ok"}'
    llm.chat([{"role": "user", "content": "again"}], json_mode=True)
    assert seen == [True, False, False]  # second call does not try JSON mode again


def test_bad_key_and_rate_limit_give_clear_errors():
    with pytest.raises(LLMUnavailable, match="API key was rejected"):
        _client(lambda r: httpx.Response(401, json={"error": {"message": "invalid key"}})).chat([])
    with pytest.raises(LLMUnavailable, match="Rate limit"):
        _client(lambda r: httpx.Response(429, json={"error": {"message": "slow down"}})).chat([])
    with pytest.raises(LLMUnavailable, match="not found"):
        _client(lambda r: httpx.Response(404, json={"error": {"message": "no such model"}})).chat([])


def test_malformed_tool_arguments_become_empty():
    call = {
        "id": "t1",
        "type": "function",
        "function": {"name": "search_knowledge_base", "arguments": "{bad"},
    }
    resp = _client(lambda r: httpx.Response(200, json=_completion(None, [call]))).chat([], tools=[{}])
    assert resp.tool_calls[0].name == "search_knowledge_base" and resp.tool_calls[0].arguments == {}
