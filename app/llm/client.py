"""LLM clients behind one small interface.

`OpenAICompatibleClient` talks to any OpenAI-style endpoint: free online tiers (Groq, Google
Gemini, OpenRouter) or a local Ollama server. Switching provider only means changing
LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env.
`MockLLM` returns deterministic answers built from the prompt's own context; it lets the
tests, evals and demos run offline with no API key.
"""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from app.core.config import get_settings
from app.core.logging import logger


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    def chat(
        self, messages: list[dict], *, tools: list[dict] | None = None, json_mode: bool = False
    ) -> LLMResponse: ...


class LLMUnavailable(RuntimeError):
    """The model API could not be used: unreachable, bad key, unknown model, or rate limit exhausted."""


def _tool_arguments(raw: str | None) -> dict:
    """Some providers occasionally return malformed tool arguments: treat them as empty."""
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        logger.warning(f"Ignoring malformed tool arguments: {raw!r:.200}")
        return {}
    return args if isinstance(args, dict) else {}


class OpenAICompatibleClient:
    """Works with any OpenAI-compatible chat API: Groq, Google Gemini, OpenRouter, Ollama, OpenAI..."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 5,
        json_mode: bool = True,
        http_client=None,  # tests pass an httpx.Client with a fake transport
    ):
        from openai import OpenAI

        # The SDK retries 429 / 5xx / connection errors with exponential backoff and honours Retry-After.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
        )
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.json_mode = json_mode

    def chat(self, messages, *, tools=None, json_mode=False) -> LLMResponse:
        import openai

        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
        if json_mode and self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except openai.BadRequestError as exc:
                if "response_format" not in kwargs:
                    raise
                # This provider/model does not support JSON mode: the prompt already asks for JSON
                # and the validator checks it, so retry without it and remember for next time.
                logger.warning(f"{self.model} rejected JSON mode ({exc.message}); continuing without it")
                self.json_mode = False
                kwargs.pop("response_format")
                resp = self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise LLMUnavailable(f"The API key was rejected by {self.base_url}: check LLM_API_KEY.") from exc
        except openai.RateLimitError as exc:
            raise LLMUnavailable(
                f"Rate limit or free quota exhausted at {self.base_url} even after retries. Wait a minute, "
                "use a smaller model, or switch provider in .env."
            ) from exc
        except openai.NotFoundError as exc:
            raise LLMUnavailable(
                f"Model {self.model!r} not found at {self.base_url}: check LLM_MODEL against the "
                "provider's model list (with Ollama: `ollama pull <model>`)."
            ) from exc
        except openai.APIConnectionError as exc:
            raise LLMUnavailable(
                f"Cannot reach the model API at {self.base_url}: check LLM_BASE_URL and your internet "
                "connection (or that Ollama is running, if you use it)."
            ) from exc
        if not resp.choices:
            raise LLMUnavailable(
                f"{self.base_url} returned no answer (possibly blocked by its safety filter)."
            )
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=_tool_arguments(c.function.arguments))
            for c in (msg.tool_calls or [])
        ]
        usage = getattr(resp, "usage", None)
        if usage:
            logger.info(
                f"LLM call: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens"
            )
        return LLMResponse(content=msg.content, tool_calls=calls)


# ── Offline mock ─────────────────────────────────────────────────────────────
_CONTEXT_LINE = re.compile(r"^\[(?P<key>[^\]]+)\]\s(?P<text>.+)$", re.MULTILINE)
_WORD = re.compile(r"[a-z0-9]+")
NO_ANSWER_MOCK = "I don't know based on the available medical literature."


def _words(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "of",
        "to",
        "is",
        "are",
        "and",
        "in",
        "for",
        "what",
        "how",
        "be",
        "must",
        "on",
        "by",
        "who",
        "which",
        "does",
    }
    return {w for w in _WORD.findall(text.lower()) if w not in stop and len(w) > 2}


class MockLLM:
    """Deterministic stand-in for a real model. Reads `[S1] text` lines from the prompt."""

    def chat(self, messages, *, tools=None, json_mode=False) -> LLMResponse:
        user = messages[-1]["content"] if messages else ""
        if tools:
            return self._agent_turn(messages)
        context = [(m["key"], m["text"]) for m in _CONTEXT_LINE.finditer(user)]
        return LLMResponse(content=json.dumps(self._answer(user, context)))

    def _answer(self, prompt: str, context):
        question = prompt.split("QUESTION:", 1)[-1].strip()
        q = _words(question)
        best, best_overlap = None, 0
        for key, text in context:
            text = re.sub(r"^\[[^\]]*\]\s*", "", text)  # drop the "[Article title]" prefix
            text = re.sub(r"^\([^)]*\)\s*", "", text)  # drop the "(Section > Sub)" prefix
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                overlap = len(q & _words(sentence))
                if overlap > best_overlap:
                    best, best_overlap = (key, sentence), overlap
        if best is None or best_overlap < 1:
            return {"answer": NO_ANSWER_MOCK, "citations": [], "quotes": []}
        key, sentence = best
        return {"answer": f"{sentence} [{key}]", "citations": [key], "quotes": [sentence]}

    def _agent_turn(self, messages):
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        question = next(m["content"] for m in messages if m["role"] == "user")
        if not tool_msgs:
            name = (
                "compare_versions"
                if re.search(r"\b(change|changed|differ|diff|version)", question, re.I)
                else "search_knowledge_base"
            )
            args = {"query": question} if name == "search_knowledge_base" else {}
            return LLMResponse(content=None, tool_calls=[ToolCall(id="call_1", name=name, arguments=args)])
        observation = tool_msgs[-1]["content"]
        return LLMResponse(content=f"Based on the tool results: {observation[:600]}")


@lru_cache(maxsize=4)
def _cached_client(*config) -> OpenAICompatibleClient:
    # One client per configuration: keeps the HTTP connection pool and what was learnt about the
    # provider (e.g. that it rejects JSON mode) across requests.
    return OpenAICompatibleClient(*config)


def get_llm_client() -> LLMClient:
    s = get_settings()
    if s.llm_provider == "mock":
        return MockLLM()
    return _cached_client(
        s.llm_base_url,
        s.llm_api_key,
        s.llm_model,
        s.llm_temperature,
        s.llm_timeout_seconds,
        s.llm_max_retries,
        s.llm_json_mode,
    )
