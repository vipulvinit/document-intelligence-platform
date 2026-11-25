"""LLM clients behind one small interface.

`OpenAICompatibleClient` talks to any OpenAI-style endpoint, so the same code runs on
open-source models (Llama 3.1 or Qwen on Groq, anything on a local Ollama server) or on OpenAI.
`MockLLM` returns deterministic answers built from the prompt's own context; it lets the
tests, evals and demos run offline with no API key.
"""

import json
import re
from dataclasses import dataclass, field
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


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0):
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        self.model = model
        self.temperature = temperature

    def chat(self, messages, *, tools=None, json_mode=False) -> LLMResponse:
        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=json.loads(c.function.arguments or "{}"))
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
    """Deterministic stand-in for a real model. Reads `[node_key] text` lines from the prompt."""

    def chat(self, messages, *, tools=None, json_mode=False) -> LLMResponse:
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        if tools:
            return self._agent_turn(messages)
        context = [(m["key"], m["text"]) for m in _CONTEXT_LINE.finditer(user)]
        if "TASK: generate_test_cases" in system:
            return LLMResponse(content=json.dumps(self._test_cases(context)))
        return LLMResponse(content=json.dumps(self._answer(user, context)))

    def _answer(self, prompt: str, context):
        question = prompt.split("QUESTION:", 1)[-1].strip()
        q = _words(question)
        best, best_overlap = None, 0
        for key, text in context:
            text = re.sub(r"^\([^)]*\)\s*", "", text)  # drop the "(Section > Sub)" prefix
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                overlap = len(q & _words(sentence))
                if overlap > best_overlap:
                    best, best_overlap = (key, sentence), overlap
        if best is None or best_overlap < 1:
            return {"answer": "I don't know based on the provided documents.", "citations": [], "quotes": []}
        key, sentence = best
        return {"answer": f"{sentence} [{key}]", "citations": [key], "quotes": [sentence]}

    def _test_cases(self, context):
        cases = []
        for i, (key, text) in enumerate(context, start=1):
            if len(text.split()) < 6:  # skip headings
                continue
            first = re.split(r"(?<=[.!?])\s+", text)[0]
            cases.append(
                {
                    "id": f"TC-{i:03d}",
                    "title": f"Verify: {first[:70]}",
                    "objective": f"Confirm the system behaves as the requirement states: {first}",
                    "preconditions": ["Test environment configured", "User has required role"],
                    "steps": [
                        "Set up the scenario described in the requirement",
                        "Perform the action",
                        "Observe the system's response",
                    ],
                    "expected_result": first,
                    "priority": "high"
                    if any(w in text.lower() for w in ("must", "shall", "within"))
                    else "medium",
                    "source_node_keys": [key],
                    "evidence_quote": first,
                }
            )
        return {"test_cases": cases}

    def _agent_turn(self, messages):
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        question = next(m["content"] for m in messages if m["role"] == "user")
        if not tool_msgs:
            name = (
                "compare_versions"
                if re.search(r"\b(change|changed|differ|diff|version)", question, re.I)
                else "search_documents"
            )
            args = {"query": question} if name == "search_documents" else {}
            return LLMResponse(content=None, tool_calls=[ToolCall(id="call_1", name=name, arguments=args)])
        observation = tool_msgs[-1]["content"]
        return LLMResponse(content=f"Based on the tool results: {observation[:600]}")


def get_llm_client() -> LLMClient:
    s = get_settings()
    if s.llm_provider == "mock":
        return MockLLM()
    return OpenAICompatibleClient(s.llm_base_url, s.llm_api_key, s.llm_model, s.llm_temperature)
