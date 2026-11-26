"""A tool-using agent: the LLM decides which tools to call, we run them, and loop.

Loop: send goal + tool list -> model answers or requests tool calls -> run each tool ->
append results -> repeat, at most `agent_max_steps` times. Every step is recorded in a trace
that the API returns.

Tools are read-only over the knowledge base for now. The next step adds a tool that searches the
registered literature sources and ingests what it finds (for diseases the knowledge base does
not cover yet), limited to those sources only.
"""

import json

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import logger
from app.core.safety import wrap_untrusted
from app.llm.client import LLMClient
from app.llm.prompts import AGENT_SYSTEM
from app.models import Article
from app.services import retrieval_service, staleness_service, version_service
from app.services.ingestion_service import get_version

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the stored medical articles; returns relevant passages with article ids.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look for"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_section",
            "description": "Read a section (a heading's node key) of an article's latest version.",
            "parameters": {
                "type": "object",
                "properties": {"article_id": {"type": "integer"}, "node_key": {"type": "string"}},
                "required": ["article_id", "node_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_versions",
            "description": "List what changed between two versions of an article (defaults: the last two).",
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "integer"},
                    "from_version": {"type": "string"},
                    "to_version": {"type": "string"},
                },
                "required": ["article_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_stale_answers",
            "description": "List past answers whose cited sources were later revised or retracted.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _latest_changed_article(db: Session) -> int | None:
    """Default for compare_versions when the model gives no id: the newest article with 2+ versions."""
    for article in db.query(Article).order_by(Article.id.desc()).limit(200):
        if len(article.versions) >= 2:
            return article.id
    return None


def run_tool(db: Session, name: str, args: dict) -> dict:
    if name == "search_knowledge_base":
        hits = retrieval_service.search(db, args.get("query", ""))
        results = []
        for h in hits:
            article = db.get(Article, h.article_id)
            results.append(
                {
                    "article_id": h.article_id,
                    "title": article.title,
                    "node_keys": h.node_keys,
                    "text": h.chunk_text,
                }
            )
        return {"results": results}
    if name == "get_section":
        version = get_version(db, int(args.get("article_id") or 0))
        if not version:
            return {"error": "Unknown article"}
        key = args.get("node_key", "")
        nodes = [n for n in version.nodes if n.node_key == key or n.node_key.startswith(key + "/")]
        if not nodes:
            return {"error": f"No section {key!r}"}
        return {"section": [{"node_key": n.node_key, "text": n.text[:2000]} for n in nodes[:40]]}
    if name == "compare_versions":
        article_id = args.get("article_id") or _latest_changed_article(db)
        if not article_id:
            return {"error": "No article has more than one version yet"}
        diff = version_service.diff_versions(
            db, int(article_id), args.get("from_version"), args.get("to_version")
        )
        if "changes" in diff:
            diff["changes"] = diff["changes"][:20]
        return diff
    if name == "list_stale_answers":
        answers = staleness_service.list_answers(db, status="stale", limit=20)
        return {"stale": [{"id": a.id, "question": a.question, "reason": a.stale_reason} for a in answers]}
    return {"error": f"Unknown tool {name!r}"}


def run_agent(db: Session, llm: LLMClient, goal: str) -> dict:
    messages = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": goal}]
    trace = []
    for step in range(1, get_settings().agent_max_steps + 1):
        resp = llm.chat(messages, tools=TOOLS)
        if not resp.tool_calls:
            trace.append({"step": step, "type": "answer"})
            return {"answer": resp.content, "steps": step, "trace": trace}
        messages.append(
            {
                "role": "assistant",
                "content": resp.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in resp.tool_calls
                ],
            }
        )
        for call in resp.tool_calls:
            logger.info(f"Agent step {step}: {call.name}({call.arguments})")
            try:
                result = run_tool(db, call.name, call.arguments)
            except Exception as exc:  # report tool errors back to the model instead of crashing
                result = {"error": str(exc)}
            trace.append(
                {
                    "step": step,
                    "type": "tool",
                    "tool": call.name,
                    "arguments": call.arguments,
                    "result_preview": json.dumps(result)[:300],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": wrap_untrusted(json.dumps(result), label="tool_result"),
                }
            )
    return {
        "answer": "Stopped: reached the step limit before finishing.",
        "steps": get_settings().agent_max_steps,
        "trace": trace,
    }
