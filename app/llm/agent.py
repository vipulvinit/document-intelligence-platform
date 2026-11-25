"""A tool-using agent: the LLM decides which tools to call, we run them, and loop.

Loop: send goal + tool list -> model answers or requests tool calls -> run each tool ->
append results -> repeat, at most `agent_max_steps` times.
All tools are read-only (least privilege): the agent can look things up but cannot
change data. Every step is recorded in a trace that the API returns.
"""

import json

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import logger
from app.core.safety import wrap_untrusted
from app.llm.client import LLMClient
from app.llm.prompts import AGENT_SYSTEM
from app.services import retrieval_service, staleness_service, version_service
from app.services.document_service import get_document, get_version

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the latest document versions; returns relevant passages with node keys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for"},
                    "document_name": {"type": "string", "description": "Optional: limit to one document"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_section",
            "description": "Read the full text of a section (a heading's node key) in the latest version.",
            "parameters": {
                "type": "object",
                "properties": {"document_name": {"type": "string"}, "node_key": {"type": "string"}},
                "required": ["document_name", "node_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_versions",
            "description": "List what changed between two versions of a document (defaults: the last two).",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_name": {"type": "string"},
                    "from_version": {"type": "string"},
                    "to_version": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_stale_test_cases",
            "description": "List generated test cases made stale by changes to their source text.",
            "parameters": {
                "type": "object",
                "properties": {"document_name": {"type": "string"}},
                "required": [],
            },
        },
    },
]


def _only_document_name(db: Session) -> str | None:
    from sqlalchemy import select

    from app.models import Document

    names = db.scalars(select(Document.name)).all()
    return names[0] if len(names) == 1 else None


def run_tool(db: Session, name: str, args: dict) -> dict:
    doc_name = args.get("document_name") or _only_document_name(db)
    if name == "search_documents":
        hits = retrieval_service.search(db, args.get("query", ""), args.get("document_name"))
        return {
            "results": [
                {"document": h.document, "version": h.version, "node_keys": h.node_keys, "text": h.chunk_text}
                for h in hits
            ]
        }
    if name == "get_section":
        version = get_version(db, doc_name) if doc_name else None
        if not version:
            return {"error": "Unknown document"}
        key = args.get("node_key", "")
        nodes = [n for n in version.nodes if n.node_key == key or n.node_key.startswith(key + "/")]
        return (
            {"section": [{"node_key": n.node_key, "text": n.text} for n in nodes]}
            if nodes
            else {"error": f"No section {key!r}"}
        )
    if name == "compare_versions":
        if not doc_name:
            return {"error": "Specify document_name"}
        diff = version_service.diff_versions(db, doc_name, args.get("from_version"), args.get("to_version"))
        if "changes" in diff:
            diff["changes"] = diff["changes"][:20]
        return diff
    if name == "list_stale_test_cases":
        doc = get_document(db, doc_name) if doc_name else None
        if not doc:
            return {"error": "Specify document_name"}
        arts = staleness_service.list_artifacts(db, doc.id, status="stale")
        return {
            "stale": [{"id": a.id, "title": a.payload.get("title"), "reason": a.stale_reason} for a in arts]
        }
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
