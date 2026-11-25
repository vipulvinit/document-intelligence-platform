# Document Intelligence Platform

A FastAPI backend that turns unstructured documents (PDF, Markdown, text) into validated section
trees, tracks how they change across versions using semantic versioning, answers questions with
citations (RAG), and generates QA test cases with an LLM, each traceable to the exact paragraph it
came from. When a document changes, test cases built from the changed text are flagged as stale.

Built for regulated documents such as SOPs and specifications, where you need to know *what changed*
and *what it affects*.

## Architecture

```mermaid
flowchart LR
    U[Upload PDF / MD] --> P[Parser<br/>PyMuPDF fonts, OCR fallback]
    P --> H[Hierarchy builder<br/>section tree + stable node keys]
    H --> V{Compare with<br/>previous version}
    V --> S[Semantic version<br/>MAJOR.MINOR.PATCH]
    V --> T[Staleness check<br/>on generated test cases]
    H --> C[Chunk + embed]
    C --> DB[(PostgreSQL)]
    S --> DB
    Q[Question] --> R[Hybrid retrieval<br/>BM25 + embeddings, RRF]
    DB --> R
    R --> G[Guardrails<br/>PII mask, injection quarantine]
    G --> L[LLM<br/>Llama 3.1 / Qwen via Groq or Ollama]
    L --> X[Validator<br/>schema + citation grounding, 1 retry]
    X --> A[Answer with citations]
    DB --> TG[Test-case generator] --> L
    AG[Agent: read-only tools] --> R
```

## What each part does

| Stage | How it works | Code |
| --- | --- | --- |
| Parsing | PyMuPDF reads text with font size and weight; Tesseract OCR for scanned pages (optional); Markdown parsed directly | `app/parsers/` |
| Structure | Headings inferred from font size and boldness, nested into a tree. Each node gets a stable key from its heading path, e.g. `3-data-validation/3-2-query-management/p1`. Pydantic validates the tree (unique keys, parents exist) | `app/parsers/hierarchy_builder.py`, `app/schemas/node.py` |
| Versioning | Nodes are hashed (SHA-256 of normalised text) and matched to the previous version in four passes: same key and hash; same hash, moved; same key, similar text; best similar text. | `app/versioning/` |
| Semantic versioning | Section added or removed = MAJOR. Paragraph added, removed or changed = MINOR. Cosmetic edit only = PATCH. No change = upload rejected. A changed number or negation always counts as a real change, even at 98% text similarity | `app/versioning/differ.py`, `matcher.py` |
| Traceability | Each generated test case stores the hash of every source node. A new version that changes or removes one marks the test case stale, with the reason | `app/services/staleness_service.py` |
| Retrieval | Sections split into ~800-character chunks that never cross a heading, prefixed with their heading path. Hybrid search: BM25 (exact terms) + cosine similarity on `bge-small-en-v1.5` embeddings, merged with reciprocal rank fusion | `app/services/chunking.py`, `retrieval_service.py` |
| Generation | Structured JSON output with a few-shot example. Validated with Pydantic, then grounded: citations must be excerpts the model was given, and quotes must appear in them. On failure, one retry with the errors fed back | `app/llm/` |
| Agent | Tool-calling loop (max 5 steps) with read-only tools: search, read a section, compare versions, list stale test cases. Returns a full trace | `app/llm/agent.py` |
| Safety | PII (email, phone, PAN, Aadhaar) masked before the LLM and in all logs. Document text is fenced as untrusted data; passages matching injection patterns are quarantined and reported, never sent to the model | `app/core/safety.py` |
| Evaluation | 11 cases: answer correctness and citation, retrieval hit@3, refusals on out-of-scope questions, injection resistance, PII leakage | `evals/` |

## Run it

```bash
uv sync --extra dev --extra embeddings
cp .env.example .env            # add a free Groq API key, or set LLM_PROVIDER=mock
docker compose up -d db         # PostgreSQL 16 (or set DATABASE_URL=sqlite:///./data/dip.db)
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API.

```bash
# 1. Upload v1, generate test cases for section 3
curl -F name="Clinical Data Management SOP" -F file=@samples/sop_v1.md localhost:8000/documents
curl -X POST "localhost:8000/documents/Clinical%20Data%20Management%20SOP/test-cases" \
     -H "content-type: application/json" -d '{"node_keys": ["3-data-validation"]}'

# 2. Upload v2 (query deadline 5 -> 3 days): version 1.1.0, one test case goes stale
curl -F name="Clinical Data Management SOP" -F file=@samples/sop_v2.md localhost:8000/documents

# 3. Ask a question, or give the agent a goal
curl -X POST localhost:8000/ask -H "content-type: application/json" \
     -d '{"question": "How many days does a site have to answer data queries?"}'
curl -X POST localhost:8000/agent -H "content-type: application/json" \
     -d '{"goal": "What changed in the latest version, and which test cases need updating?"}'
```

## Tests and evals

```bash
uv run pytest                                              # 23 tests, offline (mock LLM, SQLite)
TEST_DATABASE_URL=postgresql+psycopg://dip:dip@localhost:5432/dip uv run pytest   # same, on PostgreSQL
uv run python -m evals.run_evals                           # against the model in .env
```

The offline mock LLM makes tests deterministic. It is not a quality measure: run the evals against
a real model to measure answer quality.

## Design decisions

- **RAG, not fine-tuning:** documents change often and answers must cite sources. Re-indexing a new
  version takes seconds; fine-tuning would need retraining and gives no citations.
- **Hybrid retrieval:** embeddings handle paraphrases; BM25 catches exact codes and names.
- **Stable node keys from heading paths:** readable in citations, and they survive re-uploads so
  versions can be compared node by node.
- **Validation and retry over trust:** the model's output is checked against the excerpts it saw;
  ungrounded output is retried once, then refused.
- **OpenAI-compatible client:** the same code runs on open-source models (Llama, Qwen via Groq or a
  local Ollama) or a hosted model, chosen in `.env`.

## Known limitations

- Heading detection on PDFs uses font heuristics; unusual layouts (multi-column, headings same size
  as body text) can mis-nest sections. Tables are read as text.
- Injection detection is pattern-based, so it catches common phrasings, not every attack.
- Embeddings are stored as JSON and compared in Python, which is fine for thousands of chunks.
  For larger corpora, move them to `pgvector` with an approximate nearest-neighbour index.
- Similarity thresholds (0.6 to match, 0.95 for cosmetic) were set by hand on sample documents.
