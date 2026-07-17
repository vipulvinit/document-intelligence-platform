# Document Intelligence Platform

A backend platform for intelligent document parsing, hierarchical reconstruction, document versioning, traceability, and AI-assisted QA test generation.

## Tech Stack

- FastAPI
- SQLAlchemy
- SQLite
- MongoDB (planned)
- PyMuPDF (planned)
- OCR fallback (planned)
- OpenAI-compatible LLMs (planned)

## Development

Install dependencies:

```bash
uv sync
uv sync --extra dev
```

Run the server:

```bash
uv run uvicorn app.main:app --reload
```