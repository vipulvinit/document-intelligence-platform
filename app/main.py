from fastapi import FastAPI

app = FastAPI(
    title="Document Intelligence Platform",
    description=(
        "Backend platform for document parsing, versioning, "
        "traceability, and AI-assisted QA generation."
    ),
    version="0.1.0",
)


@app.get("/", tags=["Health"])
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Welcome to the Document Intelligence Platform",
    }