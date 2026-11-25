"""QA test-case generation from selected sections, stored with traceability to source nodes."""

from sqlalchemy.orm import Session

from app.llm.client import LLMClient
from app.llm.generator import generate_test_cases
from app.models import DocumentVersion, GeneratedArtifact
from app.services.selection_service import content_nodes, expand_selection


class GenerationFailed(RuntimeError):
    pass


def generate_for_selection(
    db: Session, llm: LLMClient, version: DocumentVersion, node_keys: list[str], max_cases: int = 8
) -> list[GeneratedArtifact]:
    nodes = content_nodes(expand_selection(version, node_keys))
    if not nodes:
        raise GenerationFailed("The selection contains no paragraphs or list items.")
    context = {n.node_key: n.text for n in nodes}
    hashes = {n.node_key: n.content_hash for n in nodes}

    result, attempts, errors = generate_test_cases(llm, context, max_cases=max_cases)
    if result is None:
        raise GenerationFailed(f"LLM output failed validation after {attempts} attempts: {errors}")

    artifacts = []
    for tc in result.test_cases:
        art = GeneratedArtifact(
            document_id=version.document_id,
            version_id=version.id,
            kind="qa_test_case",
            payload=tc.model_dump() | {"generated_from_version": version.label, "attempts": attempts},
            source_hashes={k: hashes[k] for k in tc.source_node_keys},
        )
        db.add(art)
        artifacts.append(art)
    db.commit()
    return artifacts
