"""Chunking: group the paragraphs of each section into ~800-character chunks.

Chunks never cross a section boundary, so each one stays on one topic. Every chunk starts
with its heading path ("Results > Outcomes") so the embedding knows the context.
A paragraph longer than the limit is split into overlapping windows.

Figures and tables are always a chunk of their own. A figure's chunk is its caption plus the
vision model's description; a table's chunk is its caption plus the start of the table (enough
for its column names), while the answer model later sees the whole table.
"""

from dataclasses import dataclass

from app.core.constants import MEDIA_KINDS, STRUCTURAL_KINDS, NodeKind


@dataclass
class ChunkDraft:
    heading_path: str
    node_keys: list[str]
    body: str
    kind: str = "text"

    @property
    def text(self) -> str:
        return f"{self.heading_path}\n{self.body}" if self.heading_path else self.body


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    parts, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # try not to cut words
            space = text.rfind(" ", start + size // 2, end)
            end = space if space > start else end
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return parts


def build_chunks(nodes, size: int = 800, overlap: int = 120) -> list[ChunkDraft]:
    """`nodes` are ordered objects with node_key, kind, text, heading_path."""
    chunks: list[ChunkDraft] = []
    current: ChunkDraft | None = None
    for n in nodes:
        if n.kind in STRUCTURAL_KINDS:
            current = None  # a new section starts
            continue
        if n.kind in MEDIA_KINDS:
            kind = "figure" if n.kind == NodeKind.FIGURE else "table"
            body = n.text if len(n.text) <= size * 2 else n.text[: size * 2].rsplit("\n", 1)[0]
            chunks.append(ChunkDraft(n.heading_path, [n.node_key], body, kind))
            current = None
            continue
        if len(n.text) > size:
            for piece in _split_long(n.text, size, overlap):
                chunks.append(ChunkDraft(n.heading_path, [n.node_key], piece))
            current = None
            continue
        if current and current.heading_path == n.heading_path and len(current.body) + len(n.text) + 1 <= size:
            current.body += "\n" + n.text
            current.node_keys.append(n.node_key)
        else:
            current = ChunkDraft(n.heading_path, [n.node_key], n.text)
            chunks.append(current)
    return chunks
