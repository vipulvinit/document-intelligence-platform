"""Rebuilds the document's section tree from flat blocks.

Heading detection for PDFs: a block is a heading if its font is clearly larger than the
body text, or it is short, bold and doesn't end like a sentence. Larger fonts map to
higher levels. Each node gets a stable key built from its heading path, for example
"3-data-validation/edit-checks/p2". Figures and tables get keys from their label, for
example "results/figure-2", and are never merged into paragraphs.
"""

import re
from collections import Counter

from app.core.constants import NodeKind
from app.schemas.node import ParsedDocument, ParsedNode, RawBlock
from app.versioning.hashing import content_hash


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "section"


def _assign_heading_levels(blocks: list[RawBlock]) -> list[int | None]:
    """Return a heading level per block (None = body text)."""
    if any(b.heading_level for b in blocks):  # Markdown: levels are explicit
        return [b.heading_level for b in blocks]

    sizes = [b.font_size for b in blocks if b.font_size]
    if not sizes:
        return [None] * len(blocks)
    body_size = Counter(sizes).most_common(1)[0][0]

    def is_heading(b: RawBlock) -> bool:
        if b.media_kind is not None:
            return False
        short = len(b.text) <= 90 and not b.text.endswith((".", ":", ";", ","))
        bigger = (b.font_size or 0) >= body_size + 1.5
        return short and (bigger or (b.bold and not b.is_list_item))

    heading_sizes = sorted({b.font_size or body_size for b in blocks if is_heading(b)}, reverse=True)
    size_to_level = {s: min(i + 1, 6) for i, s in enumerate(heading_sizes)}
    return [size_to_level[b.font_size or body_size] if is_heading(b) else None for b in blocks]


def _media_text(block: RawBlock) -> str:
    head = f"{block.label}: {block.text}" if block.label else block.text
    if block.table_markdown:
        return f"{head}\n{block.table_markdown}"
    return head


def build_hierarchy(blocks: list[RawBlock], title: str, detect_title: bool = True) -> ParsedDocument:
    """`detect_title=False` when the title is already known (JATS XML gives it in the metadata)."""
    levels = _assign_heading_levels(blocks)
    # A single top-level heading at the very start is the document's own title: use it as the
    # root instead of nesting every section under it (keeps node keys short and stable).
    heading_levels = [lv for lv in levels if lv is not None]
    if detect_title and levels and levels[0] is not None and heading_levels.count(min(heading_levels)) == 1:
        title = blocks[0].text
        blocks, levels = blocks[1:], levels[1:]
    root_key = "root"
    nodes: list[ParsedNode] = [
        ParsedNode(
            node_key=root_key,
            parent_key=None,
            kind=NodeKind.DOCUMENT,
            level=0,
            title=title,
            text=title,
            heading_path="",
            position=0,
            content_hash=content_hash(title),
        )
    ]
    # stack of (level, key, title) for currently open headings
    stack: list[tuple[int, str, str]] = [(0, root_key, title)]
    used_keys: Counter = Counter()
    child_counts: Counter = Counter()

    def unique(key: str) -> str:
        used_keys[key] += 1
        return key if used_keys[key] == 1 else f"{key}-{used_keys[key]}"

    for block, level in zip(blocks, levels):
        if level is not None:
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_level, parent_key, _ = stack[-1]
            key = unique(
                f"{parent_key}/{slugify(block.text)}" if parent_key != root_key else slugify(block.text)
            )
            path = " > ".join([t for _, _, t in stack[1:]] + [block.text])
            nodes.append(
                ParsedNode(
                    node_key=key,
                    parent_key=parent_key,
                    kind=NodeKind.HEADING,
                    level=level,
                    title=block.text,
                    text=block.text,
                    heading_path=path,
                    page=block.page,
                    position=len(nodes),
                    content_hash=content_hash(block.text),
                )
            )
            stack.append((level, key, block.text))
        elif block.media_kind is not None:
            parent_level, parent_key, _ = stack[-1]
            kind = NodeKind(block.media_kind)
            child_counts[(parent_key, kind)] += 1
            fallback = f"{'fig' if kind == NodeKind.FIGURE else 'table'}{child_counts[(parent_key, kind)]}"
            local = slugify(block.label) if block.label else fallback
            key = unique(f"{parent_key}/{local}" if parent_key != root_key else local)
            text = _media_text(block)
            nodes.append(
                ParsedNode(
                    node_key=key,
                    parent_key=parent_key,
                    kind=kind,
                    level=min(parent_level + 1, 6),
                    title=block.label,
                    text=text,
                    heading_path=" > ".join(t for _, _, t in stack[1:]),
                    page=block.page,
                    position=len(nodes),
                    content_hash=content_hash(text),
                    asset_ref=block.asset_ref,
                    asset_bytes=block.asset_bytes,
                    asset_ext=block.asset_ext,
                    extra={
                        "label": block.label,
                        "caption": block.text,
                        **({"table_markdown": block.table_markdown} if block.table_markdown else {}),
                    },
                )
            )
        else:
            parent_level, parent_key, _ = stack[-1]
            child_counts[parent_key] += 1
            prefix = "li" if block.is_list_item else "p"
            key = unique(f"{parent_key}/{prefix}{child_counts[parent_key]}")
            nodes.append(
                ParsedNode(
                    node_key=key,
                    parent_key=parent_key,
                    kind=NodeKind.LIST_ITEM if block.is_list_item else NodeKind.PARAGRAPH,
                    level=min(parent_level + 1, 6),
                    text=block.text,
                    heading_path=" > ".join(t for _, _, t in stack[1:]),
                    page=block.page,
                    position=len(nodes),
                    content_hash=content_hash(block.text),
                )
            )
    return ParsedDocument(title=title, nodes=nodes)
