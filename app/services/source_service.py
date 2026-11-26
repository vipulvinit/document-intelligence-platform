"""Literature sources: polling them on a schedule, and searching them on demand.

poll_source(): asks Europe PMC for articles indexed since the source's watermark, oldest first,
ingests what is new or revised, then moves the watermark forward. Each poll is recorded as an
IngestionRun (counts, errors, timings) for monitoring and for the project report.

ingest_query(): the same pipeline for an arbitrary query. It is the building block for the
on-demand agent ("nothing in the knowledge base about this disease: go and look").

Cheap re-polls: an article we already hold is re-fetched only when Europe PMC's record changed
(index/revision/full-text dates), so a daily poll mostly costs one search request.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.constants import ContentKind, RunStatus, SourceKind
from app.core.logging import logger
from app.models import Article, IngestionRun, Source
from app.parsers.hierarchy_builder import build_hierarchy
from app.parsers.jats_parser import parse_jats
from app.services.ingestion_service import ArticleMeta, ingest_parsed, licence_allows_storing, set_retracted
from app.sources import europepmc
from app.sources.defaults import DEFAULT_SOURCES
from app.versioning.hashing import content_hash, file_hash

PAGE_SIZE = 25
MAX_EXAMINED_PER_POLL = 500  # unchanged articles are cheap to skip, but stop somewhere
RETRACTED_TYPES = {"retracted publication", "retraction of publication"}


def utcnow() -> datetime:
    return datetime.now(UTC)


# ── Source registry ──────────────────────────────────────────────────────────


def seed_default_sources(db: Session) -> int:
    existing = set(db.scalars(select(Source.name)).all())
    added = 0
    for spec in DEFAULT_SOURCES:
        if spec["name"] not in existing:
            db.add(Source(kind=SourceKind.EUROPEPMC, **spec))
            added += 1
    db.commit()
    return added


def due_sources(db: Session, now: datetime | None = None) -> list[Source]:
    now = now or utcnow()
    due = []
    for s in db.scalars(select(Source).where(Source.enabled.is_(True)).order_by(Source.id)).all():
        last = s.last_polled_at
        if last is not None and last.tzinfo is None:  # SQLite returns naive datetimes
            last = last.replace(tzinfo=UTC)
        if last is None or now - last >= timedelta(hours=s.poll_interval_hours):
            due.append(s)
    return due


# ── One article ──────────────────────────────────────────────────────────────


def _is_retracted(result: dict) -> bool:
    types = (result.get("pubTypeList") or {}).get("pubType") or []
    types = [types] if isinstance(types, str) else types
    return any(t.lower() in RETRACTED_TYPES for t in types)


def ingest_result(
    db: Session, client: europepmc.EuropePMCClient, result: dict, source_id: int | None = None
) -> str:
    """Ingest one search result. Returns new | updated | unchanged | retracted | skipped."""
    ext_id = europepmc.external_id(result)
    fp = europepmc.fingerprint(result)
    article = db.scalar(select(Article).where(Article.external_id == ext_id))

    if _is_retracted(result):
        if article and not article.retracted:
            set_retracted(db, article.id)
            return "retracted"
        return "skipped"
    if article and (article.extra or {}).get("epmc_fingerprint") == fp:
        return "unchanged"

    fields = europepmc.meta_fields(result)
    extra = {
        "epmc_fingerprint": fp,
        "epmc_source": result.get("source"),
        "epmc_id": result.get("id"),
        "first_index_date": result.get("firstIndexDate"),
    }
    meta = ArticleMeta(external_id=ext_id, source_id=source_id, extra=extra, **fields)

    xml = client.full_text_xml(result["pmcid"]) if europepmc.wants_full_text(result) else None
    parsed, raw_hash, kind, fetch_asset = None, None, None, None
    if xml:
        try:
            jats = parse_jats(xml)
        except Exception as exc:  # malformed XML: fall back to the abstract
            logger.warning(f"Could not parse full text of {ext_id}: {exc}")
            jats = None
        if jats and jats.blocks:
            # Keep whichever licence statement is usable: the XML's URL, else Europe PMC's "cc by"
            if jats.license and (licence_allows_storing(jats.license) or not meta.license):
                meta.license = jats.license
            meta.authors = meta.authors or jats.authors
            parsed = build_hierarchy(jats.blocks, title=meta.title, detect_title=False)
            raw_hash, kind = file_hash(xml), ContentKind.FULL_TEXT
            pmcid = result["pmcid"]
            fetch_asset = lambda href: client.figure(pmcid, href)  # noqa: E731
    if parsed is None:
        blocks = europepmc.abstract_blocks(result)
        if not blocks:
            return "skipped"  # no full text and no abstract: nothing to answer from
        parsed = build_hierarchy(blocks, title=meta.title, detect_title=False)
        raw_hash = content_hash(meta.title + "\n" + (result.get("abstractText") or ""))
        kind = ContentKind.ABSTRACT

    outcome = ingest_parsed(db, meta, parsed, raw_hash, kind, fetch_asset=fetch_asset)
    if outcome.status == "unchanged":  # same content: still remember the new fingerprint
        outcome.article.extra = (outcome.article.extra or {}) | extra
        db.commit()
    return outcome.status


# ── Many articles ────────────────────────────────────────────────────────────


def _new_run(db: Session, source_id: int | None, trigger: str, query: str) -> IngestionRun:
    run = IngestionRun(
        source_id=source_id, trigger=trigger, status=RunStatus.RUNNING, details={"query": query}
    )
    db.add(run)
    db.commit()
    return run


def _finish(db: Session, run: IngestionRun, details: dict) -> IngestionRun:
    run.finished_at = utcnow()
    run.details = (run.details or {}) | details
    if run.failed and (run.new or run.updated or run.unchanged):
        run.status = RunStatus.PARTIAL
    elif run.failed:
        run.status = RunStatus.FAILED
    else:
        run.status = RunStatus.SUCCESS
    db.commit()
    return run


def ingest_query(
    db: Session,
    query: str,
    *,
    limit: int,
    source_id: int | None = None,
    trigger: str = "manual",
    client: europepmc.EuropePMCClient | None = None,
) -> tuple[IngestionRun, str | None, bool]:
    """Search Europe PMC and ingest results until `limit` articles were added or updated.
    Returns (run, newest index date examined, whether every result was examined)."""
    client = client or europepmc.EuropePMCClient()
    run = _new_run(db, source_id, trigger, query)
    cursor, examined, last_date, exhausted = "*", 0, None, False
    statuses: dict[str, int] = {}
    article_ids: list[int] = []
    try:
        while True:
            page = client.search(query, cursor=cursor, page_size=PAGE_SIZE)
            run.found = page.hit_count
            for result in page.results:
                if run.new + run.updated >= limit or examined >= MAX_EXAMINED_PER_POLL:
                    break
                examined += 1
                try:
                    status = ingest_result(db, client, result, source_id)
                except Exception as exc:
                    db.rollback()
                    run.failed += 1
                    run.errors = (run.errors or [])[:49] + [f"{europepmc.external_id(result)}: {exc}"]
                    logger.warning(f"Failed to ingest {europepmc.external_id(result)}: {exc}")
                    status = "failed"
                statuses[status] = statuses.get(status, 0) + 1
                if status == "new":
                    run.new += 1
                elif status == "updated":
                    run.updated += 1
                elif status in ("unchanged", "skipped", "retracted"):
                    run.unchanged += 1
                if status in ("new", "updated"):
                    found = db.scalar(
                        select(Article.id).where(Article.external_id == europepmc.external_id(result))
                    )
                    if found:
                        article_ids.append(found)
                last_date = result.get("firstIndexDate") or last_date
            else:
                if page.next_cursor:
                    cursor = page.next_cursor
                    db.commit()
                    continue
                exhausted = True
            break
    except Exception as exc:  # the search itself failed (network, server down)
        db.rollback()
        run.failed += 1
        run.errors = (run.errors or []) + [f"search: {exc}"]
        logger.error(f"Europe PMC search failed: {exc}")
    _finish(db, run, {"examined": examined, "statuses": statuses, "article_ids": article_ids})
    return run, last_date, exhausted


def poll_source(
    db: Session, source: Source, trigger: str = "schedule", client: europepmc.EuropePMCClient | None = None
) -> IngestionRun:
    s = get_settings()
    since = source.watermark or (utcnow() - timedelta(days=s.initial_lookback_days)).date().isoformat()
    query = f"({source.query}) AND FIRST_IDATE:[{since} TO 3000-12-31]"
    logger.info(f"Polling source {source.name!r} since {since}")
    run, last_date, exhausted = ingest_query(
        db, query, limit=s.max_articles_per_poll, source_id=source.id, trigger=trigger, client=client
    )
    search_failed = any(e.startswith("search:") for e in (run.errors or []))
    if not search_failed:
        # Oldest first, so everything before last_date is done. If every result was examined,
        # start the next poll from today; otherwise resume from the last date reached
        # (inclusive: articles of that day are re-checked cheaply by fingerprint).
        if exhausted:
            source.watermark = utcnow().date().isoformat()
        elif last_date:
            source.watermark = last_date
        source.last_polled_at = utcnow()
    db.commit()
    logger.info(
        f"Source {source.name!r}: {run.new} new, {run.updated} updated, {run.unchanged} unchanged, "
        f"{run.failed} failed; watermark now {source.watermark}"
    )
    return run


def poll_due_sources(db: Session, client: europepmc.EuropePMCClient | None = None) -> list[IngestionRun]:
    return [poll_source(db, s, client=client) for s in due_sources(db)]
