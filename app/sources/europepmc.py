"""Europe PMC connector (free, no API key).

One API covers PubMed abstracts (source MED), PubMed Central open-access full text (with
figures and tables, as JATS XML) and preprints from medRxiv/bioRxiv and others (source PPR).
Docs: https://europepmc.org/RestfulWebService

  search()          one page of results, oldest first, with a cursor for the next page
  full_text_xml()   JATS XML of an open-access article, or None (not open access, or server error)
  figure()          a figure image from PubMed Central, given the file name in the XML

The server is shared by everyone: requests carry a descriptive User-Agent, transient errors
(429/5xx) are retried with backoff, and callers keep the page size small.
"""

import html
import re
import time
from dataclasses import dataclass

import httpx

from app.core.config import get_settings
from app.core.logging import logger
from app.schemas.node import RawBlock

RETRY_STATUSES = {429, 500, 502, 503, 504}
_TAG = re.compile(r"<[^>]+>")
_ABSTRACT_HEADING = re.compile(r"<h[1-6][^>]*>(.*?)</h[1-6]>", re.IGNORECASE | re.DOTALL)


@dataclass
class SearchPage:
    hit_count: int
    results: list[dict]
    next_cursor: str | None


class EuropePMCClient:
    def __init__(self, http: httpx.Client | None = None, max_retries: int = 3, backoff_seconds: float = 2.0):
        s = get_settings()
        self.base = s.europepmc_base_url.rstrip("/")
        self.figure_base = s.pmc_figure_base_url.rstrip("/")
        self.http = http or httpx.Client(
            timeout=s.http_timeout_seconds, headers={"User-Agent": s.http_user_agent}, follow_redirects=True
        )
        self.max_retries = max_retries
        self.backoff = backoff_seconds

    def _get(self, url: str, params: dict | None = None) -> httpx.Response | None:
        """GET with retries on transient failures. Returns None on 404 or persistent failure."""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.http.get(url, params=params)
            except httpx.HTTPError as exc:
                if attempt == self.max_retries:
                    raise
                logger.warning(f"Europe PMC request failed ({exc}); retrying")
            else:
                if resp.status_code == 200:
                    return resp
                if resp.status_code not in RETRY_STATUSES or attempt == self.max_retries:
                    logger.info(f"Europe PMC {resp.status_code} for {url}")
                    return None
            time.sleep(self.backoff * (2**attempt))
        return None

    def search(self, query: str, cursor: str = "*", page_size: int = 25) -> SearchPage:
        resp = self._get(
            f"{self.base}/search",
            {
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": page_size,
                "cursorMark": cursor,
                "sort": "FIRST_IDATE_D asc",
            },
        )
        if resp is None:
            raise RuntimeError(f"Europe PMC search failed for query {query!r}")
        data = resp.json()
        results = data.get("resultList", {}).get("result", [])
        nxt = data.get("nextCursorMark")
        return SearchPage(int(data.get("hitCount", 0)), results, nxt if results and nxt != cursor else None)

    def full_text_xml(self, pmcid: str) -> bytes | None:
        resp = self._get(f"{self.base}/{pmcid}/fullTextXML")
        return resp.content if resp is not None and resp.content.strip().startswith(b"<") else None

    def figure(self, pmcid: str, href: str) -> tuple[bytes, str] | None:
        """PMC serves figures at /articles/instance/<number>/bin/<file>. Some XML omits the extension."""
        number = pmcid.removeprefix("PMC")
        names = [href] if "." in href.rsplit("/", 1)[-1] else [f"{href}.jpg", f"{href}.png", f"{href}.gif"]
        for name in names:
            resp = self._get(f"{self.figure_base}/{number}/bin/{name}")
            if resp is not None and resp.headers.get("content-type", "").startswith("image/"):
                ext = resp.headers["content-type"].split("/", 1)[1].split(";")[0].replace("jpeg", "jpg")
                return resp.content, ext
        return None


# ── Mapping a search result to our records ─────────────────────────────────────


def external_id(result: dict) -> str:
    """PMC id when there is one (full text possible), else "<source>:<id>" (MED:123, PPR:PPR456)."""
    if result.get("pmcid"):
        return result["pmcid"]
    return f"{result.get('source', 'EPMC')}:{result.get('id')}"


def fingerprint(result: dict) -> str:
    """Changes when Europe PMC revises the record or receives new full text: then we re-fetch."""
    return "|".join(
        str(result.get(k) or "") for k in ("firstIndexDate", "dateOfRevision", "fullTextReceivedDate")
    )


def wants_full_text(result: dict) -> bool:
    return bool(result.get("pmcid")) and result.get("isOpenAccess") == "Y" and result.get("inEPMC") == "Y"


def article_url(result: dict) -> str:
    src, rid = result.get("source", "MED"), result.get("id")
    if result.get("pmcid"):
        return f"https://europepmc.org/article/PMC/{result['pmcid']}"
    return f"https://europepmc.org/article/{src}/{rid}"


def meta_fields(result: dict) -> dict:
    journal = (result.get("journalInfo") or {}).get("journal") or {}
    return {
        "title": _plain(result.get("title") or "") or "Untitled",
        "pmid": result.get("pmid"),
        "pmcid": result.get("pmcid"),
        "doi": result.get("doi"),
        "journal": journal.get("title") or (result.get("bookOrReportDetails") or {}).get("publisher"),
        "authors": (result.get("authorString") or "")[:2000] or None,
        "published_date": result.get("firstPublicationDate") or result.get("pubYear"),
        "url": article_url(result),
        "license": result.get("license"),
        "is_preprint": result.get("source") == "PPR",
    }


def _plain(text: str) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", text)).split())


def abstract_blocks(result: dict) -> list[RawBlock]:
    """The abstract as blocks: "Abstract" heading, then its sub-headings (Background, Methods...)
    and paragraphs. Used when full text is not available."""
    raw = result.get("abstractText") or ""
    if not raw.strip():
        return []
    blocks = [RawBlock(text="Abstract", heading_level=1)]
    parts = _ABSTRACT_HEADING.split(raw)  # [text, heading, text, heading, text...]
    for i, part in enumerate(parts):
        if i % 2 == 1:
            if title := _plain(part).rstrip(":"):
                blocks.append(RawBlock(text=title, heading_level=2))
            continue
        for para in re.split(r"(?:<br\s*/?>|</p>|\n\s*\n)", part, flags=re.IGNORECASE):
            if text := _plain(para):
                blocks.append(RawBlock(text=text))
    return blocks
