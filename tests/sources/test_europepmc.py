"""Scheduled ingestion against a fake Europe PMC server (httpx.MockTransport: no network)."""

import copy
from datetime import UTC, datetime

import httpx
import pytest

from app.models import Article, Source
from app.services import source_service
from app.services.qa_service import ask
from app.services.retrieval_service import search
from app.sources.europepmc import EuropePMCClient, abstract_blocks
from app.storage import get_storage
from tests.conftest import SAMPLES

FULL_TEXT = {
    "id": "99990001",
    "source": "MED",
    "pmid": "99990001",
    "pmcid": "PMC9990001",
    "doi": "10.0000/synthetic.0001",
    "title": "Oral rehydration in <i>Vibrio cholerae</i> infection: a synthetic trial",
    "authorString": "Rao A, Synthetic Trials Group.",
    "journalInfo": {"journal": {"title": "Journal of Synthetic Test Medicine"}},
    "firstPublicationDate": "2026-11-03",
    "firstIndexDate": "2026-11-20",
    "dateOfRevision": "2026-11-20",
    "isOpenAccess": "Y",
    "inEPMC": "Y",
    "license": "cc by",
    "abstractText": "Cholera causes rapid dehydration.",
    "pubTypeList": {"pubType": ["research-article", "Journal Article"]},
}
PREPRINT = {
    "id": "PPR123",
    "source": "PPR",
    "title": "Clinical features of a novel arbovirus outbreak (synthetic preprint)",
    "firstPublicationDate": "2026-11-21",
    "firstIndexDate": "2026-11-21",
    "isOpenAccess": "N",
    "inEPMC": "N",
    "abstractText": "<h4>Background</h4>A novel arbovirus caused fever and rash.<h4>Findings</h4>"
    "Most patients recovered within seven days with supportive care.",
}


class FakeEuropePMC:
    def __init__(self):
        self.results = [copy.deepcopy(FULL_TEXT), copy.deepcopy(PREPRINT)]
        self.xml = (SAMPLES / "sample_article.xml").read_bytes()
        self.full_text_status = 200
        self.search_status = 200
        self.requests: list[str] = []
        self.queries: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url.path)
        path = request.url.path
        if path.endswith("/search"):
            if self.search_status != 200:
                return httpx.Response(self.search_status)
            self.queries.append(request.url.params["query"])
            if request.url.params["cursorMark"] == "*":
                return httpx.Response(
                    200,
                    json={
                        "hitCount": len(self.results),
                        "nextCursorMark": "c1",
                        "resultList": {"result": self.results},
                    },
                )
            return httpx.Response(
                200,
                json={"hitCount": len(self.results), "nextCursorMark": "c1", "resultList": {"result": []}},
            )
        if path.endswith("/fullTextXML"):
            if self.full_text_status != 200:
                return httpx.Response(self.full_text_status)
            return httpx.Response(200, content=self.xml, headers={"content-type": "application/xml"})
        if "/bin/" in path:
            if path.endswith("synthetic-g001.jpg"):
                return httpx.Response(
                    200, content=b"\xff\xd8fake-jpeg", headers={"content-type": "image/jpeg"}
                )
            return httpx.Response(404)
        return httpx.Response(404)

    def client(self) -> EuropePMCClient:
        return EuropePMCClient(
            http=httpx.Client(transport=httpx.MockTransport(self.handler)), backoff_seconds=0
        )


@pytest.fixture
def fake():
    return FakeEuropePMC()


@pytest.fixture
def source(db):
    s = Source(name="Cholera", kind="europepmc", query="cholera treatment")
    db.add(s)
    db.commit()
    return s


def _article(db, ext_id) -> Article:
    return db.query(Article).filter_by(external_id=ext_id).one()


def test_poll_ingests_full_text_figures_and_preprint_abstracts(db, source, fake):
    run = source_service.poll_source(db, source, client=fake.client())
    assert (run.status, run.new, run.failed) == ("success", 2, 0)
    assert "FIRST_IDATE:[" in fake.queries[0] and "(cholera treatment)" in fake.queries[0]

    full = _article(db, "PMC9990001")
    assert (
        full.versions[-1].content_kind == "full_text" and full.journal == "Journal of Synthetic Test Medicine"
    )
    assert full.license == "https://creativecommons.org/licenses/by/4.0/"  # taken from the XML
    fig = next(n for n in full.versions[-1].nodes if n.kind == "figure")
    assert fig.object_key and get_storage().get(fig.object_key) == b"\xff\xd8fake-jpeg"
    table = next(n for n in full.versions[-1].nodes if n.kind == "table")
    assert "Stool output (ml/kg)" in table.text

    pre = _article(db, "PPR:PPR123")
    assert pre.is_preprint and pre.versions[-1].content_kind == "abstract"
    assert [n.node_key for n in pre.versions[-1].nodes if n.kind == "heading"] == [
        "abstract",
        "abstract/background",
        "abstract/findings",
    ]
    assert source.watermark == datetime.now(UTC).date().isoformat()
    assert search(db, "stool output table oral rehydration")[0].article_id == full.id


def test_second_poll_skips_unchanged_articles_without_refetching(db, source, fake):
    source_service.poll_source(db, source, client=fake.client())
    fake.requests.clear()
    run = source_service.poll_source(db, source, client=fake.client())
    assert (run.new, run.updated, run.unchanged) == (0, 0, 2)
    assert not any(p.endswith("/fullTextXML") for p in fake.requests)  # fingerprint unchanged
    assert len(_article(db, "PMC9990001").versions) == 1


def test_revised_article_gets_a_new_version_and_answers_citing_it_go_stale(db, source, fake):
    from app.llm.client import MockLLM

    source_service.poll_source(db, source, client=fake.client())
    result = ask(
        db, MockLLM(), "Is low-osmolarity oral rehydration solution effective for adults with cholera?"
    )
    assert result["grounded"], result
    assert result["citations"][0]["external_id"] == "PMC9990001"

    fake.results[0]["dateOfRevision"] = "2026-11-24"
    fake.xml = fake.xml.replace(b"is effective for adults", b"is not effective for adults")
    run = source_service.poll_source(db, source, client=fake.client())
    assert run.updated == 1
    article = _article(db, "PMC9990001")
    assert article.versions[-1].label == "1.1.0"
    assert article.versions[-1].change_summary["stale_answer_ids"] == [result["answer_id"]]


def test_full_text_failure_falls_back_to_abstract(db, source, fake):
    fake.full_text_status = 500
    run = source_service.poll_source(db, source, client=fake.client())
    assert run.new == 2
    assert _article(db, "PMC9990001").versions[-1].content_kind == "abstract"


def test_retracted_article_is_excluded_from_search(db, source, fake):
    source_service.poll_source(db, source, client=fake.client())
    fake.results[0]["pubTypeList"] = {"pubType": ["Retracted Publication"]}
    fake.results[0]["dateOfRevision"] = "2026-11-25"
    source_service.poll_source(db, source, client=fake.client())
    article = _article(db, "PMC9990001")
    assert article.retracted
    assert all(h.article_id != article.id for h in search(db, "oral rehydration cholera stool output"))


def test_search_outage_is_recorded_and_watermark_kept(db, source, fake):
    fake.search_status = 503
    source.watermark = "2026-11-01"
    db.commit()
    run = source_service.poll_source(db, source, client=fake.client())
    assert run.status == "failed" and run.errors
    assert source.watermark == "2026-11-01"


def test_due_sources_respects_interval(db, source):
    assert source_service.due_sources(db) == [source]
    source.last_polled_at = datetime.now(UTC)
    db.commit()
    assert source_service.due_sources(db) == []


def test_abstract_blocks_split_headings():
    blocks = abstract_blocks(PREPRINT)
    assert [b.text for b in blocks if b.heading_level] == ["Abstract", "Background", "Findings"]
