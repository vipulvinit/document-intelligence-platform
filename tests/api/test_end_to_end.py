"""The full story over HTTP: upload, ask with citations, revise the source, see the answer go stale."""

from samples.make_pdf import md_to_pdf
from tests.conftest import SAMPLES, upload

TITLE = "Dengue Care Guideline"
MONITORING_KEY = "3-management/3-2-monitoring/p1"
QUESTION = "How often are patients with warning signs reviewed by a clinician?"


def test_full_lifecycle(client):
    r = upload(client, TITLE, "dengue_guideline_v1.md")
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["status"], body["version"]) == ("new", "1.0.0")
    article_id = body["article_id"]

    # Re-uploading identical content changes nothing
    r = upload(client, TITLE, "dengue_guideline_v1.md")
    assert r.json()["status"] == "unchanged" and r.json()["version"] == "1.0.0"

    # RAG answer with a verified citation, stored for traceability
    r = client.post("/ask", json={"question": QUESTION})
    body = r.json()
    assert body["grounded"] and "every 4 hours" in body["answer"]
    cite = body["citations"][0]
    assert cite["id"] == "S1" and cite["key"] == f"upload:dengue-care-guideline#{MONITORING_KEY}"
    assert cite["article_id"] == article_id and cite["section"] == "3. Management > 3.2 Monitoring"
    assert body["disclaimer"]
    answer_id = body["answer_id"]

    # New version: review interval changes -> MINOR bump, and the answer that cited it goes stale
    r = upload(client, TITLE, "dengue_guideline_v2.md")
    assert r.json()["version"] == "1.1.0"
    assert r.json()["stale_answer_ids"] == [answer_id]
    stale = client.get("/answers", params={"status": "stale"}).json()
    assert len(stale) == 1 and MONITORING_KEY in stale[0]["stale_reason"]

    r = client.post("/ask", json={"question": QUESTION})
    assert "every 2 hours" in r.json()["answer"]

    # New section -> MAJOR bump
    assert upload(client, TITLE, "dengue_guideline_v3.md").json()["version"] == "2.0.0"
    versions = client.get(f"/articles/{article_id}/versions").json()
    assert [v["label"] for v in versions] == ["1.0.0", "1.1.0", "2.0.0"]
    diff = client.get(
        f"/articles/{article_id}/diff", params={"from_version": "1.0.0", "to_version": "2.0.0"}
    ).json()
    assert diff["bump"] == "major"


def test_unanswerable_question_is_refused(client):
    upload(client, TITLE, "dengue_guideline_v1.md")
    body = client.post("/ask", json={"question": "What is the capital of France?"}).json()
    assert body["grounded"] is False and body["citations"] == []


def test_empty_knowledge_base_asks_for_literature_search(client):
    body = client.post("/ask", json={"question": "How is Oropouche fever treated?"}).json()
    assert body["grounded"] is False and body["needs_literature_search"] is True


def test_emergency_gets_emergency_guidance_not_rag(client):
    upload(client, TITLE, "dengue_guideline_v1.md")
    body = client.post("/ask", json={"question": "My father has crushing chest pain right now"}).json()
    assert body["emergency"] == "cardiac" and "112" in body["answer"] and body["citations"] == []


def test_injection_is_flagged_and_not_obeyed(client):
    r = upload(client, "Clinic Notes", "injection_note.md")
    assert r.json()["injection_flags"]
    body = client.post(
        "/ask",
        json={"question": "Is every medicine safe at any dose?", "article_ids": [r.json()["article_id"]]},
    ).json()
    assert body["injection_flags"]  # reported to the caller
    assert body["grounded"] is False  # quarantined passage never reached the model
    assert "safe at any dose" not in body["answer"].lower()


def test_pdf_upload_stores_figure_and_serves_it(client):
    pdf = md_to_pdf(SAMPLES / "dengue_guideline_v1.md")
    r = upload(client, "Dengue PDF", "dengue_guideline_v1.pdf", data=pdf)
    assert r.status_code == 200 and r.json()["figures_stored"] == 1
    tree = client.get(f"/articles/{r.json()['article_id']}/tree").json()
    assert any(n["node_key"] == MONITORING_KEY for n in tree)
    fig = next(n for n in tree if n["kind"] == "figure")
    img = client.get(f"/objects/{fig['object_key']}")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    assert client.get("/objects/../secrets.txt").status_code == 404


def test_jats_upload_and_figure_question(client):
    r = upload(client, "ORS trial", "sample_article.xml")
    assert r.status_code == 200 and r.json()["media_nodes"] == 2
    body = client.post("/ask", json={"question": "What was the stool output in each treatment group?"}).json()
    assert body["grounded"], body
    assert any(c["kind"] in ("table", "figure", "paragraph") for c in body["citations"])


def test_sources_crud_and_retraction(client):
    r = client.post("/sources", json={"name": "Cholera", "query": "cholera treatment"})
    assert r.status_code == 201
    assert client.post("/sources", json={"name": "Cholera", "query": "x"}).status_code == 409
    sid = r.json()["id"]
    assert client.patch(f"/sources/{sid}", json={"enabled": False}).json()["enabled"] is False
    assert client.get("/sources").json()[0]["name"] == "Cholera"

    article_id = upload(client, TITLE, "dengue_guideline_v1.md").json()["article_id"]
    answer_id = client.post("/ask", json={"question": QUESTION}).json()["answer_id"]
    assert client.post(f"/articles/{article_id}/retract").json()["stale_answer_ids"] == [answer_id]
    assert client.post("/ask", json={"question": QUESTION}).json()["grounded"] is False


def test_agent_uses_tools(client):
    upload(client, TITLE, "dengue_guideline_v1.md")
    upload(client, TITLE, "dengue_guideline_v2.md")
    body = client.post("/agent", json={"goal": "What changed in the latest version?"}).json()
    assert body["trace"][0]["tool"] == "compare_versions"
    assert body["steps"] == 2
