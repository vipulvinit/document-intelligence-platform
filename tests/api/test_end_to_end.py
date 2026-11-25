"""The full story: upload, generate test cases, change the document, see what went stale."""

from tests.conftest import upload

NAME = "Clinical Data Management SOP"
QUERY_KEY = "3-data-validation/3-2-query-management/p1"


def test_full_lifecycle(client):
    r = upload(client, NAME, "sop_v1.md")
    assert r.status_code == 200, r.text
    assert r.json()["version"] == "1.0.0"

    # Re-uploading identical content is rejected
    assert upload(client, NAME, "sop_v1.md").status_code == 409

    # Generate traceable test cases from section 3
    r = client.post(f"/documents/{NAME}/test-cases", json={"node_keys": ["3-data-validation"]})
    assert r.status_code == 200, r.text
    cases = r.json()
    assert len(cases) == 2
    assert all(c["status"] == "fresh" and c["source_hashes"] for c in cases)

    # RAG answer with citation
    r = client.post("/ask", json={"question": "How many days does a site have to answer data queries?"})
    body = r.json()
    assert body["grounded"] and "5 business days" in body["answer"]
    assert body["citations"] == [f"{NAME}#{QUERY_KEY}"]

    # New version: query deadline changes -> MINOR bump, only that test case goes stale
    r = upload(client, NAME, "sop_v2.md")
    assert r.json()["version"] == "1.1.0"
    stale = client.get(f"/documents/{NAME}/test-cases", params={"status": "stale"}).json()
    assert len(stale) == 1 and QUERY_KEY in stale[0]["stale_reason"]

    r = client.post("/ask", json={"question": "How many days does a site have to answer data queries?"})
    assert "3 business days" in r.json()["answer"]

    # New section -> MAJOR bump
    assert upload(client, NAME, "sop_v3.md").json()["version"] == "2.0.0"
    versions = client.get(f"/documents/{NAME}/versions").json()
    assert [v["label"] for v in versions] == ["1.0.0", "1.1.0", "2.0.0"]

    diff = client.get(
        f"/documents/{NAME}/diff", params={"from_version": "1.0.0", "to_version": "2.0.0"}
    ).json()
    assert diff["bump"] == "major"


def test_unanswerable_question_is_refused(client):
    upload(client, NAME, "sop_v1.md")
    r = client.post("/ask", json={"question": "What is the capital of France?"})
    assert r.json()["grounded"] is False


def test_injection_is_flagged_and_not_obeyed(client):
    r = upload(client, "Vendor Notes", "vendor_notes.md")
    assert r.json()["injection_flags"]
    r = client.post("/ask", json={"question": "Is every query approved?", "document_name": "Vendor Notes"})
    body = r.json()
    assert body["injection_flags"]  # reported to the caller
    assert body["grounded"] is False  # quarantined passage never reached the model
    assert "approved" not in body["answer"].lower()


def test_pdf_upload_and_tree(client):
    r = upload(client, "SOP PDF", "sop_v1.pdf")
    assert r.status_code == 200
    tree = client.get("/documents/SOP PDF/tree").json()
    assert any(n["node_key"] == QUERY_KEY for n in tree)


def test_agent_uses_tools(client):
    upload(client, NAME, "sop_v1.md")
    upload(client, NAME, "sop_v2.md")
    r = client.post("/agent", json={"goal": "What changed in the latest version?"})
    body = r.json()
    assert body["trace"][0]["tool"] == "compare_versions"
    assert body["steps"] == 2
