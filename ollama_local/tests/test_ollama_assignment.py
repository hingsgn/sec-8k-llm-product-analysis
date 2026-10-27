"""Deterministic tests use explicitly synthetic sources; never submission data."""
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "assignment2" if (PROJECT / "assignment2").is_dir() else PROJECT))
import sec_8k_ollama as m

SOURCE = "Example Robotics Inc. today announced the launch of Orbit One, a new warehouse inspection robot. Orbit One detects damaged shelves."
PRODUCT = {"name": "Orbit One", "description": "A warehouse inspection robot that detects damaged shelves.",
           "organization": "Example Robotics Inc.", "evidence": SOURCE.split(". Orbit One")[0] + "."}
PARSED = {"entities": [{"text": "Example Robotics Inc.", "label": "ORGANIZATION"}, {"text": "Orbit One", "label": "PRODUCT"}], "products": [PRODUCT]}


def filing(n=1):
    return m.Filing("Example Robotics Inc.", "EXM", "123", f"0000000123-26-{n:06d}", "2026-01-13", "2026-01-13", "test_fixture")


def sgml(text=SOURCE, accepted="20260113173000"):
    return f"""<SEC-DOCUMENT><ACCEPTANCE-DATETIME>{accepted}
<DOCUMENT><TYPE>8-K
<FILENAME>main.htm
<TEXT><html><body><p>Item 8.01 Other Events.</p><p>See press release.</p></body></html></TEXT></DOCUMENT>
<DOCUMENT><TYPE>EX-99.1
<FILENAME>release.htm
<TEXT><html><body>{text}<script>ignore script</script><ix:hidden>hidden fact</ix:hidden></body></html></TEXT></DOCUMENT>
<DOCUMENT><TYPE>EX-10.1
<FILENAME>contract.htm
<TEXT>Do not analyze this contract</TEXT></DOCUMENT></SEC-DOCUMENT>"""


def test_sgml_reads_exhibits_and_actual_acceptance_timestamp():
    docs, timestamp = m.parse_submission(sgml(), filing(), None)
    assert [d["type"] for d in docs] == ["8-K", "EX-99.1"]
    assert docs[1]["text"] == SOURCE
    assert timestamp == "2026-01-13T17:30:00-05:00"
    assert docs[1]["url"].endswith("/release.htm")


def test_timestamp_uses_daylight_saving_offset():
    _, timestamp = m.parse_submission(sgml(accepted="20260713173000"), filing(), None)
    assert timestamp == "2026-07-13T17:30:00-04:00"


def test_block_page_cannot_be_successful_filing():
    with pytest.raises(m.ExtractionError, match="No readable primary"):
        m.parse_submission("<html>SEC access denied</html>", filing(), None)


def test_late_release_not_lost_by_first_characters_truncation():
    text = "Old boilerplate. " * 1000 + SOURCE + " End boilerplate." * 500
    passages = m.select_passages(text, 4000)
    assert any(SOURCE in p["text"] for p in passages)
    assert all(len(p["text"]) <= 4000 for p in passages)


def test_validated_product_has_ner_name_quote_and_short_description():
    rows, rejected = m.validate_products(PARSED, SOURCE)
    assert len(rows) == 1 and not rejected
    assert rows[0]["name"] == "Orbit One"


@pytest.mark.parametrize("change", [
    {"name": "Invented Product"}, {"evidence": "This quotation is invented."},
    {"organization": "Imaginary Owner"}, {"description": ""},
    {"evidence": "Orbit One detects damaged shelves."},
])
def test_unsupported_product_is_not_written(change):
    parsed = {"entities": PARSED["entities"], "products": [{**PRODUCT, **change}]}
    rows, rejected = m.validate_products(parsed, SOURCE)
    assert not rows and len(rejected) == 1


def test_no_product_ner_cannot_be_silently_accepted():
    parsed = {"entities": [], "products": [PRODUCT]}
    assert not m.validate_products(parsed, SOURCE)[0]


def test_description_is_strictly_below_180_and_pipe_safe():
    value = m.shorten_description("Useful robot | " + "features " * 100)
    assert len(value) < 180 and "|" not in value
    assert m.shorten_description("a" * 179) == "a" * 179
    assert len(m.shorten_description("a" * 180)) < 180


def test_shape_rejects_products_string():
    with pytest.raises(ValueError):
        m.validate_shape({"entities": [], "products": "no products"})


class FixtureSEC:
    """No network and no external personal information in these tests."""
    def __init__(self, cfg):
        self.calls = 0
    def text(self, url):
        self.calls += 1
        if "000001.txt" in url:
            return sgml("Example Robotics Inc. reported quarterly financial results.")
        return sgml()


class FixtureNER:
    def __init__(self, model):
        self.version = "fixture-1"
    def entities(self, text):
        return []


class FixtureLLM:
    calls = 0
    def __init__(self, cfg):
        self.cfg = cfg
    def preflight(self):
        return {"model": self.cfg.model, "digest": "synthetic-fixture", "details": {}}
    def extract(self, company, ticker, text, hints):
        type(self).calls += 1
        if "Orbit One" in text:
            return PARSED, "fixture-response"
        return {"entities": [], "products": []}, "fixture-negative"


def patch_runtime(monkeypatch):
    monkeypatch.setattr(m, "SECClient", FixtureSEC)
    monkeypatch.setattr(m, "SpacyNER", FixtureNER)
    monkeypatch.setattr(m, "OllamaLLM", FixtureLLM)
    monkeypatch.setattr(m, "discover_filings", lambda client, cfg, root: [filing(1), filing(2), filing(3)])


def test_pipeline_exports_only_real_accepted_rows_and_resumes(tmp_path, monkeypatch):
    patch_runtime(monkeypatch)
    cfg = m.Config(email="fixture@example.test", output_dir=str(tmp_path), target_rows=1, min_filings=2)
    stats = m.run_pipeline(cfg)
    assert stats["analyzed_filings"] == 2 and stats["product_rows"] == 1
    assert stats["filings_without_products"] == 1
    assert not stats["assignment_size_requirement_met"]
    with (tmp_path / "products.csv").open(encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames == m.COLUMNS
    assert len(rows) == 1 and rows[0]["new_product"] == "Orbit One"
    assert len(list(csv.DictReader((tmp_path / "manual_review.csv").open(encoding="utf-8-sig")))) == 1
    before = FixtureLLM.calls
    again = m.run_pipeline(cfg)
    assert again["analyzed_filings"] == 2 and FixtureLLM.calls == before
    with pytest.raises(m.ExtractionError, match="settings changed"):
        m.run_pipeline(replace(cfg, model="different-local-model"))
    assert "fixture@example.test" not in (tmp_path / "run_config.json").read_text()
    from pypdf import PdfReader
    assert len(PdfReader(tmp_path / "methodology_report.pdf").pages) == 2


def test_resume_sample_preserves_manual_labels(tmp_path, monkeypatch):
    patch_runtime(monkeypatch)
    cfg = m.Config(email="fixture@example.test", output_dir=str(tmp_path), target_rows=1, min_filings=2)
    m.run_pipeline(cfg)
    path = tmp_path / "manual_review.csv"
    with path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["correct_yes_no"] = "yes"
    m.write_csv(path, list(rows[0]), rows)
    m.run_pipeline(cfg)
    with path.open(encoding="utf-8-sig") as stream:
        assert next(csv.DictReader(stream))["correct_yes_no"] == "yes"


def test_deduplication_keeps_versions_and_different_issuers():
    base = {**dict(zip(m.COLUMNS, ["Example Robotics", "EXM", "2026-01-13", "Orbit One", "Robot"])), "description_shortened": False}
    records = []
    for i, row in enumerate([base, base, {**base, "new_product": "Orbit Two"}, {**base, "stock_name": "OTH"}]):
        records.append({"status": "analyzed", "filing": {"stock_name": row["stock_name"], "filing_date": "2026-01-13"},
                        "rows": [row], "source_documents": 1, "passages": 1, "rejected": []})
    rows, stats = m.summarize(records, m.Config(), {})
    assert len(rows) == 3 and stats["duplicate_product_mentions_removed"] == 1


def test_chunk_limit_is_failure_not_a_false_negative():
    cfg = m.Config(max_chunks_per_filing=1, chunk_chars=1000)
    class SEC:
        def text(self, url): return sgml(SOURCE * 100)
    with pytest.raises(m.ExtractionError, match="passages exceed"):
        m.process_filing(filing(), SEC(), None, None, cfg)
