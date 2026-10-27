"""Build the self-contained submission notebook from the tested implementation."""
from pathlib import Path
import json
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
source=(root/'sec_product_extractor.py').read_text()
source=source[:source.index('\ndef main() -> None:')]
repo=(root/'repository_url.txt').read_text().strip() if (root/'repository_url.txt').exists() else 'Repository publication pending'
nb=nbf.v4.new_notebook()
cells=[]
def md(text):cells.append(nbf.v4.new_markdown_cell(text))
def code(text):cells.append(nbf.v4.new_code_cell(text))
md(f'''# Assignment 2 — LLM Document Analysis
**Hieu Nguyen | CAP-5619 | SEC Form 8-K product announcements**

GitHub: {repo}

This notebook identifies new product releases in SEC 8-K filings and their
attached press releases. A real local Qwen instruction-tuned LLM performs
ORGANIZATION and PRODUCT named entity recognition through Python. Authoritative
SEC metadata, source evidence, source-based descriptions, and documented review
decisions produce the required five-column CSV.

The included checkpoint files contain actual model responses. Running this
notebook with those files reproduces the reviewed output without another model
download. Running it without the files downloads the SEC sample and the model,
then performs new inference. No paid API or API key is required.''')
md('''## 1 Environment
Python 3.12 was used for the recorded LLM run. Checkpoint replay installs only
the small data-processing dependencies. A fresh inference run installs the CPU
LLM runtime when it is needed. This notebook uses llama.cpp rather than Ollama.''')
code('''import importlib.util
import subprocess
import sys

requirements = {
    "requests": "requests>=2.32,<3", "bs4": "beautifulsoup4>=4.12,<5",
    "pandas": "pandas>=2.2,<4"
}
missing = [package for module, package in requirements.items()
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "--prefer-binary", "--extra-index-url",
                           "https://abetlen.github.io/llama-cpp-python/whl/cpu", *missing])
print("Python:", sys.version.split()[0])
print("Dependencies available")''')
md('''## 2 Implementation
The following cells contain the complete implementation, rather than importing a
missing external script. HTTP requests share a global two-per-second limit and
raw-response caching. The official Submissions JSON API replaces the older Atom
lookup, and filing-detail pages identify the actual 8-K and EX-99 exhibits.''')
parts=[]
breaks=[source.index('\ndef collect_documents('),source.index('\nEXTRACTION_SCHEMA ='),source.index('\ndef analyze_documents(')]
positions=[0,*breaks,len(source)]
for a,b in zip(positions,positions[1:]):parts.append(source[a:b].strip())
for i,part in enumerate(parts):
    if i==1:md('### Download filings and select announcement passages')
    if i==2:md('### Query the LLM and ground recognized entities')
    if i==3:md('### Export results and enforce submission requirements')
    code(part)
md('''## 3 Configuration and data
The sample starts with ten Item 2.02 earnings 8-K filings for each of ten technology
issuers within 2023–2025, then adds two older filings each for NVIDIA and AMD. This enrichment strategy is intended to find product
updates in attached earnings releases; it is not a random or market-wide sample.

The next cell uses data in the current project folder, automatically detects an
uploaded `Assignment2_Submission.zip`, or downloads the project from GitHub.
No file-selection popup is required.''')
code('''from pathlib import Path
import zipfile
import urllib.request
import io

ROOT = Path.cwd()
if not (ROOT / "data" / "filing_manifest.json").exists():
    packages = sorted(ROOT.glob("Assignment2_Submission*.zip"))
    if packages:
        destination = ROOT / "assignment2_project"
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(packages[-1]) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe ZIP member path")
            archive.extractall(destination)
        ROOT = destination
    else:
        repository_archive = "https://github.com/hingsgn/sec-8k-llm-product-analysis/archive/refs/heads/main.zip"
        print("Downloading the published project data and saved LLM checkpoints")
        with urllib.request.urlopen(repository_archive, timeout=120) as response:
            archive_bytes = response.read()
        destination = ROOT / "assignment2_project"
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe ZIP member path")
            archive.extractall(destination)
        manifests = list(destination.glob("*/data/filing_manifest.json"))
        if len(manifests) != 1:
            raise FileNotFoundError("The project archive must contain one filing manifest")
        ROOT = manifests[0].parents[1]

SEC_USER_AGENT = "CAP5619-SEC8K-ProductAnalysis/1.0 AcademicResearch"
MODEL_PATH = ROOT / "models" / MODEL_FILE
REFRESH_SEC = False
RERUN_LLM = False
print("Project:", ROOT)
print("Tickers:", ", ".join(TICKERS))''')
code('''manifest_path = ROOT / "data" / "filing_manifest.json"
if REFRESH_SEC or not manifest_path.exists():
    documents = collect_documents(ROOT, SEC_USER_AGENT)
else:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = [json.loads((ROOT / "data" / "documents" /
                            f"{filing['accession_number']}.json").read_text(encoding="utf-8"))
                 for filing in manifest]

assert len(documents) >= 100
assert all(document["form"] == "8-K" for document in documents)
print("8-K filings:", len(documents))
print("Companies:", len({document["stock_name"] for document in documents}))
print("Primary documents and attached files:",
      sum(len(document["source_documents"]) for document in documents))''')
md('''## 4 Real LLM extraction and NER
Qwen2.5-3B-Instruct Q4_K_M returns schema-constrained JSON. The entity task is
performed by the LLM, not by a fixed list of product names. Company/ticker
metadata and accepted timestamps come from SEC records. Each product must map to
an actual source span and an announcement sentence. Descriptions use that source
text, with documented edits for clarity.

Set `RERUN_LLM=True` to create a new inference run. This preserves the old
checkpoints in a separate directory. A fresh run should be reviewed using the
evidence sidecar before submission; source matching alone is not proof of
semantic accuracy.''')
code('''import urllib.request
import shutil

if RERUN_LLM:
    current = ROOT / "data" / "llm_responses"
    if current.exists():
        backup = ROOT / "data" / ("llm_responses_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.move(str(current), str(backup))

missing_checkpoints = []
for document in documents:
    selected = select_analysis_source(document)
    passage = candidate_passage(selected["text"])
    key = hashlib.sha256((PROMPT_VERSION + selected["url"] + passage).encode()).hexdigest()
    if not (ROOT / "data" / "llm_responses" / f"{key}.json").exists():
        missing_checkpoints.append(document["accession_number"])

if missing_checkpoints and not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix(".part")
    print("Downloading public model weights, approximately 2.1 GB")
    with urllib.request.urlopen(MODEL_URL, timeout=120) as source, temporary.open("wb") as target:
        while chunk := source.read(8 * 1024 * 1024):
            target.write(chunk)
    assert temporary.stat().st_size == 2104932768, "Incomplete model download"
    temporary.replace(MODEL_PATH)

if missing_checkpoints and importlib.util.find_spec("llama_cpp") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "--prefer-binary", "--extra-index-url",
                           "https://abetlen.github.io/llama-cpp-python/whl/cpu",
                           "llama-cpp-python==0.3.36"])

print("New LLM calls needed:", len(missing_checkpoints))
summary = analyze_documents(ROOT, documents, MODEL_PATH, threads=min(8, os.cpu_count() or 2))
print(json.dumps(summary, indent=2))''')
md('''## 5 Required CSV and pipe-separated table
One row represents one accepted product entity. Multiple products in a filing can
produce multiple rows. Filings with no qualifying announcement remain recorded
in the manifest and inference log but do not become blank product rows.''')
code('''import pandas as pd
products = pd.read_csv(ROOT / "output" / "products.csv", dtype=str, keep_default_na=False)
display(products.head(10))
print("CSV rows:", len(products))
print(products.head(5).to_csv(sep="|", index=False))''')
md('''### NER and source audit
NER offsets refer to the exact passage in the associated inference checkpoint,
not to an independently reformatted web page. Review decisions exclude corporate
events, partner products, customer deployments, generic entities, and repeated
product references. Original model responses remain unchanged.''')
code('''entities = pd.DataFrame(json.loads((ROOT / "output" / "ner_entities.json").read_text()))
print(entities["entity_type"].value_counts().to_string())
display(entities[["entity_type", "text", "start", "end"]].head(12))
evidence = pd.DataFrame(json.loads((ROOT / "output" / "provenance.json").read_text()))
display(evidence[["stock_name", "new_product", "source_url", "evidence"]].head(3))''')
md('''### Validation
The release fails validation if it contains fewer than 100 product rows, any
missing required field, an incorrect CSV header, a description of 180 characters
or more, a timestamp without a timezone, an exact duplicate, or missing source
evidence. These checks measure formatting and grounding, not independent recall.''')
code('''validation = validate_output(ROOT, minimum_rows=100)
assert len(documents) >= 100
assert list(products.columns) == COLUMNS
assert all(len(value) < 180 for value in products["product_description"])
assert ground_product("Imaginary Product", "Acme reported quarterly earnings.")[0] is None
assert ground_product("Alpha Server", "Alpha Server revenue increased.")[0] is None
assert ground_product("Alpha Server", "Acme introduced Alpha Server for small offices.")[0] is not None
print(json.dumps(validation, indent=2))
print("All submission checks passed")''')
md('''### Methodology and limits
The report states the actual model, quantization, sampling method, document and
company counts, results, and challenges. Source-based review improves precision,
but the sample, selected exhibit, passage selection, and model recall can omit
announcements. The project does not claim complete coverage or 100% accuracy.

**Primary documentation:**
- https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- https://www.sec.gov/about/developer-resources
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
- https://github.com/abetlen/llama-cpp-python

AI assistance supported implementation, documentation, and source-evidence
review. The saved inference checkpoints record the local model's actual outputs.''')
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
             'language_info':{'name':'python','version':'3.12.14'}}
nbf.write(nb,root/'Assignment2_LLM_Document_Analysis.ipynb')
print(root/'Assignment2_LLM_Document_Analysis.ipynb')
