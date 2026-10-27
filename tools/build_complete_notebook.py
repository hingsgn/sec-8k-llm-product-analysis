"""Build the complete Colab notebook, including report and ZIP export."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb = nbf.read(root / "Assignment2_LLM_Document_Analysis.ipynb", as_version=4)
nb.cells[0].source = """# Assignment 2 - Complete Submission
**Hieu Nguyen | CAP-5619 | SEC 8-K LLM Document Analysis**

GitHub: https://github.com/hingsgn/sec-8k-llm-product-analysis

**Run all cells in order. The default run needs no API key, Ollama server or GPU.**
It reproduces the reviewed CSV from 104 saved real local-LLM responses and generates
the two-page report, audit files and a downloadable submission ZIP. It does not
pretend to perform fresh inference when replaying checkpoints. The supplied data
contains 104 filings from 10 companies and 102 retained product rows.

The complete Python implementation is included below. To make new model calls,
set `RERUN_LLM=True` in Configuration; the notebook then downloads the free Qwen
GGUF model and uses llama.cpp directly through Python. No localhost server is used.
New inference outputs need renewed source review.

The archive and model evidence are historical (2023-2025); there is no requirement
in the assignment to use only the latest filings. Product releases reported for
the filing's covered period are eligible; the filing timestamp is not a launch date.

**Deliverables produced:** `products.csv` (exactly five columns), a pipe-separated
table, `Assignment2_Methodology_Report.pdf`, Python source and this notebook.
The private submission package also includes the prepared CV with the repository
URL at the top. Personal CV files are not published to GitHub.
"""
nb.cells[1].source = """## 1. Install only missing dependencies
Checkpoint replay is fast and does not load a model. The recorded inference used
Python 3.12 and Qwen2.5-3B-Instruct Q4_K_M; replay also works on newer Python.
"""
nb.cells[2].source = '''import importlib.util
import subprocess
import sys

requirements = {
    "requests": "requests>=2.32,<3",
    "bs4": "beautifulsoup4>=4.12,<5",
    "pandas": "pandas>=2.2,<4",
    "reportlab": "reportlab>=4,<5",
    "pypdf": "pypdf>=5,<7"
}
missing = [package for module, package in requirements.items()
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
print("Python:", sys.version.split()[0])
print("Dependencies ready. No paid API key is needed.")
'''
nb.cells[12].source = '''from pathlib import Path
import zipfile
import urllib.request
import io

# Keep False to reproduce the reviewed submission without new inference.
REFRESH_SEC = False
RERUN_LLM = False
SEC_CONTACT_EMAIL = ""  # Required only when REFRESH_SEC=True.
GITHUB_URL = "https://github.com/hingsgn/sec-8k-llm-product-analysis"
DATA_COMMIT = "c7aa65e30b77f13e763355aa38ee5323b2c169ac"

def unpack_project(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe ZIP member path")
        package.extractall(destination)
    candidates = [destination / "data/filing_manifest.json"]
    candidates += list(destination.glob("*/data/filing_manifest.json"))
    manifests = [path for path in candidates if path.exists()]
    if len(manifests) != 1:
        raise FileNotFoundError("Expected exactly one filing manifest in the package")
    return manifests[0].parents[1]

ROOT = Path.cwd()
if not (ROOT / "data/filing_manifest.json").exists():
    packages = sorted(set(ROOT.glob("Assignment2*Submission*.zip")))
    if packages:
        print("Opening the uploaded submission package...")
        ROOT = unpack_project(packages[-1], ROOT / "assignment2_project")
    else:
        print("Downloading the public SEC data and real saved LLM checkpoints...")
        url = GITHUB_URL + "/archive/" + DATA_COMMIT + ".zip"
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        ROOT = unpack_project(io.BytesIO(payload), ROOT / "assignment2_project")

if REFRESH_SEC and "@" not in SEC_CONTACT_EMAIL:
    raise ValueError("Enter your contact email before refreshing SEC data.")
SEC_USER_AGENT = "CAP5619 Academic Research " + SEC_CONTACT_EMAIL
MODEL_PATH = ROOT / "models" / MODEL_FILE
print("Mode:", "NEW MODEL INFERENCE" if RERUN_LLM else "REPLAY SAVED REAL LLM RESPONSES")
print("Tickers:", ", ".join(TICKERS))
'''
nb.cells.append(nbf.v4.new_markdown_cell("""## 6. Generate the 1-2 page report
The report reads the actual run summary and validation results. It describes data
selection, preprocessing, LLM settings, NER, evidence review, limitations and the
GitHub URL. The following function is also available as `tools/build_pdf_report.py`.
"""))
report_code = (root / "tools/build_pdf_report.py").read_text()
report_code = report_code[:report_code.index('\n\nif __name__ == "__main__":')]
nb.cells.append(nbf.v4.new_code_cell(report_code + '''

report_path = build_pdf_report(ROOT)
from pypdf import PdfReader
assert len(PdfReader(report_path).pages) == 2, "Report must be two pages."
assert GITHUB_URL in " ".join(page.extract_text() for page in PdfReader(report_path).pages)
print("Report created: two pages, with actual counts and the GitHub URL.")
'''))
nb.cells.append(nbf.v4.new_markdown_cell("""## 7. Download the complete submission
The ZIP includes the notebook, Python source, CSV, PDF report and saved source and
LLM audit records. If the prepared CV is present in this project folder, it is
included in this local ZIP as well; it is never uploaded to GitHub by this notebook.
Code and outputs are already published at the repository linked above. Canvas
submission itself is performed by the student.
"""))
nb.cells.append(nbf.v4.new_code_cell('''from IPython.display import display, FileLink

notebook_path = ROOT / "Assignment2_Complete.ipynb"
if not notebook_path.exists():
    url = "https://raw.githubusercontent.com/hingsgn/sec-8k-llm-product-analysis/main/Assignment2_Complete.ipynb"
    with urllib.request.urlopen(url, timeout=60) as response:
        notebook_path.write_bytes(response.read())

instructions = """ASSIGNMENT 2 - SUBMISSION CONTENTS
1. Assignment2_Complete.ipynb: full code, replay and fresh local-LLM inference.
2. sec_product_extractor.py: standalone extraction script.
3. output/products.csv: 102 reviewed product entries, exact five-column schema.
4. Assignment2_Methodology_Report.pdf: two-page methodology report.
5. data/ and output/*.json: source documents, actual LLM responses and audit records.
GitHub: https://github.com/hingsgn/sec-8k-llm-product-analysis
The repository URL must appear at the top of your CV. The separately prepared
Hieu_Nguyen_CV.pdf already contains it. Keep personal CV files off the public repo.
Run all notebook cells in order. Default mode replays saved real model responses;
RERUN_LLM=True makes fresh Python calls to the free local model.
If the CSV has fewer than 100 rows after new inference, expand/review the data;
do not pad the file with invented products or blank entries.
"""
# Use current counts even after a newly reviewed run.
instructions = instructions.replace("102 reviewed", f"{len(products)} reviewed")
(ROOT / "START_HERE.txt").write_text(instructions, encoding="utf-8")

package_path = ROOT / "Assignment2_Submission.zip"
include = [notebook_path, ROOT / "sec_product_extractor.py", report_path,
           ROOT / "START_HERE.txt", ROOT / "repository_url.txt"]
include += [ROOT / name for name in ["requirements-replay.txt", "requirements-llm.txt",
                                    "download_model.py", "test_extractor.py"]]
for folder in ["data/documents", "data/llm_responses", "output"]:
    include += sorted(path for path in (ROOT / folder).rglob("*") if path.is_file())
include += [ROOT / "data/filing_manifest.json", ROOT / "data/review_decisions.json"]
cv = ROOT / "Hieu_Nguyen_CV.pdf"
if cv.exists():
    include.append(cv)
with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as package:
    for path in dict.fromkeys(include):
        if path.exists():
            package.write(path, path.relative_to(ROOT))

print(f"READY: {len(documents)} filings, {len(products)} product rows, two-page report.")
print("GitHub:", GITHUB_URL)
display(FileLink(str(ROOT / "output/products.csv")))
display(FileLink(str(report_path)))
display(FileLink(str(package_path)))
if "google.colab" in sys.modules:
    from google.colab import files
    files.download(str(package_path))
'''))
for c in nb.cells:
    if c.cell_type == "code":
        c.outputs = []
        c.execution_count = None
        compile(c.source, "<notebook-cell>", "exec")
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12"}
nbf.validate(nb)
nbf.write(nb, root / "Assignment2_Complete.ipynb")
print(root / "Assignment2_Complete.ipynb")
