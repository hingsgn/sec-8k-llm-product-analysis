# SEC 8-K Product Announcement Extraction

CAP-5619 Assignment 2 - Hieu Nguyen

GitHub: https://github.com/hingsgn/sec-8k-llm-product-analysis

[Open the notebook in Google Colab](https://colab.research.google.com/github/hingsgn/sec-8k-llm-product-analysis/blob/main/Assignment2_LLM_Document_Analysis.ipynb)

Python queries a local Qwen instruction-tuned LLM to recognize ORGANIZATION and
PRODUCT entities in SEC 8-K filings and their attached press releases. The output
contains the five required columns. Separate source evidence and inference
checkpoints make each retained row traceable.

## Recorded results

- 104 filings from 10 companies; 245 primary reports and attached documents downloaded.
- One selected source passage analyzed per filing; 104 saved LLM responses.
- 102 reviewed product entries from 31 filings across five issuers.
- Exactly `company_name,stock_name,filing_time,new_product,product_description`.
- Every description is shorter than 180 characters; the maximum is 129.
- Checkpoint replay reproduced the CSV byte for byte.
- All 11 notebook code cells and five offline regression checks passed.

## Files

- `Assignment2_LLM_Document_Analysis.ipynb`: self-contained notebook with saved outputs.
- `sec_product_extractor.py`: documented Python implementation and command line.
- `output/products.csv`: required comma-separated output.
- `output/products_table.txt`: the same results separated by vertical bars.
- `Assignment2_Methodology_Report.docx` and `.pdf`: two-page methodology report.
- `data/filing_manifest.json`: issuer metadata, acceptance timestamps, and SEC URLs.
- `data/documents/`: cleaned primary filings and attached EX-99 documents.
- `data/llm_responses/`: actual model responses and their input passages.
- `data/review_decisions.json`: review exclusions and source-based name and description edits.
- `output/provenance.json` and `ner_entities.json`: evidence and labeled entity spans.
- `output/inference_log.json`, `run_summary.json`, and `validation.json`: run records.

## Reproduce the submitted output

Use Python 3.12. Saved checkpoints reproduce the output without downloading
the model or making new inference calls:

```bash
python -m pip install -r requirements-replay.txt
python sec_product_extractor.py extract
python sec_product_extractor.py validate
python -m unittest test_extractor.py -v
```

In Colab, open the link above and choose **Run all**. The notebook uses data in
the current project directory, detects an uploaded `Assignment2_Submission.zip`,
or downloads the public GitHub archive. No file-selection popup is required.

## Run fresh LLM inference

Install the full runtime, download the original publisher's model, and move
`data/llm_responses/` to a backup directory before extraction:

```bash
python -m pip install -r requirements-llm.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --prefer-binary
python download_model.py
python sec_product_extractor.py extract --model-path models/qwen2.5-3b-instruct-q4_k_m.gguf
python sec_product_extractor.py validate
```

The notebook also supports `RERUN_LLM=True`, which preserves old checkpoints.
New responses may differ and should be reviewed against the evidence before submission.

## Collect SEC filings

```bash
python sec_product_extractor.py collect --user-agent 'YourName AcademicResearch your-email@example.com'
```

The collector resolves tickers using `company_tickers.json`, retrieves histories
from the official Submissions JSON API, and follows filing-detail pages to 8-K
and EX-99 files. It follows relevant historical continuation files, caches raw
responses, retries temporary errors, and shares a two-requests-per-second limit.
The JSON API replaces the older Atom lookup for the same public filing source.

The sample uses ten Item 2.02 earnings 8-Ks for each of NVDA, AMD, INTC, CSCO,
CRM, ORCL, IBM, HPE, HPQ, and ADBE within 2023-2025, plus two older filings each
for NVDA and AMD. This targeted sample is not a complete survey of the S&P 500.

## Model and NER

Model: `Qwen/Qwen2.5-3B-Instruct-GGUF`, revision
`7dabda4d13d513e3e842b20f0d435c732f172cbe`, file
`qwen2.5-3b-instruct-q4_k_m.gguf`, through `llama-cpp-python==0.3.36`.
Settings: temperature 0, seed 42, 8,192 context tokens, and up to 900 generated
tokens. The recorded run consumed 56,496 total tokens. No paid API was used.

The LLM returns schema-constrained JSON containing organization and product
entities. Names must map to source spans and an announcement or availability
sentence. SEC metadata supplies company names, tickers, and filing acceptance
times with Eastern UTC offsets. Descriptions are short source-based extracts
or documented review edits. Repeated company/product pairs are deduplicated.
Filings without accepted announcements remain in audit records without blank rows.

## Limitations and attribution

Source matching establishes traceability rather than independently measured
precision or recall. The selected exhibit, announcement-sentence filter,
10,000-character passage limit, targeted sample, and model omissions can miss
announcements. Review decisions remove generic entities, partner products,
corporate events, and existing-product mentions. Original model responses
remain unchanged. AI assistance supported implementation, documentation, and review.

Model weights are not redistributed. Obtain them from the original publisher
and follow the upstream license. SEC documents and attached press releases
remain attributable to their original sources.

The real repository URL appears in the report, notebook, and at the top of the
separately delivered CV. Personal CV files are excluded from this public repository.

## Primary references

- https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- https://www.sec.gov/about/developer-resources
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
- https://github.com/abetlen/llama-cpp-python
