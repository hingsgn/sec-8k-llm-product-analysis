# SEC 8-K Product Announcement Extraction

CAP-5619 Assignment 2 - Hieu Nguyen

https://github.com/hingsgn/sec-8k-llm-product-analysis

[Open source notebook in Colab](https://colab.research.google.com/github/hingsgn/sec-8k-llm-product-analysis/blob/main/Assignment2_LLM_Document_Analysis.ipynb)

Python code for extracting product announcements from SEC Form 8-K filings and
attached EX-99 releases using a local Qwen2.5-3B-Instruct model and named entity
recognition. Outputs use exactly:

```text
company_name,stock_name,filing_time,new_product,product_description
```

This public repository contains **source code**. The separately submitted ZIP
contains the recorded SEC sample, model responses, review decisions, executed
notebook, CSV, and two-page methodology report. The CV is submitted separately.

## Reproduce the submitted results

Use Python 3.12. Extract `Assignment2_Submission.zip`, keep its folders together,
and run:

```bash
python -m pip install -r requirements.txt
python sec_product_extractor.py extract
python sec_product_extractor.py validate
python -m unittest -v test_extractor.py
```

This reuses saved inference checkpoints without installing the inference runtime
or downloading model weights. On Colab, upload the ZIP to the Files panel; the
notebook detects and extracts it automatically. The notebook has five main steps
and contains the implementation rather than importing an unavailable script.

## Run new extraction from this repository

```bash
python -m pip install -r requirements-llm.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --prefer-binary
python download_model.py
python sec_product_extractor.py collect --user-agent 'Your Name AcademicResearch your-contact-email@example.edu'
python sec_product_extractor.py extract
python sec_product_extractor.py validate
```

Replace the identifying User-Agent example with your actual contact information.
The collector uses the SEC ticker mapping and Submissions JSON API, then parses
filing detail pages with BeautifulSoup to locate 8-K and EX-99 documents. Requests
share a two-per-second limit, caching and retry backoff. The default sample is
104 Item 2.02 filings from ten technology issuers within 2023-2025.

Without the submission ZIP, the notebook also collects the sample and runs the
model. It downloads approximately 2.1 GB of model weights. New output must be
reviewed: source-string matching alone cannot establish semantic accuracy.
The optional `data/review_decisions.json` records exclusions, source-based name
corrections and short description edits; raw model responses remain unchanged.

## Model and validation

The local model is `Qwen/Qwen2.5-3B-Instruct-GGUF`, publisher revision
`7dabda4d13d513e3e842b20f0d435c732f172cbe`, file
`qwen2.5-3b-instruct-q4_k_m.gguf`. It runs through `llama-cpp-python==0.3.36`
with temperature 0, seed 42 and an 8,192-token context.

The LLM returns ORGANIZATION and PRODUCT named entities in schema-constrained
JSON. SEC metadata supplies the company name, ticker and acceptance timestamp.
Accepted product names and evidence must match actual source spans. Descriptions
are source-based and strictly shorter than 180 characters, including spaces.
Repeated company/product pairs are deduplicated. Negative filings remain in
audit files rather than becoming blank product rows.

Validation checks the exact five-column header, at least 100 real product rows,
complete fields, timestamp offsets, duplicate pairs, source grounding and NER
span offsets. It does not measure independent recall or guarantee 100% accuracy.
The selected issuer sample, exhibit selection, sentence filter, passage cap and
small-model omissions limit coverage. Expand the sample if review leaves fewer
than 100 valid product rows; do not pad it with artificial rows.

The code saves `output/products.csv`, `output/products_table.txt`, provenance,
NER entities, inference logs, run statistics and validation results.
Model weights are excluded; obtain them from the publisher under the upstream
Qwen research license. Public SEC documents remain attributable to their sources.

## Primary references

- https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- https://www.sec.gov/about/developer-resources
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
- https://github.com/abetlen/llama-cpp-python
