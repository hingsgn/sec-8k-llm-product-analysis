# Validation status

This is a **complete implementation**, with a small live functional check. It is
not an already executed 100-product submission dataset. Running the notebook
creates the assignment CSV and report using the current run's measured results.

## Automated checks

17 deterministic tests passed. They cover primary/exhibit parsing, actual
acceptance timestamps and daylight-saving offsets, late-document release
retrieval, evidence/NER validation, the strict description limit, required CSV
columns, no-product filings, deduplication, checkpoint reuse, preservation of
manual review labels, incompatible-model detection and two-page PDF generation.

All code cells parse. The notebook's implementation cells execute in sequence,
and the embedded standalone script matches `sec_8k_ollama.py` exactly.

## Live local-LLM check

- Ollama: 0.40.11, CPU inference.
- Model: `llama3.2:3b`, 3.2B parameters, Q4_K_M quantization.
- Model digest: `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`.
- Final prompt version: `cap5619-ollama-1.2`; temperature 0; seed 42;
  context 8192; output limit 3072 tokens.
- A clearly labeled synthetic launch passed the model/schema/evidence diagnostic.
  It was excluded from all SEC result counts.
- Two genuine SEC 8-K submissions were parsed and queried: AAPL accession
  `0000320193-26-000018` and NVDA accession `0001045810-26-000078`.
- The live check analyzed 3 readable primary/exhibit documents in 4 passages.
  It exported 1 grounded product row and rejected 2 unsupported candidates.
- CSV had the exact five required columns. The accepted description was 19
  characters. The generated PDF was two pages and was rendered and inspected.

This smoke test confirms the main execution path; it does not establish a
dataset-level accuracy score or satisfy the 100-row requirement. The full
Colab/T4 run and a human accuracy review remain to be performed by running the
distributed notebook. No previously generated dataset is replayed as a new run.

## Run the checks

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q tests/test_ollama_assignment.py
```

For the actual assignment, install the trained NER pipeline with
`python -m spacy download en_core_web_sm`, start Ollama, and execute the notebook
or script with your own SEC contact email. The live smoke checks used only
public SEC source data; no personal contact email is included in this package.
