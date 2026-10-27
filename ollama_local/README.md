# CAP-5619 Assignment 2 - Local Ollama Version

This implementation follows the Python-to-Ollama document workflow in the
instructor's `local_LLM.ipynb`. It uses **Llama 3.2 3B**, trained spaCy NER, and
evidence-backed extraction from SEC 8-K filings and their EX-99 exhibits.
There is no OpenAI API key or paid inference API.

## Chạy trên Google Colab

1. Upload `Assignment2_Ollama_Colab.ipynb` vào Google Colab.
2. Chọn **Runtime > Change runtime type > T4 GPU** nếu có. CPU vẫn chạy được
   nhưng chậm hơn. Chạy lần lượt từ trên xuống hoặc chọn **Run all**.
3. Ở ô cấu hình, nhập email liên hệ thật khi được hỏi. SEC cần nhận diện công cụ
   truy cập; đây **không phải API key**. Email không được ghi vào `run_config.json`.
4. Notebook cài Ollama **bên trong Colab**, khởi động server, đợi server sẵn sàng
   rồi tải model. Ollama trên laptop không phải localhost của Colab.
5. Giữ `TARGET_ROWS = 100` và `MIN_FILINGS = 100`. Nếu chưa đủ sản phẩm, tăng
   `MAX_FILINGS`, `FILINGS_PER_COMPANY` hoặc mở rộng ticker/ngày rồi chạy lại.
6. Tải `Assignment2_Submission.zip` ở ô cuối. Zip chứa script, CSV và báo cáo.

`SAVE_CHECKPOINTS_TO_DRIVE = True` là lựa chọn giữ checkpoint qua lần reset
Colab; cần tự cho phép kết nối Google Drive. Nếu để False, tải file trước khi
Colab kết thúc phiên. Chạy lại cùng thư mục để tiếp tục; đổi model/prompt/NER
thì dùng thư mục kết quả mới.

## Chạy trên máy cá nhân

Python 3.10 trở lên. Cài Ollama từ <https://ollama.com/download>, mở ứng dụng
Ollama hoặc chạy `ollama serve` trong một terminal. Trong terminal khác:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull llama3.2:3b
python sec_8k_ollama.py --email YOUR_REAL_EMAIL --output assignment2_output
```

Có thể chạy notebook bằng Jupyter/PyCharm trên máy cá nhân. Với máy RAM ít,
đổi model thành `llama3.2:1b` và dùng thư mục kết quả mới; model nhỏ cần kiểm
tra kết quả kỹ hơn. DeepSeek cũng có thể dùng qua Ollama nhưng không phải model
mặc định của bản này. Không chạy tất cả các lệnh tải model trong notebook thầy;
chỉ cần một model.

## Pipeline

1. Map tickers to CIKs using `company_tickers.json`.
2. Parse the instructor's `browse-edgar?...type=8-K&output=atom` feed with
   BeautifulSoup; fill short/unavailable feeds from the official SEC submissions API.
3. Download complete submissions; parse the primary 8-K and readable EX-99
   press-release attachments. Text PDFs use PyMuPDF. Hidden HTML is removed.
4. Select contextual release-language windows and split with overlap. Trained
   spaCy ORG/PRODUCT NER supplies hints. The LLM additionally labels literal
   ORGANIZATION and PRODUCT spans and decides whether each mention is a new release.
5. Query local Ollama `/api/chat` with a JSON schema, temperature 0 and seed 42.
6. Validate names, entities and verbatim evidence. Enforce **<180 characters**
   (179 maximum). Deduplicate company-product names across the observed corpus.
7. Save checkpoint and CSV after each completed filing. A failed/truncated
   extraction is recorded as an error, not a filing with no products.
8. Stop once both configured row and analyzed-filing targets are met, or an
   explicit source/filing limit is reached. No synthetic rows or “N/A products”
   are inserted to reach 100.

## Output files

| File | Purpose |
| --- | --- |
| `products.csv` | Exactly `company_name,stock_name,filing_time,new_product,product_description` |
| `products_table.txt` | Table using ` | ` separators |
| `methodology_report.pdf` | Two-page report with actual run counts and model settings |
| `methodology_report.md` | Editable report text |
| `run_stats.json` | Counts, maximum description length and data-size status |
| `product_audit.csv` | Accession, source URL and evidence for each product |
| `filing_audit.csv` | Every analyzed/error filing, including negative findings |
| `rejected_candidates.csv` | Unsupported model candidates and rejection reasons |
| `manual_review.csv` | First 20 rows for human labels; labels survive export/resume |
| `filing_manifest.json` | Discovered source filings |
| `environment.json` | Installed dependency versions |
| `run_signature.json` | Actual model digest, model/NER versions and settings |
| `filings/`, `cache/` | Resume checkpoints, raw SEC responses and model responses |

The script returns exit code 0 only after at least 100 actual product rows and
100 successfully analyzed filings; code 2 means incomplete size, and code 1
means an actionable error. A data-size check does not certify extraction accuracy.

## Submission

Submit the notebook or script, `products.csv`, and the PDF report. Inspect
`product_audit.csv` and complete a human review before submitting. The report
must link to the repository containing this code; put that same URL at the top
of your CV. No CV is included or modified by this project.

**100 filings are not necessarily 100 products.** Do not claim completion if
`assignment_size_requirement_met` in `run_stats.json` is false. A small smoke
test is a functional check, not the final 100-row assignment dataset.

## Troubleshooting

- **ConnectionRefusedError:** rerun the Ollama setup cell in the current runtime.
  On a computer, start the Ollama app / `ollama serve`. The setup waits for readiness
  and reports a log path if the process exits.
- **Model not found:** pull exactly the configured tag, e.g. `llama3.2:3b`.
- **SEC 403:** check contact identification, pause and retry later; do not use
  block-avoidance proxies. The code does not replace a block page with filing data.
- **SEC 429:** backoff is automatic; keep one run active and retain the cache.
- **Ollama model download fails:** rerun setup; Ollama resumes partial downloads.
- **Invalid/truncated JSON:** the response is logged and retried; a persistently
  failing filing is not marked complete. Reduce `chunk_chars` or increase
  `num_predict`, then use a new output directory when changing model settings.
- **Passage limit reached:** increase `max_chunks_per_filing` and rerun; previously
  successful filings stay complete.
- **Not enough genuine products:** expand coverage and rerun. Many 8-Ks only
  announce earnings, personnel changes or financing.

Limitations: contextual retrieval can miss unusual announcement wording;
exhibit types other than EX-99 are outside this selection; scanned PDFs need
OCR; repeated launches with exactly the same company-product name may merge.
Evidence validation is not a substitute for a labeled accuracy evaluation.

## Sources

- Instructor-supplied `local_LLM.ipynb`, Python/Ollama document-query examples.
- <https://www.sec.gov/search-filings/edgar-application-programming-interfaces>
- <https://www.sec.gov/about/developer-resources>
- <https://docs.ollama.com/linux>
- <https://docs.ollama.com/api/chat>
- <https://docs.ollama.com/capabilities/structured-outputs>
- <https://spacy.io/models/en>
