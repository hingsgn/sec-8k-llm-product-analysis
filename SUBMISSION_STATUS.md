# Assignment 2 submission status

The project contains the required Python implementation, executed notebook,
five-column CSV, and a two-page methodology report in Word and PDF formats.

- 104 SEC Form 8-K filings from 10 companies were analyzed in the saved local LLM run.
- 102 reviewed product entries come from 31 filings across five issuers.
- Model: Qwen2.5-3B-Instruct Q4_K_M through llama-cpp-python 0.3.36.
- All product descriptions are shorter than 180 characters; the maximum is 129.
- Every row retains source evidence, an SEC URL, and its LLM response checkpoint.
- Checkpoint replay reproduced the CSV byte for byte.
- The complete notebook's 13 code cells passed sequential Python execution; five regression checks passed.
- The final report was regenerated and visually verified as two pages.
- The default mode replays saved real inference records; fresh inference remains optional.
- Three source exhibits matched current SEC downloads.

Repository: https://github.com/hingsgn/sec-8k-llm-product-analysis

The repository URL appears in the report, notebook, and at the top of the
separately delivered CV. The public repository excludes personal CV files.
The submission package is prepared; it has not been submitted through Canvas.
