# Assignment 4: Amazon review classification

Work in progress: compare zero-shot and few-shot prompting of the class-hosted Qwen model with an Unsloth QLoRA adapter trained on a small open-weight model.

[Live Colab notebook](https://colab.research.google.com/drive/1QIWpAMqI8fy3pAlM70s-s1heIhAgWYet)

## Protocol

`data_protocol.py` reconstructs the exact required splits from Amazon Reviews '23 Gift Cards and checks IDs, counts, labels, contents, and disjointness. The user's supplied split files are audited locally against that reconstruction. Raw data and personal metadata are not committed.

Training uses 900 reviews (300 per class), validation reserves 300 reviews and is unused, and all approaches use the same 300 test reviews. No test-driven prompt/model selection is allowed. Credentials must be supplied at runtime, never saved in notebook source or outputs.

## Status

The supplied train/validation/test files passed an exact audit against the authoritative source: 900/300/300 rows, required class balances, unique IDs, disjoint splits, matching contents, and the prescribed training shuffle. See `data_audit.json`. The same checks and actual prepared-training-record checks were executed successfully in the live Colab notebook.

`assignment4_comparison.ipynb` is a **source snapshot**, not the final executed submission. Runtime inspection currently reports CPU only; T4 selection is required before training. Recommended model: Qwen2.5-1.5B-Instruct with 4-bit QLoRA. Prompt and training choices are pending discussion. No training or test results are claimed. Final submission files will only be marked complete after execution, saved-adapter reload, prediction alignment checks, and output validation.

## References

- https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
- https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- https://github.com/davedgd/colab-mcp
