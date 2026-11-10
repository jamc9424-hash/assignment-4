# Assignment 4: Amazon review classification

Completed comparison of zero-shot and few-shot prompting of the same hosted Qwen model with an Unsloth QLoRA adapter trained on **Qwen/Qwen2.5-3B-Instruct**. Full training and saved-adapter inference were directly tested on a **Tesla T4**.

## Submission files

Download all four files from the [final submission release](https://github.com/jamc9424-hash/assignment-4/releases/tag/assignment4-final):

- `assignment4_comparison.ipynb` — executed notebook, data checks, complete prompts, metrics, confusion matrices, exact McNemar comparisons, final report, and standalone saved-adapter inference example.
- `adapter.zip` — exactly `adapter_model.safetensors` and `adapter_config.json`, with the original base model and pinned revision identified.
- `fine_tuned_predictions.jsonl` — all 300 prescribed test reviews with the required seven fields, including unmodified raw responses.
- `prompt_examples.jsonl` — three actual fine-tuned test prompts, one per reference class.

The adapter is a release asset because it exceeds GitHub's ordinary file-size limit. The other submission files are also available directly in this repository. The four files were independently validated after download; see [verification evidence](evidence/final/independent_submission_check.json).

## Results

| Approach | Balanced accuracy | Macro-F1 | Correct / 300 | Invalid |
|---|---:|---:|---:|---:|
| Zero-shot | 75.67% | 0.7312 | 227 | 0 |
| Few-shot | 76.67% | 0.7492 | 230 | 0 |
| Fine-tuned | 80.00% | 0.7985 | 240 | 0 |

Fine-tuning achieved the highest macro-F1 on this fixed test set. Neutral recall improved from 44% with few-shot prompting to 70%, while negative recall declined from 95% to 77%. None of the exact paired McNemar comparisons established a difference in correctness at the 0.05 level: zero/few p=0.548828, zero/fine-tuned p=0.123731, few/fine-tuned p=0.245061. These are exploratory comparisons, not proof of general superiority.

Training completed two epochs and 114 optimizer steps in 8.68 minutes on a T4. Peak PyTorch-allocated training memory was 2.39 GiB. A fresh base model and tokenizer were loaded, then the adapter extracted from the submitted ZIP was attached. That model generated all 300 final predictions in 94.46 seconds on the T4, with peak allocated memory of 2.82 GiB. Adapter and prediction hashes link the submitted files to the reported results.

## Fixed protocol and reproducibility

The supplied files were checked against the original Amazon Reviews '23 Gift Cards source. The prescribed 900/300/300 train/validation/test sets have exact IDs, original text, class balances, unique IDs, and no overlap. The actual tokenized training inputs were also checked: all 900 reviews, 300 per class, with no truncation. Validation remained unused.

The model, settings, instruction, and demonstrations were frozen before test evaluation. The approved few-shot training IDs are 8849, 2359, and 4447. Inputs contain only title and text. All decoding uses greedy/temperature-zero behavior, eight output tokens at most, and thinking disabled. Invalid responses would be retained and counted as incorrect.

The hosted baselines were generated from a local Windows client during a Colab outage; their original requests and responses are in `evidence/hosted_local`. The final notebook imports these files from an immutable commit and verifies byte hashes, every request, every raw response, and test IDs before reuse. The separate `hosted_results.ipynb` preserves the earlier executed two-method audit. No test results were used to adjust training or prompts.

To run inference without training or calling the class endpoint, follow the saved-adapter section of the final notebook: install the listed packages, place `adapter.zip` at `/content/assignment4/submission/adapter.zip`, then run the standalone inference cell. It loads the original model/tokenizer revision specified by the adapter and classifies a sample review.

Credentials are supplied only at runtime and are not included in the notebook or repository. Exact package versions, training manifests, timings, file hashes, and the final report are in `evidence/final`. Repository attributes preserve artifact bytes across Windows/Linux transfers.

[Working Colab notebook](https://colab.research.google.com/drive/1Nn1Lo8GYPx7wxsoMF60YFIb_my2yRfBl) · [Final report](evidence/final/report.md)

## References

- [Unsloth fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [Unsloth memory requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements)
- [Original Qwen model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Colab MCP](https://github.com/davedgd/colab-mcp)
