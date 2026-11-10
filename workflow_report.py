# Final report: generated only from verified saved files and completed hardware measurements.
from IPython.display import Markdown, display
assert submission_checks["passed"]
assert training_run["completed_optimizer_steps"] == 114
assert inference_run["count"] == 300
assert "T4" in training_run["gpu"] and "T4" in inference_run["gpu"]
saved_evaluation = json.loads((WORK/"evaluation.json").read_text())
assert saved_evaluation["files"]["Fine-tuned"]["sha256"] == hashlib.sha256((SUBMISSION/"fine_tuned_predictions.jsonl").read_bytes()).hexdigest()
report = "## Final report — three approaches to review classification\n\n"
report += ("We compared zero-shot and three-example few-shot prompting of the same hosted "
           "Qwen3.6-35B-A3B AWQ model with Qwen2.5-3B-Instruct fine-tuned on the prescribed 900 reviews. "
           "The hosted model has about 35B total parameters and about 3B active per token; the smaller model has about 3.09B base parameters. "
           "We chose the 3B model to meet the 2–5B guidance and leave room for training on a 16 GB T4. "
           "Four-bit QLoRA trains small adapter matrices rather than all base weights.\n\n")
report += ("All approaches used the same task instruction, verbatim title/text formatting, and the same 300 test IDs. "
           "The approved demonstrations (#8849, #2359, #4447) illustrate satisfaction, a non-evaluative question, and dissatisfaction; all belong to training. "
           "The validation set was reserved and unused. No test result changed the prompts, examples, model, or learning settings. "
           "Greedy/temperature-zero decoding used an eight-token cap; thinking was disabled.\n\n")
report += "| Approach | Balanced accuracy | Macro-F1 | Correct | Incorrect | Invalid |\n|---|---:|---:|---:|---:|---:|\n"
for item in saved_evaluation["metrics"]:
    report += f'| {item["method"]} | {item["balanced_accuracy"]:.2%} | {item["macro_f1"]:.4f} | {item["correct"]} | {item["incorrect"]} | {item["invalid"]} |\n'
report += f'\n**{saved_evaluation["best_macro_f1"]} had the highest macro-F1 on this test set.** '
report += "Balanced accuracy is also ordinary accuracy here because there are exactly 100 references per class. "
neutral_scores = [f'{p["method"]}: {p["recall"]:.0%}' for p in saved_evaluation["per_class"] if p["class"]=="NEUTRAL"]
report += "Neutral-class recall was " + "; ".join(neutral_scores) + ". "
per_class_map={(p["method"],p["class"]):p for p in saved_evaluation["per_class"]}
report += (f'Compared with few-shot prompting, fine-tuning raised neutral recall from {per_class_map["Few-shot","NEUTRAL"]["recall"]:.0%} '
           f'to {per_class_map["Fine-tuned","NEUTRAL"]["recall"]:.0%}, while negative recall changed from '
           f'{per_class_map["Few-shot","NEGATIVE"]["recall"]:.0%} to {per_class_map["Fine-tuned","NEGATIVE"]["recall"]:.0%}. ')
for review_id in (30003,30008,30017):
    row=aligned["Fine-tuned"][review_id]
    if row["prediction"] != row["reference_label"]:
        report += f'For example, review #{review_id} (title: {row["title"]!r}) was predicted {row["prediction"]} rather than its {row["reference_label"]} reference. '
report += ("The confusion matrices and concrete errors above show the remaining mistakes. "
           "Star-derived reference labels are imperfect measures of textual sentiment, particularly for mixed reviews. "
           "Titles such as Five Stars and HTML entities were intentionally retained.\n\n")
report += "**Paired statistical comparisons.** "
for pair in saved_evaluation["mcnemar"]:
    interpretation = "evidence of asymmetric correctness at the unadjusted 0.05 level" if pair["exact_p_value"] < .05 else "no established difference in correctness at the 0.05 level"
    report += f'{pair["A"]} vs {pair["B"]}: exact McNemar p={pair["exact_p_value"]:.6g} ({interpretation}); only the first was correct on {pair["A_only_correct"]} reviews and only the second on {pair["B_only_correct"]}. '
report += ("These three unadjusted tests are exploratory, test correctness rather than macro-F1, and do not establish general superiority or equivalence. "
           "Different base models and sizes also prevent attributing the whole difference solely to fine-tuning.\n\n")
report += (f'**Compute and reproducibility.** Training completed two epochs / 114 optimizer steps directly on {training_run["gpu"]}, '
           f'taking {training_run["training_seconds"]/60:.2f} minutes, with peak allocated memory {training_run["peak_allocated_gib"]:.2f} GiB '
           f'and peak reserved memory {training_run["peak_reserved_gib"]:.2f} GiB. '
           f'The fresh saved-adapter model generated all 300 predictions on {inference_run["gpu"]} in {inference_run["seconds"]/60:.2f} minutes; '
           f'peak allocated memory during that run was {inference_run["peak_allocated_gib"]:.2f} GiB. '
           'Thus full training and submitted-adapter inference were directly tested on a T4. '
           'Training used learning rate 0.0002, rank 16/alpha 32, fp16, batch 2 with eight-step accumulation, '
           'six warm-up steps, seed 43, response-only loss, and no truncation. ')
report += ("The hosted requests were completed from a local Windows client during the Colab outage, with two requests in flight. "
           "They were imported here only after verifying immutable file hashes, every request, every raw response, and the prescribed IDs. "
           "Hosted and local timings include different hardware, concurrency, networking, and compilation costs, so they are not a controlled speed benchmark. "
           "Prompting avoids training setup; the adapter requires upfront training but permits inference on the smaller local model.\n\n")
report += ("**An agent decision investigated.** Unsloth initially substituted a prequantized repository and ignored the requested original-base revision. "
           "We caught this before training, required the exact original model name, and pinned its revision. "
           "The earlier interrupted runtime's adapter was not assumed valid or complete; this submission uses a fresh, fully audited training run. "
           "The final ZIP was extracted separately, its original base and matching tokenizer were loaded afresh, and its adapter was attached before generating the submitted predictions and three actual prompt examples. "
           "The final checks match the ZIP's adapter hashes to inference and match the prediction-file hash to these reported metrics.\n")
(WORK/"report.md").write_text(report,encoding="utf-8")
display(Markdown(report))
