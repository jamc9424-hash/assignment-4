# Final predictions must come from the adapter extracted from the submitted ZIP.
assert INFERENCE_INSTRUCTION == SYSTEM_PROMPT
assert fingerprint(test_rows) == SUPPLIED_CONTENT_SHA256["test"]
assert len(test_rows) == len({r["filtered_index"] for r in test_rows}) == 300
assert ADAPTER_HASHES == {n:hashlib.sha256((EXTRACTED/n).read_bytes()).hexdigest() for n in ADAPTER_HASHES}
prediction_fields = ("original_row","filtered_index","title","text","reference_label","prediction","raw_response")
fine_tuned_predictions, actual_test_prompts = [], []
torch.cuda.reset_peak_memory_stats()
inference_started = time.perf_counter()
with (SUBMISSION / "fine_tuned_predictions.jsonl").open("w", encoding="utf-8") as out:
    for index, row in enumerate(test_rows):
        result = classify_review(row["title"], row["text"])
        assert result["messages"] == zero_shot_messages(row)
        record = {k:row[k] for k in prediction_fields[:5]}
        record.update(prediction=result["prediction"], raw_response=result["raw_response"])
        out.write(json.dumps(record, ensure_ascii=False)+"\n")
        out.flush()
        fine_tuned_predictions.append(record)
        actual_test_prompts.append({k:row[k] for k in ("original_row","filtered_index","reference_label")} |
                                  {"messages":result["messages"]})
        if (index+1)%50 == 0: print(f"Generated {index+1}/300 fine-tuned predictions", flush=True)
torch.cuda.synchronize()
inference_run = {"seconds":time.perf_counter()-inference_started,"gpu":INFERENCE_GPU,
                 "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                 "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
                 "adapter_hashes":ADAPTER_HASHES,"count":len(fine_tuned_predictions)}
examples = [next(p for p in actual_test_prompts if p["reference_label"]==label) for label in LABELS]
with (SUBMISSION/"prompt_examples.jsonl").open("w",encoding="utf-8") as out:
    for p in examples: out.write(json.dumps(p,ensure_ascii=False)+"\n")
(WORK/"inference_run.json").write_text(json.dumps(inference_run,indent=2))
print(json.dumps(inference_run,indent=2))
print("PASS: 300 predictions and three actual prompts generated using the submitted adapter.")
