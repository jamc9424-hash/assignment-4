# Final submission checks: fail rather than submit inconsistent artifacts.
submitted_predictions=load_jsonl(SUBMISSION/"fine_tuned_predictions.jsonl")
submitted_examples=load_jsonl(SUBMISSION/"prompt_examples.jsonl")
assert len(submitted_predictions)==300
assert {r["filtered_index"] for r in submitted_predictions}=={r["filtered_index"] for r in test_rows}
assert len({r["original_row"] for r in submitted_predictions})==300
assert all(set(r)==set(prediction_fields) for r in submitted_predictions)
assert len(submitted_examples)==3
assert Counter(r["reference_label"] for r in submitted_examples)=={label:1 for label in LABELS}
predictions_by_id={r["filtered_index"]:r for r in submitted_predictions}
for example in submitted_examples:
    assert set(example)=={"original_row","filtered_index","reference_label","messages"}
    pred=predictions_by_id[example["filtered_index"]]
    assert example["original_row"]==pred["original_row"] and example["reference_label"]==pred["reference_label"]
    assert example["messages"]==inference_messages(pred["title"],pred["text"])
    assert example in actual_test_prompts
with zipfile.ZipFile(SUBMISSION/"adapter.zip") as z:
    assert sorted(z.namelist())==["adapter_config.json","adapter_model.safetensors"]
    assert {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}==inference_run["adapter_hashes"]
    submitted_config=json.loads(z.read("adapter_config.json"))
    assert submitted_config["base_model_name_or_path"]==INFERENCE_BASE
    assert submitted_config["revision"]==INFERENCE_REVISION
assert evaluation["files"]["Fine-tuned"]["sha256"]==hashlib.sha256((SUBMISSION/"fine_tuned_predictions.jsonl").read_bytes()).hexdigest()
submission_checks={"passed":True,"predictions":300,"prompt_examples":3,
                  "zip_entries":["adapter_config.json","adapter_model.safetensors"],
                  "submitted_adapter_matches_inference":True,"reported_prediction_hash_matches":True}
(WORK/"submission_checks.json").write_text(json.dumps(submission_checks,indent=2))
print(json.dumps(submission_checks,indent=2))
