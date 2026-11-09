# Audit the exact tokenized records that will enter Trainer; never truncate or drop reviews.
assert audit_prepared_training(prepared_training_records)["prepared_records"] == 900
def render_prompt(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
tokenized_records = []
for record in prepared_training_records:
    messages = record["messages"]
    prompt_text = render_prompt(messages[:-1])
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    assert full_ids[:len(prompt_ids)] == prompt_ids, "Chat-template boundary mismatch"
    assert len(full_ids) <= MAX_SEQ_LENGTH, (record["filtered_index"], len(full_ids), "No silent truncation")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    supervised = tokenizer.decode([t for t in labels if t != -100], skip_special_tokens=True).strip()
    assert supervised == messages[-1]["content"], (record["filtered_index"], supervised)
    tokenized_records.append({"filtered_index":record["filtered_index"], "original_row":record["original_row"],
        "input_ids":full_ids, "attention_mask":[1]*len(full_ids), "labels":labels})
assert len(tokenized_records) == 900
assert [r["filtered_index"] for r in tokenized_records] == [r["filtered_index"] for r in train_rows]
assert Counter(tokenizer.decode([t for t in r["labels"] if t != -100], skip_special_tokens=True).strip()
               for r in tokenized_records) == {label:300 for label in LABELS}
print("PASS: all 900 actual tokenized training records verified; 300 per class; zero truncation.")
print("Sequence lengths:", min(len(r["input_ids"]) for r in tokenized_records),
      "to", max(len(r["input_ids"]) for r in tokenized_records))
for i in FEW_SHOT_IDS:
    k = next(k for k,r in enumerate(tokenized_records) if r["filtered_index"] == i)
    print("\nFORMATTED TRAINING EXAMPLE", i)
    print(tokenizer.decode(tokenized_records[k]["input_ids"], skip_special_tokens=False))
    print("SUPERVISED LABEL:", train_by_id[i]["reference_label"])
# Keep provenance IDs in a separate manifest; model input columns are allowlisted.
training_manifest = [{"filtered_index":r["filtered_index"], "original_row":r["original_row"],
 "input_ids_sha256":hashlib.sha256(json.dumps(r["input_ids"]).encode()).hexdigest(),
 "labels_sha256":hashlib.sha256(json.dumps(r["labels"]).encode()).hexdigest()} for r in tokenized_records]
(WORK / "training_manifest.json").write_text(json.dumps(training_manifest, indent=2))
train_dataset = Dataset.from_list([{k:r[k] for k in ("input_ids","attention_mask","labels")} for r in tokenized_records])

class ResponseOnlyCollator:
    def __call__(self, features):
        length = max(len(f["input_ids"]) for f in features)
        return {
            "input_ids":torch.tensor([f["input_ids"]+[tokenizer.pad_token_id]*(length-len(f["input_ids"])) for f in features]),
            "attention_mask":torch.tensor([f["attention_mask"]+[0]*(length-len(f["input_ids"])) for f in features]),
            "labels":torch.tensor([f["labels"]+[-100]*(length-len(f["input_ids"])) for f in features])
        }
collator = ResponseOnlyCollator()
for i, actual in enumerate(train_dataset):
    assert all(actual[k] == tokenized_records[i][k] for k in ("input_ids","attention_mask","labels"))
print("PASS: actual Trainer dataset equals audited inputs and response-only targets.")
