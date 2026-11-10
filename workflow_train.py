# Train only after every provenance and tokenization assertion above passes.
assert len(trainer_dataset_ids := [r["filtered_index"] for r in tokenized_records]) == 900
assert trainer_dataset_ids == [r["filtered_index"] for r in train_rows]
training_args = TrainingArguments(
    output_dir=str(WORK / "checkpoints"), per_device_train_batch_size=2,
    gradient_accumulation_steps=8, learning_rate=2e-4, num_train_epochs=2,
    warmup_steps=6, weight_decay=0.01, lr_scheduler_type="linear",
    fp16=True, bf16=False, optim="adamw_8bit", logging_steps=10,
    save_strategy="epoch", save_total_limit=2, eval_strategy="no", seed=43, data_seed=43,
    report_to="none", remove_unused_columns=False, dataloader_num_workers=0)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                  data_collator=collator)
assert len(trainer.train_dataset) == 900
for k, r in enumerate(trainer.train_dataset):
    assert r["input_ids"] == tokenized_records[k]["input_ids"]
    assert r["labels"] == tokenized_records[k]["labels"]
print("PASS: actual Trainer dataset contains the exact 900 audited reviews.")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
started = time.perf_counter()
train_result = trainer.train()
assert trainer.state.global_step == 114, trainer.state.global_step
assert abs(trainer.state.epoch - 2.0) < 1e-6, trainer.state.epoch
torch.cuda.synchronize()
training_seconds = time.perf_counter() - started
training_run = {
 "base_model":BASE_MODEL, "base_revision":BASE_REVISION, "tokenizer":TOKENIZER_ID,
 "gpu":gpu_name, "gpu_total_gib":torch.cuda.get_device_properties(0).total_memory/2**30,
 "training_seconds":training_seconds,
 "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
 "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
 "train_rows":900, "epochs":2, "completed_optimizer_steps":trainer.state.global_step, "seed":43, "learning_rate":2e-4,
 "warmup_steps":6, "weight_decay":0.01, "optimizer":"adamw_8bit", "gradient_checkpointing":"unsloth",
 "batch_size":2, "gradient_accumulation_steps":8, "max_seq_length":MAX_SEQ_LENGTH,
 "lora_rank":16, "lora_alpha":32, "lora_dropout":0,
 "packages":package_versions, "metrics":train_result.metrics,
 "train_fingerprint":fingerprint(train_rows),
 "system_prompt":SYSTEM_PROMPT, "few_shot_ids":FEW_SHOT_IDS,
 "generation":{"do_sample":False, "max_new_tokens":8, "thinking":False}}
print(json.dumps(training_run, indent=2))
(WORK / "training_run.json").write_text(json.dumps(training_run, indent=2))
# The submission ZIP contains only the adapter's two required files.
import zipfile
ADAPTER_DIR = WORK / "trained_adapter"
model.peft_config["default"].base_model_name_or_path = BASE_MODEL
model.peft_config["default"].revision = BASE_REVISION
model.save_pretrained(ADAPTER_DIR, safe_serialization=True)
adapter_config = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
assert adapter_config["base_model_name_or_path"] == BASE_MODEL
assert adapter_config["revision"] == BASE_REVISION
SUBMISSION = WORK / "submission"
SUBMISSION.mkdir(exist_ok=True)
ADAPTER_ZIP = SUBMISSION / "adapter.zip"
with zipfile.ZipFile(ADAPTER_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        assert (ADAPTER_DIR / name).is_file()
        z.write(ADAPTER_DIR / name, arcname=name)
with zipfile.ZipFile(ADAPTER_ZIP) as z:
    assert set(z.namelist()) == {"adapter_model.safetensors", "adapter_config.json"}
print("Saved and verified adapter ZIP:", ADAPTER_ZIP.name)

# Back up the completed adapter immediately; runtime disks are temporary.
from google.colab import files
files.download(str(ADAPTER_ZIP))
