# Reuse the completed local hosted runs only after checking their exact inputs and raw responses.
# Upload the evidence/hosted_local folder into /content/assignment4/hosted_local first.
from pathlib import Path
import json, hashlib, shutil
LOCAL_HOSTED = WORK / "hosted_local"
def validate_and_import_hosted(folder):
    manifest=json.loads((folder/"request_manifest.json").read_text())
    assert manifest["model"] == "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
    assert manifest["base_url"] == "http://dobolyi.com:9001/v1"
    assert manifest["settings"] == {"temperature":0,"max_tokens":8,"chat_template_kwargs":{"enable_thinking":False}}
    assert manifest["system_prompt"] == SYSTEM_PROMPT
    assert manifest["few_shot_filtered_indices"] == FEW_SHOT_IDS
    assert manifest["test_fingerprint"] == fingerprint(test_rows)
    assert manifest["train_fingerprint"] == fingerprint(train_rows)
    assert len(manifest["requests"]) == 600
    request_map={(r["method"],r["filtered_index"]):r for r in manifest["requests"]}
    assert len(request_map) == 600
    for method,make_messages in (("zero_shot",zero_shot_messages),("few_shot",few_shot_messages)):
        rows=load_jsonl(folder/(method+"_predictions.jsonl"))
        responses=load_jsonl(folder/(method+"_responses.jsonl"))
        assert len(rows)==len(responses)==300
        by_id={r["filtered_index"]:r for r in rows}
        res_by_id={r["prediction"]["filtered_index"]:r for r in responses}
        assert set(by_id)==set(res_by_id)=={r["filtered_index"] for r in test_rows}
        for target in test_rows:
            i=target["filtered_index"]
            payload=dict(model=manifest["model"],messages=make_messages(target),**manifest["settings"])
            digest=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            assert request_map[method,i]["payload"]==payload
            assert request_map[method,i]["request_sha256"]==res_by_id[i]["request_sha256"]==digest
            row=by_id[i]
            assert all(row[k]==target[k] for k in ("original_row","filtered_index","title","text","reference_label"))
            assert res_by_id[i]["prediction"]==row
            raw=res_by_id[i]["response"]["choices"][0]["message"].get("content")
            assert row["raw_response"]==(raw if isinstance(raw,str) else json.dumps(raw))
            assert row["prediction"]==normalize_response(row["raw_response"])
        shutil.copy2(folder/(method+"_predictions.jsonl"),WORK/(method+"_predictions.jsonl"))
        print(method, "PASS: imported 300 locally generated hosted predictions with exact request/response provenance.")
    shutil.copy2(folder/"hosted_runs.json",WORK/"hosted_runs.json")
    return json.loads((folder/"hosted_runs.json").read_text())

if (LOCAL_HOSTED/"request_manifest.json").exists():
    hosted_runs=validate_and_import_hosted(LOCAL_HOSTED)
else:
    # Hosted baselines: frozen prompts, same 300 targets; credentials stay outside saved source.
    import os, getpass, requests, time
    CLASS_BASE_URL = "http://dobolyi.com:9001/v1"
    HOSTED_MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
    HOSTED_SETTINGS = {"temperature":0,"max_tokens":8,"chat_template_kwargs":{"enable_thinking":False}}
    def class_api_key():
        key = os.environ.get("CLASS_API_KEY")
        if not key:
            try:
                from google.colab import userdata
                key = userdata.get("CLASS_API_KEY")
            except Exception:
                key = getpass.getpass("Class endpoint API key (never saved): ")
        return key
    def hosted_response(messages, key):
        payload = dict(model=HOSTED_MODEL, messages=messages, **HOSTED_SETTINGS)
        for attempt in range(4):
            try:
                response = requests.post(CLASS_BASE_URL+"/chat/completions",
                    headers={"Authorization":"Bearer "+key},json=payload,timeout=120)
                if response.status_code in (429,500,502,503,504) and attempt<3:
                    time.sleep(2**attempt); continue
                response.raise_for_status()
                body = response.json()
                raw = body["choices"][0]["message"].get("content")
                # Missing textual responses are recorded as invalid, never guessed.
                return raw if isinstance(raw,str) else json.dumps(raw), body.get("usage",{})
            except (requests.ConnectionError,requests.Timeout):
                if attempt==3: raise
                time.sleep(2**attempt)
        raise RuntimeError("Hosted endpoint retry limit reached")
    assert fingerprint(test_rows) == SUPPLIED_CONTENT_SHA256["test"]
    assert set(FEW_SHOT_IDS) <= {r["filtered_index"] for r in train_rows}
    hosted_runs = {}
    _api_key = class_api_key()
    try:
        for method, message_function in (("zero_shot",zero_shot_messages),("few_shot",few_shot_messages)):
            started_hosted = time.perf_counter()
            rows_out = []
            with (WORK/(method+"_predictions.jsonl")).open("w",encoding="utf-8") as out:
                for index,row in enumerate(test_rows):
                    messages = message_function(row)
                    assert json.loads(messages[-1]["content"]) == {"title":row["title"],"text":row["text"]}
                    raw,usage = hosted_response(messages,_api_key)
                    rec={k:row[k] for k in prediction_fields[:5]}
                    rec.update(prediction=normalize_response(raw),raw_response=raw)
                    rows_out.append(rec)
                    out.write(json.dumps(rec,ensure_ascii=False)+"\n"); out.flush()
                    if (index+1)%50==0: print(f"{method}: {index+1}/300",flush=True)
            hosted_runs[method]={"seconds":time.perf_counter()-started_hosted,"model":HOSTED_MODEL,
                                 "settings":HOSTED_SETTINGS,"count":len(rows_out)}
    finally:
        del _api_key
        os.environ.pop("CLASS_API_KEY",None)
    (WORK/"hosted_runs.json").write_text(json.dumps(hosted_runs,indent=2))
    print(json.dumps(hosted_runs,indent=2))
