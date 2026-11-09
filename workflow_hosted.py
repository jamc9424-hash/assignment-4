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
