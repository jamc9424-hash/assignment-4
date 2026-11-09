"""Run frozen hosted baselines locally; resume verified responses after interruptions."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import requests
from data_protocol import LABELS, RANGES, canonical_records, select_splits, audit_splits, fingerprint, load_jsonl

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'evidence' / 'hosted_local'
OUT.mkdir(parents=True, exist_ok=True)
CONFIG = json.loads((ROOT/'experiment_config.json').read_text())
SOURCE = ROOT/'data/Gift_Cards.jsonl.gz'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == 'e03a258ebd7b1e2591b09862aab65d3dcde9c300744d800aa1fbaf374b7c2340'
splits = select_splits(canonical_records(SOURCE))
supplied_dir = Path(os.environ['ASSIGNMENT_SPLIT_DIR'])
supplied = {n:load_jsonl(supplied_dir/(n+'.jsonl')) for n in RANGES}
audit = audit_splits(supplied, splits)
assert [r['filtered_index'] for r in supplied['train']] == [r['filtered_index'] for r in splits['train']]
train_by_id = {r['filtered_index']:r for r in splits['train']}
demo_ids = CONFIG['few_shot_filtered_indices']
assert demo_ids == [8849,2359,4447]
demos = [train_by_id[i] for i in demo_ids]
assert [r['reference_label'] for r in demos] == list(LABELS)
test = splits['test']
assert len(test) == 300 and fingerprint(test) == '1ffb05d5497fd99b01c1b2c31fd027fa4b86fb8cbe5c18db6cb1f3e804f3611f'
settings = {'temperature':0,'max_tokens':8,'chat_template_kwargs':{'enable_thinking':False}}
def review_input(row):
    return json.dumps({'title':row['title'],'text':row['text']},ensure_ascii=False)
def messages_for(method,row):
    messages=[{'role':'system','content':CONFIG['system_prompt']}]
    if method=='few_shot':
        for demo in demos:
            messages += [{'role':'user','content':review_input(demo)}, {'role':'assistant','content':demo['reference_label']}]
    messages.append({'role':'user','content':review_input(row)})
    assert json.loads(messages[-1]['content']) == {'title':row['title'],'text':row['text']}
    return messages
def payload_for(method,row):
    return dict(model=CONFIG['hosted_model'],messages=messages_for(method,row),**settings)
def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def normalize(raw):
    text=raw.strip().upper()
    return text if text in LABELS else 'INVALID'
manifest={'location':'local Windows computer; class endpoint performs model inference',
          'model':CONFIG['hosted_model'],'base_url':CONFIG['hosted_base_url'],'settings':settings,
          'system_prompt':CONFIG['system_prompt'],'few_shot_filtered_indices':demo_ids,
          'test_fingerprint':fingerprint(test),'train_fingerprint':fingerprint(splits['train']),
          'concurrent_requests':2,'requests':[{'method':m,'filtered_index':r['filtered_index'],
             'payload':payload_for(m,r),'request_sha256':digest(payload_for(m,r))}
             for m in ('zero_shot','few_shot') for r in test]}
manifest_path=OUT/'request_manifest.json'
if manifest_path.exists(): assert json.loads(manifest_path.read_text(encoding='utf-8')) == manifest
else: manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'split_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print('PASS: supplied splits match canonical IDs, contents, class counts, shuffle, and disjointness.',flush=True)
print('PASS: all 600 actual requests audited; target inputs contain only title/text; demonstrations are prescribed training rows.',flush=True)
key=os.environ['CLASS_API_KEY']
def call(method,row):
    payload=payload_for(method,row)
    start=time.perf_counter()
    for attempt in range(4):
        try:
            response=requests.post(CONFIG['hosted_base_url']+'/chat/completions',json=payload,
                headers={'Authorization':'Bearer '+key},timeout=90)
            if response.status_code in (429,500,502,503,504) and attempt<3:
                time.sleep(2**attempt);continue
            response.raise_for_status()
            body=response.json()
            raw=body['choices'][0]['message'].get('content')
            # Preserve absent content as JSON null in the response evidence, invalid textual representation in predictions.
            if not isinstance(raw,str): raw=json.dumps(raw)
            pred={k:row[k] for k in ('original_row','filtered_index','title','text','reference_label')}
            pred.update(prediction=normalize(raw),raw_response=raw)
            return {'prediction':pred,'request_sha256':digest(payload),'response':body,
                    'elapsed_seconds':time.perf_counter()-start,'attempts':attempt+1,
                    'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        except (requests.ConnectionError,requests.Timeout):
            if attempt==3: raise
            time.sleep(2**attempt)
    raise RuntimeError('Retry limit reached')
all_runs={}
for method in ('zero_shot','few_shot'):
    evidence_path=OUT/(method+'_responses.jsonl')
    saved=load_jsonl(evidence_path) if evidence_path.exists() else []
    assert len({r['prediction']['filtered_index'] for r in saved})==len(saved)
    by_id={r['prediction']['filtered_index']:r for r in saved}
    expected={r['filtered_index']:r for r in test}
    assert set(by_id)<=set(expected)
    for i,result in by_id.items():
        assert result['request_sha256']==digest(payload_for(method,expected[i]))
        assert all(result['prediction'][k]==expected[i][k] for k in ('original_row','filtered_index','title','text','reference_label'))
        assert result['prediction']['prediction']==normalize(result['prediction']['raw_response'])
    pending=[r for r in test if r['filtered_index'] not in by_id]
    started=time.perf_counter()
    with evidence_path.open('a',encoding='utf-8') as stream, concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for offset in range(0,len(pending),2):
            futures=[pool.submit(call,method,r) for r in pending[offset:offset+2]]
            for future in futures:
                result=future.result()
                stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
                by_id[result['prediction']['filtered_index']]=result
            if len(by_id)%20==0: print(f'{method}: {len(by_id)}/300 saved',flush=True)
    assert len(by_id)==300 and set(by_id)==set(expected)
    predictions=[by_id[r['filtered_index']]['prediction'] for r in test]
    with (OUT/(method+'_predictions.jsonl')).open('w',encoding='utf-8') as stream:
        for r in predictions: stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    run={'count':300,'new_requests':len(pending),'resumed_requests':len(saved),
         'this_session_wall_seconds':time.perf_counter()-started,
         'sum_request_seconds':sum(r['elapsed_seconds'] for r in by_id.values()),
         'invalid':sum(r['prediction']=='INVALID' for r in predictions),
         'model':CONFIG['hosted_model'],'settings':settings,'concurrent_requests':2,
         'execution_location':'local Windows client calling class endpoint'}
    run_path=OUT/(method+'_run.json')
    if pending or not run_path.exists(): run_path.write_text(json.dumps(run,indent=2),encoding='utf-8')
    all_runs[method]=json.loads(run_path.read_text())
    print(method, 'PASS: 300 unique prescribed test IDs saved.',flush=True)
(OUT/'hosted_runs.json').write_text(json.dumps(all_runs,indent=2),encoding='utf-8')
print('COMPLETE: both frozen hosted baselines saved with raw responses and exact request provenance.',flush=True)
