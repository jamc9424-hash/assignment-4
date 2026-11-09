"""Evaluate saved responses only; do not modify prompts or models based on these results."""
from pathlib import Path
import hashlib, json
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from data_protocol import LABELS, canonical_records, select_splits, fingerprint, load_jsonl, download_source, audit_splits

ROOT=Path.cwd()
EVIDENCE=ROOT/'evidence/hosted_local'
manifest=json.loads((EVIDENCE/'request_manifest.json').read_text(encoding='utf-8'))
source=download_source(ROOT/'data/Gift_Cards.jsonl.gz')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='e03a258ebd7b1e2591b09862aab65d3dcde9c300744d800aa1fbaf374b7c2340'
splits=select_splits(canonical_records(source))
audit=audit_splits(splits,splits)
assert fingerprint(splits['test'])==manifest['test_fingerprint']
assert fingerprint(splits['train'])==manifest['train_fingerprint']
test_by_id={r['filtered_index']:r for r in splits['test']}
train_by_id={r['filtered_index']:r for r in splits['train']}
ids=sorted(test_by_id)
assert len(ids)==300
assert manifest['few_shot_filtered_indices']==[8849,2359,4447]
assert len(manifest['requests'])==600
def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def normalize(raw):
    value=raw.strip().upper()
    return value if value in LABELS else 'INVALID'
requests_by_key={}
for request in manifest['requests']:
    method,i=request['method'],request['filtered_index']
    assert method in ('zero_shot','few_shot') and i in test_by_id
    assert (method,i) not in requests_by_key
    messages=[{'role':'system','content':manifest['system_prompt']}]
    if method=='few_shot':
        for demo_id in manifest['few_shot_filtered_indices']:
            demo=train_by_id[demo_id]
            messages.extend([{'role':'user','content':json.dumps({k:demo[k] for k in ('title','text')},ensure_ascii=False)},
                             {'role':'assistant','content':demo['reference_label']}])
    messages.append({'role':'user','content':json.dumps({k:test_by_id[i][k] for k in ('title','text')},ensure_ascii=False)})
    assert request['payload']==dict(model=manifest['model'],messages=messages,**manifest['settings'])
    assert request['request_sha256']==digest(request['payload'])
    requests_by_key[method,i]=request
print('PASS: all 600 actual requests verified; target input allowlist and training-only demonstrations match.')
reference=[test_by_id[i]['reference_label'] for i in ids]
predictions,correctness,summary,per_class={},{},[],[]
for method in ('zero_shot','few_shot'):
    rows=load_jsonl(EVIDENCE/(method+'_predictions.jsonl'))
    responses=load_jsonl(EVIDENCE/(method+'_responses.jsonl'))
    assert len(rows)==len(responses)==300
    assert len({r['original_row'] for r in rows})==300
    by_id={r['filtered_index']:r for r in rows}
    response_by_id={r['prediction']['filtered_index']:r for r in responses}
    assert set(by_id)==set(response_by_id)==set(ids)
    for i,row in by_id.items():
        assert set(row)=={'original_row','filtered_index','title','text','reference_label','prediction','raw_response'}
        assert all(row[k]==test_by_id[i][k] for k in ('original_row','filtered_index','title','text','reference_label'))
        evidence=response_by_id[i]
        assert evidence['prediction']==row
        assert evidence['request_sha256']==requests_by_key[method,i]['request_sha256']
        original=evidence['response']['choices'][0]['message'].get('content')
        assert row['raw_response']==(original if isinstance(original,str) else json.dumps(original))
        assert row['prediction']==normalize(row['raw_response'])
    pred=[by_id[i]['prediction'] for i in ids]
    predictions[method]=pred
    correct=np.array(pred)==np.array(reference)
    correctness[method]=correct
    summary.append({'method':method,'balanced_accuracy':balanced_accuracy_score(reference,pred),
        'macro_f1':f1_score(reference,pred,labels=list(LABELS),average='macro',zero_division=0),
        'correct':int(correct.sum()),'incorrect':int((~correct).sum()),'invalid':pred.count('INVALID')})
    p,r,f,s=precision_recall_fscore_support(reference,pred,labels=list(LABELS),zero_division=0)
    per_class.extend({'method':method,'class':label,'precision':float(p[j]),'recall':float(r[j]),'f1':float(f[j]),'support':int(s[j])} for j,label in enumerate(LABELS))
    print(method, 'PASS: exactly 300 prescribed IDs; raw responses and request provenance match.')
print(pd.DataFrame(summary).to_string(index=False))
print(pd.DataFrame(per_class).to_string(index=False))
a,b=correctness['zero_shot'],correctness['few_shot']
table=[[int((a&b).sum()),int((a&~b).sum())],[int((~a&b).sum()),int((~a&~b).sum())]]
test=mcnemar(table,exact=True)
paired={'A':'zero_shot','B':'few_shot','both_correct':table[0][0],'A_only_correct':table[0][1],
        'B_only_correct':table[1][0],'both_wrong':table[1][1],'exact_p_value':float(test.pvalue)}
print('Exact paired McNemar:',json.dumps(paired))
labels=list(LABELS)+(['INVALID'] if any('INVALID' in p for p in predictions.values()) else [])
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,(method,pred) in zip(axes,predictions.items()):
    counts=confusion_matrix(reference,pred,labels=labels)[:3,:]
    norm=counts/counts.sum(axis=1,keepdims=True)
    assert np.allclose(norm.sum(axis=1),1)
    ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
    ax.set(xticks=range(len(labels)),xticklabels=labels,yticks=range(3),yticklabels=LABELS,
           xlabel='Prediction',ylabel='Reference',title=method.replace('_',' ').title())
    for j in range(3):
        for k in range(len(labels)):
            ax.text(k,j,f'{norm[j,k]:.2f}\n({counts[j,k]})',ha='center',va='center',color='white' if norm[j,k]>.55 else 'black')
fig.savefig(EVIDENCE/'hosted_confusion_matrices.png',dpi=160)
plt.show()
result={'status':'two hosted baselines complete; fine-tuned comparison pending',
        'metrics':summary,'per_class':per_class,'mcnemar':paired,
        'prediction_files':{m:{'sha256':hashlib.sha256((EVIDENCE/(m+'_predictions.jsonl')).read_bytes()).hexdigest()} for m in predictions},
        'test_fingerprint':manifest['test_fingerprint'],'execution_location':manifest['location']}
(EVIDENCE/'hosted_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Saved results. No prompt, example, or training-setting changes were made after observing test results.')
