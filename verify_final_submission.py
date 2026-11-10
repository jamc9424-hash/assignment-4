"""Independently validate the four downloaded submission files against canonical data."""
import hashlib
import json
from pathlib import Path
import struct
import zipfile
import nbformat
from data_protocol import canonical_records, select_splits, LABELS, load_jsonl

ROOT=Path(__file__).resolve().parent
SUBMISSION=ROOT.parent.parent/'outputs/assignment4_submission'
REQUIRED={'assignment4_comparison.ipynb','adapter.zip','fine_tuned_predictions.jsonl','prompt_examples.jsonl'}
assert {p.name for p in SUBMISSION.iterdir() if p.is_file()}==REQUIRED
config=json.loads((ROOT/'experiment_config.json').read_text())
with zipfile.ZipFile(SUBMISSION/'adapter.zip') as archive:
    assert sorted(archive.namelist())==['adapter_config.json','adapter_model.safetensors']
    adapter=json.loads(archive.read('adapter_config.json'))
    assert adapter['base_model_name_or_path']=='Qwen/Qwen2.5-3B-Instruct'
    assert adapter['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    assert adapter['peft_type']=='LORA' and adapter['r']==16 and adapter['lora_alpha']==32
    weights=archive.read('adapter_model.safetensors')
    header_size=struct.unpack('<Q',weights[:8])[0]
    header=json.loads(weights[8:8+header_size])
    tensor_names=set(header)-{'__metadata__'}
    assert tensor_names and all('.lora_A.' in n or '.lora_B.' in n for n in tensor_names)
    adapter_hashes={n:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()}
test=select_splits(canonical_records(ROOT/'data/Gift_Cards.jsonl.gz'))['test']
expected={r['filtered_index']:r for r in test}
predictions=load_jsonl(SUBMISSION/'fine_tuned_predictions.jsonl')
assert len(predictions)==300 and len({r['original_row'] for r in predictions})==300
assert {r['filtered_index'] for r in predictions}==set(expected)
for row in predictions:
    assert set(row)=={'original_row','filtered_index','title','text','reference_label','prediction','raw_response'}
    assert all(row[k]==expected[row['filtered_index']][k] for k in ('original_row','filtered_index','title','text','reference_label'))
    normalized=row['raw_response'].strip().upper()
    assert row['prediction']==(normalized if normalized in LABELS else 'INVALID')
examples=load_jsonl(SUBMISSION/'prompt_examples.jsonl')
assert len(examples)==3 and {p['reference_label'] for p in examples}==set(LABELS)
for example in examples:
    assert set(example)=={'original_row','filtered_index','reference_label','messages'}
    row=expected[example['filtered_index']]
    assert example['original_row']==row['original_row'] and example['reference_label']==row['reference_label']
    assert example['messages']==[{'role':'system','content':config['system_prompt']},
        {'role':'user','content':json.dumps({k:row[k] for k in ('title','text')},ensure_ascii=False)}]
nb=nbformat.read(SUBMISSION/'assignment4_comparison.ipynb',as_version=4)
nbformat.validate(nb)
code_cells=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code_cells), 'Unexecuted code cell in final notebook'
assert not any(o.output_type=='error' for c in code_cells for o in c.outputs)
evidence=ROOT/'evidence/final'
evaluation=json.loads((evidence/'evaluation.json').read_text())
inference=json.loads((evidence/'inference_run.json').read_text())
training=json.loads((evidence/'training_run.json').read_text())
assert evaluation['files']['Fine-tuned']['sha256']==hashlib.sha256((SUBMISSION/'fine_tuned_predictions.jsonl').read_bytes()).hexdigest()
assert inference['adapter_hashes']==adapter_hashes and inference['count']==300
assert 'T4' in inference['gpu'] and 'T4' in training['gpu']
assert training['completed_optimizer_steps']==114 and training['epochs']==2
report={'passed':True,'predictions':300,'prompt_examples':3,'adapter_tensor_count':len(tensor_names),
        'all_notebook_code_executed':True,'no_notebook_errors':True,'adapter_hashes':adapter_hashes,
        'files':{n:hashlib.sha256((SUBMISSION/n).read_bytes()).hexdigest() for n in sorted(REQUIRED)}}
(evidence/'independent_submission_check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
