from pathlib import Path
import json, shutil
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parent
outputs=root.parent.parent/'outputs/hosted_baselines'
outputs.mkdir(parents=True,exist_ok=True)
nb=nbformat.v4.new_notebook()
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
nb.cells=[nbformat.v4.new_markdown_cell('''# Assignment 4 — completed hosted baselines

**Partial assignment: zero-shot and few-shot complete; fine-tuned model and final four-file submission are pending.**

Colab could not allocate a runtime. The approved 600 requests were therefore sent from the user's Windows computer by `run_hosted_local.py`; the hosted class endpoint performed model inference. This executed notebook independently verifies and scores the saved responses. Its cells do not claim to have generated those responses in Colab.

Both approaches used `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`, temperature 0, an 8-token limit, and `enable_thinking=False`. Two requests were allowed concurrently. Only review title and text were supplied for each target. Few-shot demonstrations were the approved training IDs 8849, 2359, and 4447. Prompts and settings were frozen before predictions. Validation remains unused. Invalid responses are retained and counted as incorrect.

To rerun this audit, install requests, numpy, pandas, matplotlib, scikit-learn, statsmodels, and Jupyter. Keep the accompanying `evidence/hosted_local` folder beside this notebook. The public source dataset is downloaded if absent. The request manifest and response records contain no API credential.
'''),nbformat.v4.new_code_cell((root/'data_protocol.py').read_text(encoding='utf-8'))]
code=(root/'evaluate_hosted_local.py').read_text(encoding='utf-8')
code='\n'.join(line for line in code.splitlines() if not line.startswith('from data_protocol import'))
nb.cells.append(nbformat.v4.new_code_cell(code))
nb.cells.append(nbformat.v4.new_markdown_cell('## Complete prompts for the same test review\nAll few-shot demonstrations are included below. The target reference label is excluded from these messages.'))
nb.cells.append(nbformat.v4.new_code_cell('''first_id=ids[0]
print("Test filtered_index:",first_id)
for method in ("zero_shot","few_shot"):
    print("\\n"+method+" complete request:")
    print(json.dumps(requests_by_key[method,first_id]["payload"],ensure_ascii=False,indent=2))
'''))
nb.cells.append(nbformat.v4.new_code_cell('''import importlib.metadata as metadata
versions={name:metadata.version(name) for name in ("numpy","pandas","matplotlib","scikit-learn","statsmodels","nbclient")}
print(json.dumps(versions,indent=2))
(EVIDENCE/"evaluation_package_versions.json").write_text(json.dumps(versions,indent=2));
'''))
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
evaluation=json.loads((root/'evidence/hosted_local/hosted_evaluation.json').read_text())
metrics=evaluation['metrics']
pair=evaluation['mcnemar']
report='# Hosted-model results — partial assignment\n\n'
report+='Both approaches classified the same 300 prescribed reviews, with 100 per reference class. All request, ID, content, and response-provenance checks passed.\n\n'
report+='| Approach | Balanced accuracy | Macro-F1 | Correct | Incorrect | Invalid |\n|---|---:|---:|---:|---:|---:|\n'
for m in metrics:
    report+=f"| {m['method'].replace('_',' ').title()} | {m['balanced_accuracy']:.2%} | {m['macro_f1']:.4f} | {m['correct']} | {m['incorrect']} | {m['invalid']} |\n"
report+=f"\nExact McNemar p-value: **{pair['exact_p_value']:.6g}**. Zero-shot alone was correct on {pair['A_only_correct']} reviews; few-shot alone on {pair['B_only_correct']}. "
report+=('This comparison provides evidence of different paired correctness at the 0.05 threshold.' if pair['exact_p_value']<.05 else 'This comparison does not establish a difference in paired correctness at the 0.05 threshold.')
report+=' This test concerns correctness, not macro-F1, and does not establish performance beyond this fixed test set.\n\n'
report+='The three examples were chosen from training reviews before testing: a favorable gift review, a non-evaluative subscription question, and a complaint about fees. Titles and text, including star-number titles and HTML entities, were preserved. Star-rating references may differ from the sentiment a reader infers from the text. No test result will be used to revise the approved prompts, examples, or fine-tuning settings.\n\n'
report+='**Still pending:** verified full fine-tuning run, submitted adapter reload, 300 fine-tuned predictions, three actual fine-tuned prompt examples, and the final three-way report. These hosted outputs are supporting artifacts, not substitutes for the required fine-tuned submission files.\n'
nb.cells.append(nbformat.v4.new_markdown_cell(report))
nbformat.validate(nb)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
assert not any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
nbformat.write(nb,root/'hosted_results.ipynb')
nbformat.write(nb,outputs/'hosted_results.ipynb')
(outputs/'hosted_report.md').write_text(report,encoding='utf-8')
shutil.copytree(root/'evidence/hosted_local',outputs/'evidence/hosted_local',dirs_exist_ok=True)
for name in ('zero_shot_predictions.jsonl','few_shot_predictions.jsonl'):
    shutil.copy2(root/'evidence/hosted_local'/name,outputs/name)
shutil.copy2(root/'run_hosted_local.py',outputs/'run_hosted_local.py')
shutil.copy2(root/'data_protocol.py',outputs/'data_protocol.py')
shutil.copy2(root/'experiment_config.json',outputs/'experiment_config.json')
print(report)
print('Executed notebook and supporting files:',outputs)
