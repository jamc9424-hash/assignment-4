# Comparison results — evaluation only, separate from prediction generation.
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from itertools import combinations
paths={"Zero-shot":WORK/"zero_shot_predictions.jsonl","Few-shot":WORK/"few_shot_predictions.jsonl",
       "Fine-tuned":SUBMISSION/"fine_tuned_predictions.jsonl"}
expected_by_id={r["filtered_index"]:r for r in test_rows}
ordered_ids=sorted(expected_by_id)
aligned={}
for name,path in paths.items():
    rows=load_jsonl(path)
    assert len(rows)==300 and len({r["filtered_index"] for r in rows})==300
    assert len({r["original_row"] for r in rows})==300
    assert {r["filtered_index"] for r in rows}==set(ordered_ids)
    for r in rows:
        e=expected_by_id[r["filtered_index"]]
        assert all(r[k]==e[k] for k in ("original_row","filtered_index","title","text","reference_label"))
        assert r["prediction"]==normalize_response(r["raw_response"])
    aligned[name]={r["filtered_index"]:r for r in rows}
    print(name,": PASS — exactly one prediction per prescribed test ID; exact content and labels.")
y_true=[expected_by_id[i]["reference_label"] for i in ordered_ids]
metrics, per_class, correctness, matrices = [],[],{},{}
matrix_labels=list(LABELS)+(["INVALID"] if any(r["prediction"]=="INVALID" for d in aligned.values() for r in d.values()) else [])
for name,by_id in aligned.items():
    predicted=[by_id[i]["prediction"] for i in ordered_ids]
    correct=np.array(predicted)==np.array(y_true)
    correctness[name]=correct
    metrics.append({"method":name,"balanced_accuracy":balanced_accuracy_score(y_true,predicted),
                    "macro_f1":f1_score(y_true,predicted,labels=list(LABELS),average="macro",zero_division=0),
                    "correct":int(correct.sum()),"incorrect":int((~correct).sum()),
                    "invalid":predicted.count("INVALID")})
    p,r,f,s=precision_recall_fscore_support(y_true,predicted,labels=list(LABELS),zero_division=0)
    for j,label in enumerate(LABELS):
        per_class.append({"method":name,"class":label,"precision":p[j],"recall":r[j],"f1":f[j],"support":int(s[j])})
    counts=confusion_matrix(y_true,predicted,labels=matrix_labels)[:3,:]
    matrices[name]=counts/counts.sum(axis=1,keepdims=True)
    assert np.allclose(matrices[name].sum(axis=1),1)
summary=pd.DataFrame(metrics).set_index("method")
display(summary)
display(pd.DataFrame(per_class))
fig,axes=plt.subplots(1,3,figsize=(18,4.5),layout="constrained")
for ax,(name,matrix) in zip(axes,matrices.items()):
    sns.heatmap(matrix,annot=True,fmt=".2f",vmin=0,vmax=1,cmap="Blues",ax=ax,
                xticklabels=matrix_labels,yticklabels=LABELS)
    ax.set(title=name,xlabel="Prediction",ylabel="Reference")
plt.show()
best=summary["macro_f1"].idxmax()
fig,ax=plt.subplots(figsize=(8,6),layout="constrained")
sns.heatmap(matrices[best],annot=True,fmt=".2f",vmin=0,vmax=1,cmap="Blues",ax=ax,
            xticklabels=matrix_labels,yticklabels=LABELS)
ax.set(title=best+" — highest macro-F1 (ties use displayed order)",xlabel="Prediction",ylabel="Reference")
plt.show()
pairwise=[]
for a,b in combinations(correctness,2):
    ca,cb=correctness[a],correctness[b]
    table=[[int((ca&cb).sum()),int((ca&~cb).sum())],
           [int((~ca&cb).sum()),int((~ca&~cb).sum())]]
    result=mcnemar(table,exact=True)
    pairwise.append({"A":a,"B":b,"both_correct":table[0][0],"A_only_correct":table[0][1],
                     "B_only_correct":table[1][0],"both_wrong":table[1][1],"exact_p_value":float(result.pvalue)})
display(pd.DataFrame(pairwise))
print("McNemar tests paired correctness, not macro-F1. Small p-values indicate asymmetric errors; "
      "they do not prove general superiority. Three unadjusted tests are exploratory.")
evaluation={"metrics":metrics,"per_class":per_class,"mcnemar":pairwise,"best_macro_f1":best,
            "files":{name:{"name":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in paths.items()}}
(WORK/"evaluation.json").write_text(json.dumps(evaluation,indent=2))
# Show concrete error examples without changing any model, prompts, or settings.
for name,by_id in aligned.items():
    errors=[by_id[i] for i in ordered_ids if by_id[i]["prediction"]!=by_id[i]["reference_label"]]
    print("\nFirst three errors in file order:",name)
    for row in errors[:3]: print(json.dumps(row,ensure_ascii=False))
