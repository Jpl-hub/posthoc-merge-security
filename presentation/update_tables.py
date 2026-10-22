"""Regenerate the report's small tables from independently verified result files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "results/phase3/results.json").read_text())
utility = json.loads((ROOT / "results/phase3/utility.json").read_text())
recent_root = ROOT / 'results/normalization_alignment'
recent = json.loads((recent_root / 'evaluation/results.json').read_text())
assert json.loads((recent_root / 'evaluation/independent_validation.json').read_text())['status'] == 'passed'
recent_gate = json.loads((recent_root / 'evaluation/full_native_gate.json').read_text())
recent_source = json.loads((recent_root / 'results.json').read_text())


def recent_value(stage, task):
    row = next(r for r in recent if (r['preset'], r['stage'], r['dataset']) == ('released', stage, task))
    return f"{100*row['top1']:.2f}"


def accuracy(defense, merger, scope, task, target="MNIST", partner="SVHN", order=None):
    hits = [r for r in rows if (r["defense"], r["merger"], r["attack_scope"],
            r["dataset"], r["target"], r["partner"]) ==
            (defense, merger, scope, task, target, partner)
            and (order is None or r.get("order") == order)]
    assert len(hits) == 1, (defense, merger, task, len(hits))
    return f'{hits[0]["accuracy_percent"]:.2f}'


def pair(defense, merger, scope, order=None):
    return " / ".join(accuracy(defense, merger, scope, t, order=order)
                      for t in ["MNIST", "SVHN"])


def table(name, columns, header, body):
    lines = [rf"\newcommand{{\{name}}}{{", rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}",
             r"\toprule", header + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in body]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", "}"])


blocks = []
body = []
for target, partner in [("MNIST", "SVHN"), ("DTD", "MNIST"), ("SVHN", "MNIST")]:
    defense = "Phase1_" + target
    body.append([target, accuracy(defense, "Standalone", "None", target, target, partner),
                 accuracy("Unprotected", "TA", "All experts", target, target, partner),
                 accuracy(defense, "TA", "All experts", target, target, partner)])
blocks.append(table("PhaseOneTable", "lrrr", r"被保护任务 & 独立 & 无保护AGA后 & 保护AGA后", body))

body = []
for name, label in [("Unprotected", "无保护"), ("Phase1_MNIST", r"偏置扰动$^*$"),
                    ("O/B", r"补偿 O/B$^\dagger$"), ("A/B", r"补偿 A/B$^\dagger$"),
                    ("PaRaMS", r"PaRaMS$^*$")]:
    body.append([label, accuracy(name, "Standalone", "None", "MNIST"),
                 pair(name, "TA", "None"), pair(name, "TA", "Protected expert"),
                 pair(name, "TA", "All experts")])
body.append(['功能保持对照', recent_value('standalone', 'MNIST')] +
            [' / '.join(recent_value(stage, t) for t in ['MNIST', 'SVHN'])
             for stage in ['direct', 'AGA_target_only', 'AGA_both']])
blocks.append(table("MainTable", "lrrrr",
    r"方法 & 独立MNIST & 直接融合 & \shortstack{AGA处理\\保护专家} & \shortstack{AGA处理\\全部专家}", body))

body = []
for name in ["O/B", "A/B"]:
    u = next(x for x in utility if x["defense"] == name)
    body.append([name, f'{u["parameter_distance_changed_tensors"]:.5f}',
                 f'{u["worst_relative_task_margin_drift"]:.5f}',
                 f'{u["n_inputs_over_task_budget"]}/{u["n"]}', "超过0.5"])
assert recent_gate['passed']
body.append(['功能保持对照', f"{recent_source['geometry']['parameter_distance']:.5f}",
             f"{recent_gate['worst_relative_margin_drift']:.5f}", f"0/{recent_gate['n']}", '已查样本通过'])
blocks.append(table("UtilityTable", "lrrrl",
    r"实现 & 参数距离 & 最大间隔漂移 & 越界样本 & 任务约束", body))

body = []
for name, label in [("Unprotected", "无保护"), ("Phase1_MNIST", "偏置扰动"), ("A/B", r"补偿 A/B$^\dagger$")]:
    body.append([label] + [pair(name, "NUFILT", s, ["MNIST", "SVHN"])
                          for s in ["None", "Protected expert", "All experts"]])
blocks.append(table("NufiltTable", "lrrr",
    r"方法 & 无攻击 & \shortstack{AGA处理\\保护专家} & \shortstack{AGA处理\\全部专家}", body))

eight = json.loads((ROOT / "results/phase3/eight_task/results.json").read_text())
assert json.loads((ROOT / "results/phase3/eight_task/independent_validation.json").read_text())["status"].startswith("passed")
body=[]
for name in ["Unprotected","O/B","A/B"]:
    values=[next(r["accuracy_percent"] for r in eight if (r["target"],r["defense"],r["attack_scope"],r["dataset"]) ==
                 ("MNIST",name,"Protected expert",task)) for task in ["MNIST","DTD","SVHN"]]
    body.append(["无保护" if name=="Unprotected" else name+r"$^\dagger$"]+[f"{v:.2f}" for v in values])
blocks.append(table("EightTable","lrrr",r"八专家TA0.2 & MNIST & DTD & SVHN",body))

nufilt=json.loads((ROOT/"results/phase3/nufilt_other_tasks/results.json").read_text())
assert json.loads((ROOT/"results/phase3/nufilt_other_tasks/independent_validation.json").read_text())["status"].startswith("passed")
body=[]
for target,partner in [("MNIST","SVHN"),("DTD","MNIST"),("SVHN","MNIST")]:
    if target=="MNIST":
        protected=pair("Phase1_MNIST","NUFILT","Protected expert",[target,partner])
        control=pair("Unprotected","NUFILT","Protected expert",[target,partner])
    else:
        def nf(name):
            return " / ".join(f'{next(r["accuracy_percent"] for r in nufilt if (r["target"],r["partner"],r["defense"],r["attack_scope"],r["order"],r["dataset"]) == (target,partner,name,"Protected expert",[target,partner],task)):.2f}' for task in [target,partner])
        protected,control=nf("Phase1_"+target),nf("Unprotected")
    direct=" / ".join(accuracy("Phase1_"+target,"TA","Protected expert",t,target,partner) for t in [target,partner])
    body.append([target+" / "+partner,direct,protected,control])
blocks.append(table("PhaseOneRecoveryTable","lrrr",r"目标／伙伴 & TA+AGA & NUFILT+AGA & NUFILT无保护",body))
(ROOT / "presentation/tables.tex").write_text(
    "% Generated by update_tables.py; do not edit numbers manually.\n\n" + "\n\n".join(blocks) + "\n")
print("Generated six tables from verified frozen records.")

