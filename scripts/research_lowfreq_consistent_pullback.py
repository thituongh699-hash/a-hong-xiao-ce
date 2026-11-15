"""Bounded retrospective repair: reconcile entry with the unchanged MA10 exit."""
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import research_lowfreq_50k as base

OUT=ROOT/"reports/lowfreq-pullback-repair-2026-10-10"


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    specs=[{**s,"id":s["id"]+"_above_ma10"} for s in base.SPECS if s["id"] in {"pullback3","pullback5"}]
    declaration=dict(version="consistent-pullback-v1",retrospective=True,recent_year_already_observed=True,
                     purpose="repair entry/exit inconsistency without optimizing thresholds or exits",
                     base_matrix_sha256=json.loads((base.OUT/"matrix-manifest.json").read_text(encoding="utf-8"))["matrix_sha256"],
                     added_condition="signal close is not below MA10: base_entry AND NOT base_exit; all other rules, scores and exits unchanged",
                     selection="maximum training CAGR with>=20closed trades; train2023-10..2025-09 then freeze; recent/full/stresses descriptive historical replays",
                     specs=specs,source_sha256=base.digest(Path(__file__)))
    declaration_path=OUT/"predeclared-repair.json"
    if declaration_path.exists():
        assert json.loads(declaration_path.read_text(encoding="utf-8"))==declaration
    else:
        base.write_json(declaration_path,declaration)
    shutil.copytree(base.OUT/"dividend-cache",OUT/"dividend-cache",dirs_exist_ok=True)
    data=base.reuse_matrix(base.OUT)
    for spec in specs:
        old=spec["id"].removesuffix("_above_ma10")
        data["entry"][spec["id"]]=data["entry"][old] & ~data["exit"][old]
        data["exit"][spec["id"]]=data["exit"][old]
        data["score"][spec["id"]]=data["score"][old]
    base.archive_run_sources(OUT,"train")
    copied=OUT/"executed-source/train/scripts/research_lowfreq_consistent_pullback.py"
    copied.write_bytes(Path(__file__).read_bytes())
    trained=base.resolved_runs(data,specs,base.START,base.TRAIN_END,OUT,"train")
    table=base.summary_table(trained)
    table.to_csv(OUT/"training-candidates.csv",index=False,encoding="utf-8-sig")
    eligible=table[table.valid.eq(True)&table.trades_closed.ge(20)&table.cagr.notna()]
    selected=eligible.sort_values(["cagr","id"],ascending=[False,True]).iloc[0].id if len(eligible) else None
    frozen=dict(selected_id=selected,selected_spec=next((s for s in specs if s["id"]==selected),None),
                retrospective=True,recent_year_already_observed=True,selection_uses_recent_year=False,
                training_table_sha256=base.digest(OUT/"training-candidates.csv"))
    base.write_json(OUT/"frozen-selection.json",frozen)
    checks=base.evaluate(data,frozen,OUT)
    summaries={k:v["summary"] for k,v in checks.items()}
    base.write_json(OUT/"study-summary.json",dict(retrospective=True,recent_year_already_observed=True,frozen_selection=frozen,validations=summaries))
    lines=["# 回调策略入场与退出一致性修正","","这是一轮回顾性修正，最近一年的原版结果已经看过，不是新的独立验证。",
           "只增加信号收盘价不低于MA10。回调阈值、排序、止损、持有期、费用和账户规则均不改变。",
           "其经济假设仍是中期上涨中的暂时回撤恢复；新增条件只让恢复确认与MA10离场规则保持一致，不能证明抛压来自获利卖盘。",
           "","## 前两年比较","",base.readable_table(table).to_markdown(index=False),"",
           f"按前两年冻结：{selected}。","","## 回顾性账户验证","",base.readable_table(base.summary_table(checks)).to_markdown(index=False),"",
           "依然存在当前名单生存者偏差、历史ST涨跌幅缺口、日线成交近似、公告数据库历史版本缺口和历史复用问题。"]
    (OUT/"修正结果.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(selected=selected,validations=summaries),ensure_ascii=False),flush=True)


if __name__=="__main__":main()
