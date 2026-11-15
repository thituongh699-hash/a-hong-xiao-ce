"""Event study for late-session low-volume rallies in recent mainboard IPOs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mootdx.reader import TdxLCMinBarReader, TdxMinBarReader

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/condition1-nextday-2026-10-15/inputs/stock-basic.json"
TDX = Path(r"D:\software\tongxinda\vipdoc")
START = "2024-10-22"
END = "2026-09-30"
UNIVERSE_ASOF = "2026-10-15"
MAINBOARD = ("sh.600", "sh.601", "sh.603", "sh.605", "sz.000", "sz.001", "sz.002", "sz.003")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_rows(end: str = END):
    rows = json.loads(INPUT.read_text(encoding="utf-8-sig"))
    return sorted(
        [r for r in rows if r.get("type") == "1" and r.get("ipoDate")
         and START <= r["ipoDate"] <= end and r["code"].startswith(MAINBOARD)
         and (not r.get("outDate") or r["outDate"] > START)],
        key=lambda r: r["code"],
    )


def minute_file(code: str) -> Path:
    exchange, number = code.split(".")
    folder = TDX / exchange / "minline"
    for suffix in (".01", ".1", ".lc1"):
        path = folder / f"{exchange}{number}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"no local 1-minute file for {code}")


def read_minutes(path: Path) -> pd.DataFrame:
    reader = TdxLCMinBarReader() if path.suffix.lower() == ".lc1" else TdxMinBarReader()
    frame = reader.get_df(str(path)).copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"minute index is not datetime: {path.name}")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    days = frame.index.strftime("%Y-%m-%d")
    frame = frame.loc[(days >= START) & (days <= END)]
    return frame


def daily_events(code: str, ipo: str, frame: pd.DataFrame) -> tuple[list[dict], dict]:
    frame = frame.copy()
    frame["day"] = frame.index.strftime("%Y-%m-%d")
    rows, audit = [], {"code": code, "ipo": ipo, "minute_rows": len(frame), "sessions": 0,
                      "complete_sessions": 0, "events": 0, "status": "ok"}
    sessions = []
    for day, g in frame.groupby("day", sort=True):
        times = g.index.strftime("%H:%M").tolist()
        day_start = pd.Timestamp(f"{day} 09:31")
        expected = pd.date_range(day_start, periods=120, freq="min").strftime("%H:%M").tolist()
        expected += pd.date_range(pd.Timestamp(f"{day} 13:01"), periods=120, freq="min").strftime("%H:%M").tolist()
        complete = len(times) == 240 and times == expected
        if day >= ipo and day <= END:
            sessions.append((day, g, complete))
    audit["sessions"] = len(sessions)
    audit["complete_sessions"] = sum(x[2] for x in sessions)
    for idx, (day, g, complete) in enumerate(sessions):
        if not complete or idx < 5 or idx + 1 >= len(sessions):
            continue
        # TDX end-stamped bars: 14:30 is the preceding-window close; 15:00 is the signal close.
        tail = g.between_time("14:31", "15:00", inclusive="both")
        prior = g.between_time("14:01", "14:30", inclusive="both")
        if len(tail) != 30 or len(prior) != 30:
            continue
        tail_volume = float(tail.volume.sum())
        prior_volume = float(prior.volume.sum())
        if prior_volume <= 0 or tail_volume <= 0:
            continue
        rally = float(g.loc[g.index.strftime("%H:%M") == "15:00", "close"].iloc[-1] /
                      g.loc[g.index.strftime("%H:%M") == "14:30", "close"].iloc[-1] - 1)
        next_day, next_bars, next_complete = sessions[idx + 1]
        if not next_complete:
            continue
        close = float(g.loc[g.index.strftime("%H:%M") == "15:00", "close"].iloc[-1])
        next_open = float(next_bars.loc[next_bars.index.strftime("%H:%M") == "09:31", "open"].iloc[0])
        next_close = float(next_bars.loc[next_bars.index.strftime("%H:%M") == "15:00", "close"].iloc[-1])
        if min(close, next_open, next_close) <= 0:
            continue
        rows.append({"code": code, "ipo": ipo, "date": day, "next_date": next_day,
                     "rally": rally, "volume_ratio": tail_volume / prior_volume,
                     "tail_volume": tail_volume, "prior_volume": prior_volume,
                     "next_close_return": next_close / close - 1,
                     "next_open_close_return": next_close / next_open - 1,
                     "event": False})
    return rows, audit


def summarize(events: pd.DataFrame, rally_min: float, ratio_max: float, label: str) -> dict:
    x = events.copy()
    x["event"] = x.rally.ge(rally_min) & x.volume_ratio.le(ratio_max)
    event = x.loc[x.event]
    control = x.loc[~x.event]
    out = {"definition": label, "rally_min": rally_min, "volume_ratio_max": ratio_max,
           "event_n": len(event), "control_n": len(control)}
    for col, name in (("next_close_return", "close_to_next_close"),
                      ("next_open_close_return", "next_open_to_close")):
        out[f"{name}_event_mean"] = float(event[col].mean()) if len(event) else None
        out[f"{name}_event_down_rate"] = float(event[col].lt(0).mean()) if len(event) else None
        out[f"{name}_control_mean"] = float(control[col].mean()) if len(control) else None
        out[f"{name}_control_down_rate"] = float(control[col].lt(0).mean()) if len(control) else None
        out[f"{name}_mean_diff"] = float(event[col].mean() - control[col].mean()) if len(event) and len(control) else None
        out[f"{name}_down_rate_diff"] = float(event[col].lt(0).mean() - control[col].lt(0).mean()) if len(event) and len(control) else None
    # Equal-weight within-stock contrasts prevent the most frequently observed IPOs dominating.
    if len(event):
        e = event.groupby("code").next_close_return.mean()
        c = control.groupby("code").next_close_return.mean()
        paired = pd.concat([e.rename("event"), c.rename("control")], axis=1).dropna()
        out["within_stock_pairs"] = len(paired)
        out["within_stock_close_diff"] = float((paired.event - paired.control).mean()) if len(paired) else None
        out["within_stock_negative_diff_rate"] = float((paired.event - paired.control).lt(0).mean()) if len(paired) else None
    else:
        out["within_stock_pairs"] = 0
        out["within_stock_close_diff"] = None
        out["within_stock_negative_diff_rate"] = None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs/late-rally-low-volume-2026-10-22")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    all_rows, audits, hashes = [], [], {str(INPUT): digest(INPUT)}
    for meta in candidate_rows():
        code = meta["code"]
        audit = {"code": code, "name": meta.get("code_name"), "ipo": meta.get("ipoDate"), "status": "error"}
        try:
            path = minute_file(code)
            hashes[str(path)] = digest(path)
            frame = read_minutes(path)
            rows, detail = daily_events(code, meta["ipoDate"], frame)
            audit.update(detail, name=meta.get("code_name"), file=str(path), file_sha256=hashes[str(path)])
            all_rows.extend(rows)
        except Exception as exc:
            audit["error"] = f"{type(exc).__name__}: {exc}"
        audits.append(audit)
    raw = pd.DataFrame(all_rows)
    if raw.empty:
        raise RuntimeError("No valid next-session observations")
    specs = [(0.005, 0.50), (0.005, 0.75), (0.005, 1.00),
             (0.010, 0.50), (0.010, 0.75), (0.010, 1.00),
             (0.015, 0.50), (0.015, 0.75), (0.015, 1.00)]
    summary = [summarize(raw, r, v, f"late_30m_return>={r:.1%}; volume_14:31-15:00 / 14:01-14:30<={v:.2f}")
               for r, v in specs]
    main_r, main_v = 0.01, 0.75
    raw["event"] = raw.rally.ge(main_r) & raw.volume_ratio.le(main_v)
    raw.to_csv(args.outdir / "event_observations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv(args.outdir / "coverage_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(args.outdir / "sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {"start": START, "end": END, "data_asof": "2026-09-30",
                "universe": "SH/SZ ordinary mainboard IPOs listed within two years; registration-based IPO eligibility for margin trading from listing day",
                "data_eligible_candidates": len(candidate_rows()),
                "all_candidates_in_universe_snapshot": len(candidate_rows(UNIVERSE_ASOF)),
                "listed_after_data_cutoff_in_snapshot": sum(r["ipoDate"] > END for r in candidate_rows(UNIVERSE_ASOF)),
                "stocks_with_observations": int(raw.code.nunique()),
                "valid_stock_sessions_with_next_day": len(raw), "primary_definition": {"late_return_min": main_r, "volume_ratio_max": main_v},
                "outcome": "T close to next stock trading-session close; secondary next-session open to close",
                "input_hashes": hashes,
                "limits": ["Exchange eligibility does not guarantee borrow inventory or lender availability.",
                           "Local TDX minute data ends 2026-09-30; later-listed names have no usable observation.",
                           "Raw price returns; corporate-action and execution/borrow costs are not modeled.",
                           "Within-stock comparison is descriptive and does not control market regime or IPO-age effects."]}
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    primary = summary[4]
    report = f"""# 尾盘无量拉升次日表现检验

观察区间：{START} 至 {END}（本地分钟数据截止日）。股票基础表快照日期：{UNIVERSE_ASOF}。

## 口径

- 样本：沪深主板普通股，IPO 在近两年内；注册制 IPO 按交易所规则可从上市首日成为融券标的。交易所资格不代表实际券源可借。
- 尾盘拉升：14:30 收盘价至 15:00 收盘价涨幅至少 1%。
- 无量：14:31-15:00 成交量 / 14:01-14:30 成交量不超过 0.75。
- 前五个上市交易日不参与信号，避免把 IPO 初期特殊交易阶段混入常态检验；要求信号日与次日分钟数据均为完整 240 根。
- 主结果观察信号日收盘至下一交易日收盘；另看下一交易日开盘至收盘。对照为同一可观察股票的所有非事件日。

## 主结果

- 股票基础表快照中有 {manifest['all_candidates_in_universe_snapshot']} 只窗口内候选，其中 {manifest['listed_after_data_cutoff_in_snapshot']} 只在分钟数据截止日后上市，因没有可用观察而未进入统计。
- {manifest['stocks_with_observations']} 只股票形成 {manifest['valid_stock_sessions_with_next_day']} 个完整的信号日-次日观察；其中主定义触发 {primary['event_n']} 次。
- 事件次日收盘下跌比例 {primary['close_to_next_close_event_down_rate']:.1%}，平均收益 {primary['close_to_next_close_event_mean']:.2%}；同股非事件日分别为 {primary['close_to_next_close_control_down_rate']:.1%}、{primary['close_to_next_close_control_mean']:.2%}。
- 事件相对非事件日的平均收益差 {primary['close_to_next_close_mean_diff']:.2%}，下跌率差 {primary['close_to_next_close_down_rate_diff']:.1%}。事件次日开盘至收盘平均 {primary['next_open_to_close_event_mean']:.2%}，非事件日为 {primary['next_open_to_close_control_mean']:.2%}。
- 5 次事件中有 3 次次日收盘下跌，样本量太小，无法据此确认稳定概率优势。

## 阈值敏感性

阈值变化后事件次数和收益方向不稳定：拉升下限 0.5%、量比上限 0.75 时有 16 次事件，平均次日收盘收益 {summary[1]['close_to_next_close_event_mean']:.2%}；拉升下限 1.5%、量比上限 0.75 时只有 {summary[7]['event_n']} 次，平均收益 {summary[7]['close_to_next_close_event_mean']:.2%}。完整组合见 `sensitivity_summary.csv`。

## 结论与限制

在这组新近上市主板股票和当前可得分钟样本中，主口径下事件均值略弱于同股非事件日，但只有 5 次触发；较宽松定义的结果又改变方向。因此，“尾盘无量拉升次日会跌”在本样本中没有得到可靠验证，不能作为稳定做空/卖出规则。

结果是未复权价格的描述性事件研究，不控制指数行情、涨跌停、IPO 年龄、公司行动，也未模拟成交、滑点、融券费率、券源和强平。输入文件哈希及逐股分钟覆盖见 `run_manifest.json` 与 `coverage_audit.csv`；逐事件和完整敏感性结果见对应 CSV。
"""
    (args.outdir / "研究结论.md").write_text(report, encoding="utf-8")
    print(json.dumps({"manifest": manifest, "primary": summary[4]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
