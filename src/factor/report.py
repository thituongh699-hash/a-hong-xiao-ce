"""把研究结果渲染成可离线打开的单文件 HTML 报告。

不引外部图表库：分层净值用内联 SVG 手绘，报告拷到任何机器上都能打开。
数值全部来自 ``run_study`` 的返回结果，渲染层不重新计算、不挑选最优参数。
"""

from __future__ import annotations

import html
import json
from typing import Iterable, Sequence

from src.utils.brand import BRAND_NAME

_PALETTE = ("#B4B2A9", "#85B7EB", "#378ADD", "#185FA5", "#042C53")
_VERDICT_STYLE = {
    "通过": ("#E1F5EE", "#0F6E56"),
    "待补证": ("#FAEEDA", "#854F0B"),
    "淘汰": ("#FCEBEB", "#A32D2D"),
}


def _fmt(value: object, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if number != number:  # NaN
        return "—"
    return f"{number:+.{digits}f}{suffix}" if abs(number) < 1e6 else f"{number:.3e}"


def _pct(value: object, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if number != number:
        return "—"
    return f"{number * 100:+.{digits}f}%"


def _badge(verdict: str) -> str:
    background, color = _VERDICT_STYLE.get(verdict, ("#F1EFE8", "#5F5E5A"))
    return (
        f'<span style="display:inline-block;padding:1px 8px;border-radius:10px;'
        f'background:{background};color:{color};font-size:12px;font-weight:500">{html.escape(verdict)}</span>'
    )


def _svg_lines(labels: Sequence[str], series: dict[str, Sequence[float]], width: int = 660, height: int = 190) -> str:
    values = [v for seq in series.values() for v in seq if v == v]
    if not values or not labels:
        return '<p style="color:#888780;font-size:13px">无分层净值数据</p>'
    low, high = min(values), max(values)
    if high - low < 1e-9:
        high = low + 1e-6
    pad_left, pad_right, pad_top, pad_bottom = 8, 8, 10, 18
    span_x = max(1, len(labels) - 1)
    inner_w = width - pad_left - pad_right
    inner_h = height - pad_top - pad_bottom

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="分层回测累计净值曲线">']
    for index, (name, seq) in enumerate(series.items()):
        points = []
        for i, value in enumerate(seq):
            if value != value:
                continue
            x = pad_left + inner_w * i / span_x
            y = pad_top + inner_h * (1 - (value - low) / (high - low))
            points.append(f"{x:.1f},{y:.1f}")
        if not points:
            continue
        colour = _PALETTE[min(index, len(_PALETTE) - 1)]
        parts.append(
            f'<polyline fill="none" stroke="{colour}" stroke-width="1.8" points="{" ".join(points)}"/>'
        )
    parts.append(
        f'<text x="{pad_left}" y="{height - 4}" font-size="11" fill="#888780">{html.escape(labels[0])}</text>'
        f'<text x="{width - pad_right}" y="{height - 4}" font-size="11" fill="#888780" text-anchor="end">'
        f'{html.escape(labels[-1])}</text>'
        f'<text x="{pad_left}" y="{pad_top + 8}" font-size="11" fill="#B4B2A9">{high:.2f}</text>'
        f'<text x="{pad_left}" y="{pad_top + inner_h - 2}" font-size="11" fill="#B4B2A9">{low:.2f}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _nav_legend(series: dict[str, Sequence[float]]) -> str:
    items = []
    for index, name in enumerate(series):
        colour = _PALETTE[min(index, len(_PALETTE) - 1)]
        items.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px">'
            f'<span style="width:10px;height:10px;border-radius:2px;background:{colour}"></span>'
            f'{html.escape(name)}</span>'
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:12px;font-size:12px;color:#5F5E5A;margin-bottom:6px">'
        + "".join(items)
        + "</div>"
    )


def _svg_ic(series: Sequence[dict[str, object]], width: int = 660, height: int = 110) -> str:
    values = [float(item["value"]) for item in series if item.get("value") is not None]
    if not values:
        return '<p style="color:#888780;font-size:13px">无 IC 序列</p>'
    limit = max(abs(min(values)), abs(max(values)), 1e-6)
    step = width / max(1, len(values))
    mid = height / 2
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="逐期 RankIC 柱状图">',
        f'<line x1="0" y1="{mid}" x2="{width}" y2="{mid}" stroke="#D3D1C7" stroke-width="0.5"/>',
    ]
    for i, item in enumerate(series):
        value = float(item["value"])
        bar_h = (mid - 6) * abs(value) / limit
        x = i * step + step * 0.15
        y = mid - bar_h if value >= 0 else mid
        colour = "#A32D2D" if value >= 0 else "#0F6E56"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{step * 0.7:.1f}" height="{bar_h:.1f}" fill="{colour}" opacity="0.75"/>'
        )
    parts.append(f'<text x="0" y="{height - 2}" font-size="11" fill="#888780">IC 峰值 ±{limit:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]], align_right: bool = True) -> str:
    head = "".join(
        f'<th style="text-align:{"right" if align_right else "left"};padding:6px 10px;'
        f'border-bottom:0.5px solid #D3D1C7;font-weight:500;color:#5F5E5A;font-size:12px">{html.escape(h)}</th>'
        for h in headers
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td style="text-align:{"right" if align_right else "left"};padding:6px 10px;'
            f'border-bottom:0.5px solid #EDEBE3;font-size:13px">{cell}</td>'
            for cell in row
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        '<table style="border-collapse:collapse;width:100%;table-layout:fixed">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _check_value(name: str, value: object, threshold: object) -> str:
    if name.startswith("样本覆盖"):
        return "已覆盖" if value else "未覆盖"
    if "率" in name or "稳定性" in name:
        return f"{_pct(value)} / 阈值 {_pct(threshold)}"
    return f"{_fmt(value, 3)} / 阈值 {_fmt(threshold, 3)}"


def _gate_block(gates: dict[str, object]) -> str:
    rows = []
    for gate in gates.get("gates", []):
        checks = "<br>".join(
            f'<span style="color:{"#0F6E56" if c["passed"] else "#A32D2D"}">'
            f'{"✓" if c["passed"] else "✗" if c["passed"] is False else "○"}</span> '
            f'{html.escape(c["name"])} '
            f'<span style="color:#888780">{_check_value(c["name"], c["value"], c["threshold"])}</span>'
            + (f'<br><span style="color:#888780;font-size:11px">{html.escape(c["note"])}</span>' if c.get("note") else "")
            for c in gate["checks"]
        )
        rows.append((
            f'{gate["gate"]}. {html.escape(gate["name"])}',
            _badge(gate["status_label"]),
            checks,
        ))
    return _table(("关卡", "结论", "检查项（实测 / 门槛）"), rows, align_right=False)


def render_html(report: dict[str, object]) -> str:
    summary = report.get("summary", {}) or {}
    config = report.get("config", {}) or {}
    quality = report.get("data_quality", {}) or {}
    sample = report.get("sample", {}) or {}
    counts = summary.get("counts", {}) or {}

    factor_cards: list[str] = []
    for item in report.get("factors", []):
        if item.get("gates") is None:
            factor_cards.append(
                f'<section style="margin:24px 0;padding:16px;border:0.5px solid #F09595;'
                f'border-radius:12px;background:#FCEBEB">'
                f'<strong>{html.escape(str(item.get("name")))}</strong>'
                f'<p style="font-size:13px;margin:6px 0 0">计算失败：{html.escape(str(item.get("error")))}</p></section>'
            )
            continue
        ic = item.get("ic", {})
        layered = item.get("layered", {})
        cost = item.get("cost", {})
        nav = item.get("nav", {}) or {}
        metric_rows = [
            ("方向调整后 IC 均值", _fmt(ic.get("mean_adj"))),
            ("ICIR", _fmt(ic.get("icir_adj"), 2)),
            ("IC 同号率", _pct(ic.get("hit_rate_adj"))),
            ("IC t 值", _fmt(ic.get("t_adj"), 2)),
            ("分层单调性", _fmt(layered.get("monotone_adj"), 2)),
            ("多空 t 值", _fmt(layered.get("spread_t"), 2)),
            ("期间换手率", _pct(cost.get("turnover"))),
            ("扣费后多头超额", _pct(cost.get("net_long_excess"), 2)),
        ]
        decay = item.get("decay") or {}
        decay_text = "、".join(f"{h} 日 {_fmt(v, 4)}" for h, v in decay.items()) if decay else "未检验"
        oos_block = item.get("oos") or {}
        oos_ratio = None
        if oos_block:
            base = item.get("ic", {}).get("mean_adj")
            oos_value = oos_block.get("ic", {}).get("mean_adj")
            if base and oos_value is not None:
                try:
                    oos_ratio = abs(float(oos_value) / float(base))
                except (TypeError, ValueError, ZeroDivisionError):
                    oos_ratio = None
        factor_cards.append(
            f'<section style="margin:28px 0;padding:18px 20px;border:0.5px solid #D3D1C7;border-radius:12px;background:#FFFFFF">'
            f'<div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap">'
            f'<h3 style="font-size:15px;font-weight:500;margin:0">{html.escape(str(item.get("name")))}</h3>'
            f'<span style="font-size:12px;color:#888780">{html.escape(str(item.get("factor_id")))} · '
            f'{html.escape(str(item.get("category")))} · 方向 {item.get("direction")}</span>'
            f'<span style="margin-left:auto">{_badge(str(item["gates"]["verdict"]))}</span></div>'
            f'<p style="font-size:13px;color:#5F5E5A;margin:8px 0 14px">假设：{html.escape(str(item.get("hypothesis")))}</p>'
            f'<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px">'
            + "".join(
                f'<div style="background:#F8F7F2;border-radius:8px;padding:10px">'
                f'<div style="font-size:11px;color:#5F5E5A">{html.escape(label)}</div>'
                f'<div style="font-size:16px;font-weight:500;margin-top:2px">{value}</div></div>'
                for label, value in metric_rows
            )
            + "</div>"
            f'<div style="font-size:12px;color:#888780;margin-bottom:6px">逐期 RankIC（红正绿负）</div>'
            + _svg_ic(ic.get("series", []))
            + f'<div style="font-size:12px;color:#888780;margin:12px 0 2px">分层累计净值（Q1 最低分位 → Q{len(nav) or 5} 最高分位）</div>'
            + _nav_legend(nav)
            + _svg_lines(layered.get("periods", []), nav)
            + f'<div style="font-size:12px;color:#5F5E5A;margin:8px 0 12px">IC 衰减：{html.escape(decay_text)}'
            f'　样本外 IC 保持率：{_pct(oos_ratio)}</div>'
            + _gate_block(item["gates"])
            + "</section>"
        )

    combo = report.get("combination")
    combo_html = ""
    if combo:
        weights = combo.get("weights", {}) or {}
        dropped = combo.get("dropped", []) or []
        corr = combo.get("correlation", {}) or {}
        combo_rows = [
            (html.escape(str(k)), _pct(v)) for k, v in weights.items()
        ]
        corr_html = ""
        if corr:
            columns = list(corr.keys())
            corr_html = _table(
                ("因子",) + tuple(columns),
                [
                    (html.escape(str(row)),)
                    + tuple(_fmt(corr[row].get(col), 2) for col in columns)
                    for row in columns
                ],
            )
        combo_html = (
            f'<section style="margin:32px 0;padding:18px 20px;border:0.5px solid #D3D1C7;'
            f'border-radius:12px;background:#F8F7F2">'
            f'<div style="display:flex;align-items:baseline;gap:10px">'
            f'<h2 style="font-size:15px;font-weight:500;margin:0">多因子合成</h2>'
            f'<span style="font-size:12px;color:#888780">方法 {html.escape(str(combo.get("method")))} · '
            f'相关性阈值 {combo.get("max_abs_corr")}</span>'
            f'<span style="margin-left:auto">{_badge(str(combo["gates"]["verdict"]))}</span></div>'
            f'<p style="font-size:13px;color:#5F5E5A;margin:8px 0 14px">合成因子 IC '
            f'{_fmt((combo.get("metrics") or {}).get("ic", {}).get("mean"))}、ICIR '
            f'{_fmt((combo.get("metrics") or {}).get("ic", {}).get("icir"), 2)}。</p>'
            + _table(("入选因子", "权重"), combo_rows)
            + ('<div style="margin-top:14px"></div>' + corr_html if corr_html else "")
            + (
                '<p style="font-size:12px;color:#854F0B;margin:12px 0 0">因相关性过高被剔除：'
                + "；".join(f'{html.escape(str(d["factor_id"]))}（{html.escape(str(d["reason"]))}）' for d in dropped)
                + "</p>"
                if dropped else ""
            )
            + "</section>"
        )

    warning_items = "".join(
        f'<li style="margin:4px 0">{html.escape(str(text))}</li>' for text in report.get("warnings", [])
    )
    rejected = quality.get("codes_rejected") or {}
    rejected_html = ""
    if rejected:
        rejected_html = (
            '<p style="font-size:12px;color:#A32D2D;margin:8px 0 0">剔除标的：'
            + "；".join(f"{html.escape(str(k))}（{html.escape(str(v))}）" for k, v in list(rejected.items())[:12])
            + ("…" if len(rejected) > 12 else "")
            + "</p>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>因子研究报告 · {html.escape(str(report.get("study_id")))}</title>
<style>
body{{margin:0;background:#F8F7F2;color:#2C2C2A;font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;line-height:1.6}}
.wrap{{max-width:760px;margin:0 auto;padding:32px 24px 64px}}
.muted{{color:#5F5E5A;font-size:13px}}
</style></head><body><div class="wrap">
<p style="font-size:12px;color:#888780;margin:0 0 6px">{html.escape(BRAND_NAME)} · 因子研究引擎（{html.escape(str(report.get("schema")))}）</p>
<h1 style="font-size:20px;font-weight:500;margin:0 0 6px">{html.escape(str(report.get("universe")))}</h1>
<p class="muted" style="margin:0 0 4px">样本区间 {html.escape(str(sample.get("date_start")))} 至 {html.escape(str(sample.get("date_end")))}
 · 截面 {sample.get("rebalance_dates")} 期（样本内 {sample.get("in_sample_periods")} / 样本外 {sample.get("out_sample_periods")}）
 · 平均 {sample.get("avg_names") and round(float(sample["avg_names"]))} 只标的 · 持有期 {config.get("horizon")} 日</p>
<p class="muted" style="margin:0">研究编号 {html.escape(str(report.get("study_id")))} · 生成时间 {html.escape(str(report.get("generated_at")))}
 · 输入指纹 {html.escape(str(report.get("fingerprint")))}</p>

<section style="margin:24px 0;padding:18px 20px;border-left:3px solid #185FA5;background:#FFFFFF">
<h2 style="font-size:15px;font-weight:500;margin:0 0 8px">结论</h2>
<p style="font-size:14px;margin:0">{html.escape(str(summary.get("headline")))}</p>
<p class="muted" style="margin:8px 0 0">通过 {counts.get("通过", 0)} · 待补证 {counts.get("待补证", 0)} · 淘汰 {counts.get("淘汰", 0)}
 · 计算失败 {counts.get("计算失败", 0)}</p>
</section>

<section style="margin:24px 0">
<h2 style="font-size:15px;font-weight:500;margin:0 0 8px">口径与数据质量</h2>
{_table(("项目", "取值"), [
  ("标的数（可用 / 请求）", f'{quality.get("codes_used", "—")} / {quality.get("codes_requested", "—")}'),
  ("行情根数 中位 / 最小", f'{quality.get("bar_count_median", "—")} / {quality.get("bar_count_min", "—")}'),
  ("预处理", html.escape(str(report.get("preprocess", {}).get("去极值", "—"))) + " · "
   + html.escape(str(report.get("preprocess", {}).get("标准化", "—"))) + " · "
   + html.escape(str(report.get("preprocess", {}).get("中性化", "—")))),
  ("前向收益口径", f'T+{config.get("entry_lag")} 收盘建仓，持有 {config.get("horizon")} 个交易日'),
  ("成本假设", f'双边 {config.get("round_trip_cost_bps")} bp，按 {config.get("cost_multiple")} 倍压力测试'),
  ("合成方法", html.escape(str(config.get("combine_method")) )+ "，相关性阈值 " + str(config.get("max_abs_corr"))),
], align_right=False)}
{rejected_html}
</section>

<h2 style="font-size:15px;font-weight:500;margin:28px 0 4px">单因子检验</h2>
{''.join(factor_cards)}
{combo_html}

<section style="margin:28px 0;padding:16px 20px;border:0.5px solid #FAC775;border-radius:12px;background:#FAEEDA">
<h2 style="font-size:14px;font-weight:500;margin:0 0 6px">已知限制</h2>
<ul style="margin:0;padding-left:18px;font-size:13px">{warning_items}</ul>
</section>

<p style="font-size:12px;color:#888780;margin-top:24px">{html.escape(str(report.get("disclaimer")))}
 · 本页所有数值均由本地研究引擎计算，可用 report.json 复核。</p>
</div></body></html>"""


def dumps(report: dict[str, object]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, default=str)
