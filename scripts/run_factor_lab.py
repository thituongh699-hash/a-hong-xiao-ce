"""命令行跑一轮因子研究，输出 JSON 与可离线打开的 HTML 报告。

示例::

    python scripts/run_factor_lab.py --demo
    python scripts/run_factor_lab.py --codes 600519,000001,000002 --start 2021-01-01 --end 2025-12-31 --factors vol_20,mom_20

只做研究输出，不改变项目的交易配置，也不触发任何买卖。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.factor.dataset import load_bars, synthetic_bars  # noqa: E402
from src.factor.panel import build_panel  # noqa: E402
from src.factor.pipeline import StudyConfig, run_study  # noqa: E402
from src.factor.report import dumps, render_html  # noqa: E402
from src.factor.spec import list_factors  # noqa: E402
from src.utils.brand import BRAND_NAME  # noqa: E402

REPORT_ROOT = ROOT / "reports" / "factor-studies"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{BRAND_NAME} · 因子研究引擎")
    parser.add_argument("--codes", default="", help="逗号分隔的标的代码，留空且未加 --demo 时报错")
    parser.add_argument("--demo", action="store_true", help="使用合成面板验证链路，数值不可作为有效性证据")
    parser.add_argument("--source", default="tdx", help="数据源，与项目 DataFactory 一致")
    parser.add_argument("--start", default=None, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--factors", default="", help="逗号分隔的因子 id，留空则跑全部内置因子")
    parser.add_argument("--horizon", type=int, default=20, help="持有期（交易日），同时作为截面取样步长")
    parser.add_argument("--quantiles", type=int, default=5, help="分层组数")
    parser.add_argument("--min-names", type=int, default=20, help="单个截面最少标的数")
    parser.add_argument("--oos-ratio", type=float, default=0.3, help="样本外区间占比")
    parser.add_argument("--industry", default="", help="行业分类 JSON 文件：{代码: 行业}")
    parser.add_argument("--survivorship-complete", action="store_true",
                        help="声明股票池已覆盖退市与历史 ST（无历史成分股时不要打开）")
    parser.add_argument("--universe", default="未命名股票池", help="股票池名称，写入报告")
    parser.add_argument("--out", default=None, help="输出目录，默认 reports/factor-studies/<研究编号>")
    parser.add_argument("--quiet", action="store_true", help="只输出结论行")
    return parser.parse_args(argv)


def _load_industry(path: str) -> dict[str, str] | None:
    if not path:
        return None
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    if not args.demo and not codes:
        print("需要 --codes 或 --demo；演示结果不能当作因子有效性证据。", file=sys.stderr)
        return 2

    if args.demo:
        bars = synthetic_bars()
    else:
        if not args.start or not args.end:
            print("非演示模式必须提供 --start 与 --end。", file=sys.stderr)
            return 2
        bars, errors = load_bars(codes, args.start, args.end, source=args.source)
        if errors:
            print(f"取数失败 {len(errors)} 只：{list(errors.items())[:5]}", file=sys.stderr)
        if not bars:
            print("没有取到任何行情，终止。", file=sys.stderr)
            return 2

    factor_ids = tuple(f.strip() for f in args.factors.split(",") if f.strip()) or tuple(
        spec.factor_id for spec in list_factors()
    )
    config = StudyConfig(
        factor_ids=factor_ids,
        horizon=args.horizon,
        quantiles=args.quantiles,
        min_names=args.min_names,
        start=args.start,
        end=args.end,
        universe=args.universe if not args.demo else "合成面板（演示）",
        survivorship_complete=True if args.demo else args.survivorship_complete,
        industry=_load_industry(args.industry),
        oos_ratio=args.oos_ratio,
        demo=args.demo,
    )

    report = run_study(build_panel(bars), config)
    out_dir = Path(args.out) if args.out else REPORT_ROOT / str(report["study_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(dumps(report), encoding="utf-8")
    (out_dir / "index.html").write_text(render_html(report), encoding="utf-8")

    if not args.quiet:
        header = f"{'因子':<14}{'结论':<8}{'IC(adj)':>10}{'ICIR':>8}{'单调性':>9}{'换手':>8}{'扣费后超额':>12}"
        print(header)
        print("-" * len(header))
        for item in report["factors"]:
            if item.get("gates") is None:
                print(f"{item['factor_id']:<14}{'计算失败':<8}  {item.get('error')}")
                continue
            cost = item.get("cost", {})
            print(
                f"{item['factor_id']:<14}{item['gates']['verdict']:<8}"
                f"{item['ic']['mean_adj']:>10.4f}{item['ic']['icir_adj']:>8.2f}"
                f"{item['layered']['monotone_adj']:>9.2f}"
                f"{(cost.get('turnover') or 0) * 100:>7.0f}%"
                f"{(cost.get('net_long_excess') or 0) * 100:>11.2f}%"
            )
        combo = report.get("combination")
        if combo:
            print("-" * len(header))
            print(f"{'合成组合':<14}{combo['gates']['verdict']:<8}"
                  f"{combo['metrics']['ic']['mean']:>10.4f}{combo['metrics']['ic']['icir']:>8.2f}")
        print()
        print(report["summary"]["headline"])
        for warning in report["warnings"]:
            print(f"  · {warning}")

    print(f"\n报告: {out_dir / 'index.html'}")
    print(f"数据: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
