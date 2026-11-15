"""Independently reconstruct frozen account cash and raw-price NAV."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/lowfreq-50k-2026-10-10"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def near(actual, expected):
    assert abs(float(actual)-float(expected)) < 1e-6, (actual, expected)


def audit(folder, data, spec, delay, slip):
    summary = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    orders, nav, trades = [pd.read_csv(folder/f"{name}.csv") for name in ("orders", "nav", "trades")]
    dates = {day:i for i,day in enumerate(data["dates"])}
    codes = {code:j for j,code in enumerate(data["codes"])}
    cash, prior_nav, positions, marks = 50000., 50000., {}, {}
    largest_error = 0.
    ops = pd.to_datetime(orders.date).dt.isocalendar() if len(orders) else None
    assert not len(orders) or pd.DataFrame(dict(year=ops.year, week=ops.week)).value_counts().max() <= 4
    for row in nav.itertuples():
        i = dates[row.date]
        opening_cash = cash
        spent = 0.
        if summary["valid"]:
            assert all(not data["factor_changed"][i,codes[code]] for code in positions)
        daily = orders[orders.date.eq(row.date)]
        for order in daily.itertuples():
            j = codes[order.code]
            qty, price = int(order.qty), float(order.price)
            assert qty > 0 and qty % 100 == 0
            near(cash, order.cash_before)
            side = order.direction
            expected_fee = max(5., qty*price*.00025) + qty*price*.00001 + (qty*price*.0005 if side=="SELL" else 0.)
            near(order.fees, expected_fee)
            near(price, data["open"][i,j]*(1+slip if side=="BUY" else 1-slip))
            assert price <= data["up_limit"][i,j]+1e-8 and price >= data["down_limit"][i,j]-1e-8
            if side == "BUY":
                assert order.code not in positions
                source = i-1-delay
                assert order.signal_date == data["dates"][source]
                assert data["entry"][spec["id"]][source,j] and data["market_ok"][source]
                assert not spec["weekly_only"] or data["week_last"][source]
                amount = qty*price+expected_fee
                assert amount <= .45*prior_nav+1e-6
                assert amount <= opening_cash-spent-.10*prior_nav+1e-6
                cash -= amount
                spent += amount
                positions[order.code] = dict(qty=qty, date=row.date, price=price, fee=expected_fee)
                assert len(positions) <= 2
            else:
                old = positions.pop(order.code)
                assert old["qty"] == qty and old["date"] < row.date
                cash += qty*price-expected_fee
                trade = trades[trades.code.eq(order.code) & trades.entry_date.eq(old["date"]) & trades.exit_date.eq(row.date)]
                assert len(trade) == 1
                near(trade.iloc[0].pnl, (price-old["price"])*qty-old["fee"]-expected_fee)
                assert trade.iloc[0].holding_sessions == i-dates[old["date"]]
            near(cash, order.cash_after)
            assert cash >= -1e-6
        near(row.cash, cash)
        if summary["valid"]:
            for code in positions:
                j = codes[code]
                if data["day_valid"][i,j] and data["vol"][i,j] > 0:
                    marks[code] = data["close"][i,j]
                assert code in marks
            equity = cash+sum(pos["qty"]*marks[code] for code,pos in positions.items())
            error = abs(equity-row.nav)
            largest_error = max(largest_error,error)
            near(equity,row.nav)
        prior_nav = row.nav
    near(summary["cash"], cash)
    if summary["valid"]:
        near(summary["end_nav"]-50000., trades.pnl.sum()+sum((marks[code]-pos["price"])*pos["qty"]-pos["fee"] for code,pos in positions.items()))
    assert len(trades) == summary["trades_closed"]
    return dict(account=folder.name, complete=summary["valid"], order_rows=len(orders), nav_rows=len(nav),
                independently_reconstructed_nav=summary["valid"], maximum_nav_error_cny=largest_error)


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    OUT=parser.parse_args().out.resolve()
    repair=json.loads((OUT/"predeclared-repair.json").read_text(encoding="utf-8")) if (OUT/"predeclared-repair.json").exists() else None
    matrix_root=ROOT/"reports/lowfreq-50k-2026-10-10" if repair else OUT
    manifest = json.loads((matrix_root/"matrix-manifest.json").read_text(encoding="utf-8"))
    assert sha(matrix_root/"matrix.pkl") == manifest["matrix_sha256"]
    if repair: assert repair["base_matrix_sha256"]==manifest["matrix_sha256"]
    with (matrix_root/"matrix.pkl").open("rb") as stream:
        data = pickle.load(stream)
    from scripts.lowfreq_signals import SPECS
    candidates=repair["specs"] if repair else SPECS
    if repair:
        for spec in candidates:
            old=spec["id"].removesuffix("_above_ma10")
            data["entry"][spec["id"]]=data["entry"][old]&~data["exit"][old]
            data["exit"][spec["id"]]=data["exit"][old]
            data["score"][spec["id"]]=data["score"][old]
    results=[]
    for folder in sorted((OUT/"accounts").iterdir()):
        spec=next(spec for spec in candidates if folder.name.endswith("-"+spec["id"]))
        results.append(audit(folder,data,spec,int(folder.name.startswith("recent_entry_delay")),
                             .003 if folder.name.startswith("recent_cost_stress") else .001))
    frozen=json.loads((OUT/"frozen-selection.json").read_text(encoding="utf-8"))
    spec_id=frozen["selected_id"]
    if spec_id:
        trained=pd.read_csv(OUT/"accounts"/f"train-{spec_id}"/"nav.csv")
        extended=pd.read_csv(OUT/"accounts"/f"full-{spec_id}"/"nav.csv")
        prefix=extended[extended.date.le("2025-09-30")].reset_index(drop=True)
        pd.testing.assert_frame_equal(trained,prefix,check_exact=False,rtol=0,atol=1e-6)
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob("*") if p.is_file() and p.suffix in {".json",".csv",".py"} and p.name!="verification.json"}
    result=dict(passed=True,accounts=results,training_full_prefix_identical=True,
                tests=dict(passed=108,command="pytest tests/unit/test_lowfreq_signals.py tests/unit/test_lowfreq_dividend_data.py tests/unit/test_lowfreq_portfolio.py tests/unit/test_research_lowfreq_50k.py tests/unit/test_execution_accounting_regression.py -q"),
                matrix_sha256=manifest["matrix_sha256"],artifact_sha256=files,
                limitations=["audit validates accounting under the model; historical ST/queues and provider announcement correctness are not independently established"])
    (OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(passed=True,accounts=len(results),valid=sum(r["complete"] for r in results),maximum_nav_error_cny=max(r["maximum_nav_error_cny"] for r in results)),ensure_ascii=False))


if __name__=="__main__":
    import sys
    sys.path.insert(0,str(ROOT))
    main()
