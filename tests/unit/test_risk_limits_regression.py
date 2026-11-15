from datetime import datetime

from src.core.risk_manager import RiskManager
from src.services.risk_audit_log import RiskAuditLog


def limits(key, default=None):
    return {
        "risk_control.max_stop_loss_pct": 1.0,
        "risk_control.max_pos_per_stock": 0.10,
        "risk_control.max_total_pos": 0.50,
        "risk_control.max_daily_loss_pct": 0.02,
    }.get(key, default)


def signal():
    return dict(strategy_id="s", code="000001.SZ", direction="BUY",
                qty=900, price=10, dt=datetime(2026, 6, 1, 10))


def test_single_stock_limit_counts_existing_holding(monkeypatch):
    monkeypatch.setattr("src.core.risk_manager.get_value", limits)
    risk = RiskManager(RiskAuditLog())
    approved, reason = risk.check_signal(
        signal(), 100000, {"000001.SZ": {"market_value": 9000}}, 0,
    )
    assert not approved
    assert "单票仓位" in reason


def test_daily_loss_gate_and_consecutive_loss_day_reset(monkeypatch):
    monkeypatch.setattr("src.core.risk_manager.get_value", limits)
    risk = RiskManager(RiskAuditLog())
    order = signal()
    risk.start_session("s", "2026-06-01")
    approved, reason = risk.check_signal(order, 100000, {}, -3000)
    assert not approved and "单日亏损" in reason

    for _ in range(3):
        risk.update_loss_count("s", True)
    risk.start_session("s", datetime(2026, 6, 1, 11))
    approved, reason = risk.check_signal(order, 100000, {}, 0)
    assert not approved and "连续亏损" in reason

    risk.start_session("s", "2026-06-02")
    approved, _ = risk.check_signal(order, 100000, {}, 0)
    assert approved
    assert risk.consecutive_losses["s"] == 0
