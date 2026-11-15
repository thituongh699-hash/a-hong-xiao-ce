from src.core.risk_manager import RiskManager
from src.services.risk_audit_log import RiskAuditLog


def test_allow_buy_when_drawdown_exceeds_limit(monkeypatch):
    def fake_get_value(key, default=None):
        table = {
            "risk_control.max_stop_loss_pct": 1.0,
            "risk_control.max_pos_per_stock": 1.0,
            "risk_control.max_total_pos": 1.0,
            "risk_control.max_daily_loss_pct": 1.0,
            "risk_control.max_drawdown_pct": 0.03
        }
        return table.get(key, default)

    monkeypatch.setattr("src.core.risk_manager.get_value", fake_get_value)
    risk_audit_log = RiskAuditLog()
    risk_review = RiskManager(risk_audit_log)
    signal = {
        "strategy_id": "T1",
        "code": "000001.SZ",
        "direction": "BUY",
        "price": 100.0,
        "qty": 100,
        "dt": "2026-03-22 10:00:00"
    }
    approved, reason = risk_review.check_signal(signal, 1_000_000.0, {}, 0.0, current_drawdown=0.05)
    assert approved is True
    assert reason == "批准执行"
    assert risk_audit_log.rejections == []


def test_allow_sell_even_if_drawdown_exceeds_limit(monkeypatch):
    def fake_get_value(key, default=None):
        table = {
            "risk_control.max_stop_loss_pct": 1.0,
            "risk_control.max_pos_per_stock": 1.0,
            "risk_control.max_total_pos": 1.0,
            "risk_control.max_daily_loss_pct": 1.0,
            "risk_control.max_drawdown_pct": 0.03
        }
        return table.get(key, default)

    monkeypatch.setattr("src.core.risk_manager.get_value", fake_get_value)
    risk_audit_log = RiskAuditLog()
    risk_review = RiskManager(risk_audit_log)
    signal = {
        "strategy_id": "T1",
        "code": "000001.SZ",
        "direction": "SELL",
        "price": 100.0,
        "qty": 100,
        "dt": "2026-03-22 10:00:00"
    }
    approved, _ = risk_review.check_signal(signal, 1_000_000.0, {"000001.SZ": {"market_value": 10000}}, 0.0, current_drawdown=0.05)
    assert approved is True
