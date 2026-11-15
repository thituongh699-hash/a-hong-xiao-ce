"""Persisted reports must keep point content under outgoing canonical keys."""
import asyncio

import pytest
import server


@pytest.mark.parametrize('location', ['cached', 'persisted'])
@pytest.mark.parametrize('version', [0, server.AI_REVIEW_SUMMARY_SCHEMA_VERSION])
def test_outgoing_report_normalizes_legacy_points(monkeypatch, location, version):
    old = {'title': 'previous review', '硅基节点分析': [{'reason': 'buy reason'}],
           '流码节点分析': [{'reason': 'sell reason'}]}
    report = {'report_id': 'old'}
    cache = {}
    if location == 'cached':
        cache = {'old__summary': old, 'old__summary_v': version}
    else:
        report.update(ai_review_summary=old, ai_review_summary_version=version)
    monkeypatch.setattr(server, 'report_ai_review_cache', cache)
    payload = server._build_ai_review_payload(report, 'old')
    assert payload['ai_review_summary']['buy_points'][0]['reason'] == 'buy reason'
    assert payload['ai_review_summary']['sell_points'][0]['reason'] == 'sell reason'
    assert '硅基节点分析' not in payload['ai_review_summary']
    assert 'buy_points' not in old  # No persisted object mutation.


def test_nested_legacy_summary_points_are_normalized():
    summary = server._normalize_ai_review_summary({'summary': {'硅基节点': ['buy'], '流码节点': ['sell']}})
    assert summary['buy_points'][0]['reason'] == 'buy'
    assert summary['sell_points'][0]['reason'] == 'sell'


@pytest.mark.parametrize('buy,sell', [('硅基节点', '流码节点'), ('买点', '卖点')])
def test_old_markdown_point_sections_are_preserved(buy, sell):
    text = f'## 1) 核心结论\nreview\n## 2) 关键问题\n- issue\n## 3) 基于交易明细的{buy}分析\n- buy reason\n## 4) 基于交易明细的{sell}分析\n- sell reason\n## 5) 参数优化建议\n- x: 1\n## 6) 下一轮实验方案\n- trial'
    summary = server._parse_ai_review_summary_from_markdown(text)
    assert summary['buy_points'][0]['reason'] == 'buy reason'
    assert summary['sell_points'][0]['reason'] == 'sell reason'


@pytest.mark.parametrize('cached', [True, False])
def test_review_post_cache_hit_returns_canonical_points(monkeypatch, cached):
    old = {'title': 'review', '硅基节点': ['buy'], '流码节点': ['sell']}
    report = {'report_id': 'old', 'ai_review_text': 'review', 'ai_review_version': server.AI_REVIEW_SCHEMA_VERSION,
              'ai_review_summary': old, 'ai_review_summary_version': server.AI_REVIEW_SUMMARY_SCHEMA_VERSION}
    cache = {'old': 'review', 'old__v': server.AI_REVIEW_SCHEMA_VERSION, 'old__summary': old} if cached else {}
    monkeypatch.setattr(server, 'load_report_history', lambda: None)
    monkeypatch.setattr(server, 'report_history', [report])
    monkeypatch.setattr(server, 'report_ai_review_cache', cache)
    payload = asyncio.run(server.api_report_ai_review('old'))
    assert payload['status'] == 'success'
    assert payload['analysis_summary']['buy_points'][0]['reason'] == 'buy'
    assert payload['analysis_summary']['sell_points'][0]['reason'] == 'sell'


def test_missing_review_stays_empty_instead_of_fabricating_a_cached_summary(monkeypatch):
    monkeypatch.setattr(server, 'report_ai_review_cache', {})
    payload = server._build_ai_review_payload({'report_id': 'no-review'}, 'no-review')
    assert payload['ai_review_text'] == ''
    assert payload['ai_review_summary_raw'] == {}
    assert payload['ai_review_summary_localized'] == {}
    assert payload['ai_review_summary_version'] == 0
