import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.factor import api as factor_api


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(factor_api, "STUDIES_ROOT", tmp_path / "factor-studies")
    app = FastAPI()
    app.include_router(factor_api.router)
    return TestClient(app)


def test_catalog_exposes_factors_and_thresholds(client):
    payload = client.get("/api/factor/catalog").json()
    ids = {item["factor_id"] for item in payload["factors"]}
    assert "vol_20" in ids
    assert payload["gate_defaults"]["IC 绝对值下限"] == pytest.approx(0.03)
    assert "中性化" in payload["preprocess_defaults"]


def test_research_page_is_served(client):
    response = client.get("/factor")
    assert response.status_code == 200
    assert "因子研究" in response.text
    assert "/api/factor/study" in response.text


def test_study_requires_codes_outside_demo_mode(client):
    response = client.post("/api/factor/study", json={"factor_ids": ["vol_20"]})
    assert response.status_code == 400
    assert "codes" in response.json()["detail"]


def test_study_rejects_too_many_codes(client):
    response = client.post(
        "/api/factor/study",
        json={"codes": [str(i) for i in range(factor_api.MAX_CODES + 1)],
              "start": "2024-01-01", "end": "2024-12-31"},
    )
    assert response.status_code == 400


def test_report_rejects_path_traversal(client):
    response = client.get("/api/factor/study/..%2F..%2Fetc/report")
    assert response.status_code in {400, 404}


def test_demo_study_runs_and_returns_report(client):
    response = client.post(
        "/api/factor/study",
        json={"demo": True, "factor_ids": ["vol_20"], "horizon": 20, "persist": False},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["summary"]["counts"]
    assert body["report"]["schema"] == "factor-study/v1"
    assert any("合成数据" in warning for warning in body["warnings"])


def test_persisted_study_can_be_listed_and_reopened(client, tmp_path):
    created = client.post(
        "/api/factor/study",
        json={"demo": True, "factor_ids": ["vol_20"], "horizon": 20, "persist": True},
    )
    assert created.status_code == 200, created.text
    study_id = created.json()["study_id"]

    listing = client.get("/api/factor/studies").json()["studies"]
    assert any(item["study_id"] == study_id for item in listing)

    detail = client.get(f"/api/factor/study/{study_id}")
    assert detail.status_code == 200
    assert detail.json()["study_id"] == study_id

    page = client.get(f"/api/factor/study/{study_id}/report")
    assert page.status_code == 200
    assert "因子研究报告" in page.text


def test_missing_study_returns_404(client):
    assert client.get("/api/factor/study/fa-20260101-000000-abcdef").status_code == 404
