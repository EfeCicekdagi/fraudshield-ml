import os
import pytest
from fastapi.testclient import TestClient
from fraudshield.api.app import create_app
import pandas as pd
from fraudshield.dashboard.pages.batch_scoring import sanitize_csv_export

@pytest.fixture(scope="module")
def client():
    os.environ["FRAUDSHIELD_API_KEY"] = "test-secret-key"
    app = create_app()
    with TestClient(app) as c:
        yield c

def test_case_creation_and_idempotency(client):
    req_data = {
        "inference_id": "test-inf-123",
        "calibrated_probability": 0.85,
        "risk_level": "HIGH",
        "reason_codes": ["LARGE_AMOUNT"],
        "model_version": "v1.0"
    }
    
    # Create first time
    res = client.post("/api/v1/cases", json=req_data, headers={"X-API-Key": "test-secret-key"})
    assert res.status_code == 200
    case_id = res.json()["case_id"]
    
    # Create again with same inference_id (Idempotency)
    res2 = client.post("/api/v1/cases", json=req_data, headers={"X-API-Key": "test-secret-key"})
    assert res2.status_code == 200
    assert res2.json()["case_id"] == case_id

def test_case_optimistic_locking_and_transitions(client):
    # Create a new case
    req_data = {
        "inference_id": "test-inf-999",
        "calibrated_probability": 0.90,
        "risk_level": "CRITICAL",
        "reason_codes": [],
        "model_version": "v1.0"
    }
    res = client.post("/api/v1/cases", json=req_data, headers={"X-API-Key": "test-secret-key"})
    case = res.json()
    case_id = case["case_id"]
    version = case["version"]
    
    # Valid transition: NEW -> UNDER_REVIEW
    update_res = client.patch(f"/api/v1/cases/{case_id}", json={
        "version": version,
        "status": "UNDER_REVIEW",
        "actor": "tester"
    }, headers={"X-API-Key": "test-secret-key"})
    assert update_res.status_code == 200
    new_version = update_res.json()["version"]
    assert new_version == version + 1
    
    # Invalid transition: UNDER_REVIEW -> CLOSED (not allowed directly)
    bad_res = client.patch(f"/api/v1/cases/{case_id}", json={
        "version": new_version,
        "status": "CLOSED"
    }, headers={"X-API-Key": "test-secret-key"})
    assert bad_res.status_code == 400
    
    # Optimistic Locking conflict
    # Attempt to update with an old version
    stale_res = client.patch(f"/api/v1/cases/{case_id}", json={
        "version": version, # old version
        "status": "CONFIRMED_FRAUD"
    }, headers={"X-API-Key": "test-secret-key"})
    assert stale_res.status_code == 409

def test_csv_sanitization():
    df = pd.DataFrame({
        "col1": ["normal", "=cmd|' /C calc'!A0", "+123", "-456", "@sum(A1)"],
        "col2": [1, 2, 3, 4, 5]
    })
    safe_df = sanitize_csv_export(df.copy())
    
    assert safe_df["col1"].iloc[0] == "normal"
    assert safe_df["col1"].iloc[1].startswith("'=")
    assert safe_df["col1"].iloc[2].startswith("'+")
    assert safe_df["col1"].iloc[3].startswith("'-")
    assert safe_df["col1"].iloc[4].startswith("'@")
    
    # Numerics should be unaffected
    assert safe_df["col2"].iloc[0] == 1
