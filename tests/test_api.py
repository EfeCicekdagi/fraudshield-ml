import os
import pytest
from fastapi.testclient import TestClient
from fraudshield.api.app import create_app

@pytest.fixture(scope="module")
def client():
    # Set fake API key
    os.environ["FRAUDSHIELD_API_KEY"] = "test-secret-key"
    app = create_app()
    with TestClient(app) as client:
        yield client

def test_health_live(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"

def test_health_ready(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

def test_model_info_requires_auth(client):
    response = client.get("/api/v1/model-info")
    # Missing API key
    assert response.status_code == 403

def test_model_info(client):
    response = client.get("/api/v1/model-info", headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert data["calibration_method"] == "Isotonic"

def test_predict_single_forbidden_fields(client):
    req = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 100,
        "oldbalanceOrg": 1000,
        "orig_account_type": "C",
        "dest_account_type": "C",
        "isFraud": 1 # FORBIDDEN
    }
    response = client.post("/api/v1/predict", json=req, headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 422
    assert "isFraud" in response.json()["error"]["details"][0]

def test_predict_single_negative_amount(client):
    req = {
        "step": 1,
        "type": "PAYMENT",
        "amount": -100, # INVALID
        "oldbalanceOrg": 1000,
        "orig_account_type": "C",
        "dest_account_type": "C"
    }
    response = client.post("/api/v1/predict", json=req, headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 422
    assert "greater than or equal to 0" in response.json()["error"]["details"][0]

def test_predict_single_success(client):
    req = {
        "step": 12,
        "type": "TRANSFER",
        "amount": 50000,
        "oldbalanceOrg": 50000,
        "orig_account_type": "C",
        "dest_account_type": "C"
    }
    response = client.post("/api/v1/predict", json=req, headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 200
    data = response.json()
    assert "fraud_score" in data
    assert "calibrated_probability" in data
    assert data["explanation"]["enabled"] is False

def test_explain_single(client):
    req = {
        "step": 12,
        "type": "TRANSFER",
        "amount": 50000,
        "oldbalanceOrg": 50000,
        "orig_account_type": "C",
        "dest_account_type": "C"
    }
    response = client.post("/api/v1/explain", json=req, headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 200
    data = response.json()
    assert data["explanation"]["enabled"] is True
    assert "convergence_delta" in data["explanation"]
    assert len(data["top_contributors"]) > 0

def test_predict_batch_limit(client):
    req = {
        "transactions": [
            {
                "step": 1,
                "type": "PAYMENT",
                "amount": 100,
                "oldbalanceOrg": 1000,
                "orig_account_type": "C",
                "dest_account_type": "C"
            } for _ in range(1001)
        ]
    }
    response = client.post("/api/v1/predict/batch", json=req, headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"

def test_predict_batch_success(client):
    req = {
        "transactions": [
            {
                "step": 1,
                "type": "PAYMENT",
                "amount": 100,
                "oldbalanceOrg": 1000,
                "orig_account_type": "C",
                "dest_account_type": "C"
            } for _ in range(2)
        ]
    }
    response = client.post("/api/v1/predict/batch", json=req, headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert "fraud_score" in data[0]

def test_csv_upload(client):
    csv_content = (
        "step,type,amount,oldbalanceOrg,orig_account_type,dest_account_type\n"
        "1,PAYMENT,100,1000,C,C\n"
        "2,TRANSFER,500,5000,C,C\n"
    )
    
    files = {"file": ("test.csv", csv_content, "text/csv")}
    response = client.post("/api/v1/predict/file", files=files, headers={"X-API-Key": "test-secret-key"})
    assert response.status_code == 200
    assert len(response.json()) == 2
