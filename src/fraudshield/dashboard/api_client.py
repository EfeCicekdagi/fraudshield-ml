import requests
import json
from fraudshield.dashboard.config import config
import streamlit as st

class APIClient:
    def __init__(self):
        self.base_url = config.api_url.rstrip('/')
        self.session = requests.Session()
        if config.api_key:
            self.session.headers.update({"X-API-Key": config.api_key})
            
    def _handle_response(self, response):
        if response.status_code >= 400:
            try:
                error_data = response.json()
                msg = error_data.get("error", {}).get("message", response.text)
            except:
                msg = response.text
            raise Exception(f"API Error ({response.status_code}): {msg}")
        return response.json()

    def get_health_ready(self):
        try:
            # Note: health endpoints are typically outside /api/v1 prefix, 
            # so we might need to adjust URL if base_url is /api/v1
            # Assuming base_url is http://127.0.0.1:8000/api/v1
            # We strip /api/v1 to get to /health/ready
            host = self.base_url.replace("/api/v1", "")
            res = self.session.get(f"{host}/health/ready", timeout=5)
            return res.status_code == 200
        except:
            return False

    def get_dashboard_summary(self):
        res = self.session.get(f"{self.base_url}/dashboard/summary", timeout=10)
        return self._handle_response(res)

    def get_model_info(self):
        res = self.session.get(f"{self.base_url}/model-info", timeout=10)
        return self._handle_response(res)

    def predict_single(self, payload: dict, explain: bool = False):
        params = {"explain": "true"} if explain else {}
        url = f"{self.base_url}/explain" if explain else f"{self.base_url}/predict"
        res = self.session.post(url, json=payload, params=params, timeout=30)
        return self._handle_response(res)

    def predict_batch(self, transactions: list):
        payload = {"transactions": transactions}
        res = self.session.post(f"{self.base_url}/predict/batch", json=payload, timeout=60)
        return self._handle_response(res)

    def create_case(self, payload: dict):
        res = self.session.post(f"{self.base_url}/cases", json=payload, timeout=10)
        return self._handle_response(res)
        
    def get_cases(self, skip=0, limit=50, status=None, risk_level=None):
        params = {"skip": skip, "limit": limit}
        if status: params["status"] = status
        if risk_level: params["risk_level"] = risk_level
        res = self.session.get(f"{self.base_url}/cases", params=params, timeout=10)
        return self._handle_response(res)

    def get_case(self, case_id: str):
        res = self.session.get(f"{self.base_url}/cases/{case_id}", timeout=10)
        return self._handle_response(res)
        
    def update_case(self, case_id: str, payload: dict):
        res = self.session.patch(f"{self.base_url}/cases/{case_id}", json=payload, timeout=10)
        return self._handle_response(res)

    def get_case_history(self, case_id: str):
        res = self.session.get(f"{self.base_url}/cases/{case_id}/history", timeout=10)
        return self._handle_response(res)

# Thread-safe singleton for Streamlit cache/session
@st.cache_resource
def get_api_client():
    return APIClient()
