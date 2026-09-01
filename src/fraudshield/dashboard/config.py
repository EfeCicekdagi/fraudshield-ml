import os

class DashboardConfig:
    def __init__(self):
        self.api_url = os.environ.get("FRAUDSHIELD_API_URL", "http://127.0.0.1:8000/api/v1")
        self.api_key = os.environ.get("FRAUDSHIELD_API_KEY", "")

config = DashboardConfig()
