import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app


def test_home():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200