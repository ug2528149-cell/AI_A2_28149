
import sys
from pathlib import Path

# Make the project root importable when pytest runs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_DATA = {
    "rainfall_mm": 1000,
    "avg_temp_c": 18,
    "soil_moisture_pct": 50,
    "fertilizer_kg": 100,
    "field_size_ha": 2,
    "season": "A",
    "cooperative_support": 1,
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200


def test_valid_prediction():
    response = client.post("/predict", json=VALID_DATA)
    assert response.status_code == 200
    assert "verification_code" in response.json()
    assert "predicted_yield_tons" in response.json()


def test_missing_required_field():
    data = VALID_DATA.copy()
    data.pop("rainfall_mm")
    response = client.post("/predict", json=data)
    assert response.status_code in (400, 422)


def test_wrong_input_type():
    data = VALID_DATA.copy()
    data["rainfall_mm"] = "not-a-number"
    response = client.post("/predict", json=data)
    assert response.status_code in (400, 422)


def test_out_of_range_input():
    data = VALID_DATA.copy()
    data["rainfall_mm"] = 99999
    response = client.post("/predict", json=data)
    assert response.status_code in (400, 422)


def test_prediction_is_repeatable():
    first = client.post("/predict", json=VALID_DATA)
    second = client.post("/predict", json=VALID_DATA)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
