"""Tests de l'API REST avec le client de test FastAPI.

Les tests couvrant l'inference reelle sont ignores automatiquement si aucun
poids n'est disponible dans ``artifacts/models``.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from ppe_detection.taxonomy import BASE_CLASSES

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


@pytest.fixture
def jpeg_bytes() -> bytes:
    """Une petite image JPEG valide en memoire."""
    import cv2

    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


@pytest.fixture
def client(real_weights: Path, monkeypatch: pytest.MonkeyPatch):
    """Client de test avec le modele charge depuis des poids reels."""
    monkeypatch.setenv("PPE_API_WEIGHTS", str(real_weights))
    monkeypatch.setenv("PPE_API_DEVICE", "cpu")
    from ppe_detection import api

    with TestClient(api.app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_health_reports_ok(client) -> None:  # noqa: ANN001
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["version"]


@pytest.mark.slow
def test_model_info(client) -> None:  # noqa: ANN001
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    # Le schema etendu conserve les identifiants des 7 classes d'origine et
    # ajoute des classes negatives : le nombre exact depend des poids charges.
    # L'invariant testable est la presence des classes de base.
    assert set(payload["class_names"]) >= set(BASE_CLASSES)
    assert payload["num_classes"] == len(payload["class_names"])
    assert "Safety Helmet" in payload["class_names"]
    assert payload["device"] == "cpu"
    assert payload["parameters"] > 0


def test_health_degrades_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sans poids valides, le service demarre mais se declare 'degraded'."""
    monkeypatch.setenv("PPE_API_WEIGHTS", "artifacts/models/absolutely_missing.pt")
    from ppe_detection import api

    with TestClient(api.app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "degraded"
        assert payload["model_loaded"] is False
        assert payload["detail"]

        # Les routes d'inference doivent repondre 503, pas planter.
        predict = test_client.post(
            "/predict/image", files={"file": ("x.jpg", b"not-an-image", "image/jpeg")}
        )
        assert predict.status_code == 503
        assert "PPE_API_WEIGHTS" in predict.json()["detail"]

        info = test_client.get("/model-info")
        assert info.status_code == 503


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_predict_image_returns_structured_payload(client, jpeg_bytes: bytes) -> None:  # noqa: ANN001
    response = client.post(
        "/predict/image", files={"file": ("scene.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")}
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["request_id"]
    assert payload["image"] == {"width": 160, "height": 120}
    assert isinstance(payload["detections"], list)
    assert "total_request" in payload["timing_ms"]
    for detection in payload["detections"]:
        assert set(detection) >= {"class_id", "class_name", "confidence", "bbox_xyxy"}
        assert 0.0 <= detection["confidence"] <= 1.0


@pytest.mark.slow
def test_predict_image_sanitises_filename(client, jpeg_bytes: bytes) -> None:  # noqa: ANN001
    """Un nom de fichier hostile ne doit jamais ressortir tel quel."""
    response = client.post(
        "/predict/image",
        files={"file": ("../../etc/passwd.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    filename = response.json()["filename"]
    assert ".." not in filename
    assert "/" not in filename and "\\" not in filename


@pytest.mark.slow
def test_predict_rejects_wrong_content_type(client, jpeg_bytes: bytes) -> None:  # noqa: ANN001
    response = client.post(
        "/predict/image", files={"file": ("x.txt", io.BytesIO(jpeg_bytes), "text/plain")}
    )
    assert response.status_code == 415
    assert "non supporte" in response.json()["detail"]


@pytest.mark.slow
def test_predict_rejects_corrupt_image(client) -> None:  # noqa: ANN001
    response = client.post(
        "/predict/image", files={"file": ("x.jpg", io.BytesIO(b"definitely not a jpeg"), "image/jpeg")}
    )
    assert response.status_code == 400
    assert "illisible" in response.json()["detail"].lower()


@pytest.mark.slow
def test_predict_rejects_empty_file(client) -> None:  # noqa: ANN001
    response = client.post(
        "/predict/image", files={"file": ("x.jpg", io.BytesIO(b""), "image/jpeg")}
    )
    assert response.status_code == 400


@pytest.mark.slow
def test_predict_rejects_oversized_file(client, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    from ppe_detection import api

    monkeypatch.setattr(api, "MAX_FILE_MB", 0.001)
    payload = b"\xff\xd8\xff" + b"0" * 40000
    response = client.post(
        "/predict/image", files={"file": ("big.jpg", io.BytesIO(payload), "image/jpeg")}
    )
    assert response.status_code == 413
    assert "volumineux" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Lot
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_predict_batch(client, jpeg_bytes: bytes) -> None:  # noqa: ANN001
    files = [
        ("files", (f"img{i}.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")) for i in range(3)
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    payload = response.json()
    assert payload["n_images"] == 3
    assert payload["n_failed"] == 0
    assert len(payload["results"]) == 3
    assert payload["timing_ms"]["mean_per_image"] >= 0


@pytest.mark.slow
def test_predict_batch_reports_partial_failures(client, jpeg_bytes: bytes) -> None:  # noqa: ANN001
    """Un fichier invalide ne doit pas faire echouer tout le lot."""
    files = [
        ("files", ("good.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")),
        ("files", ("bad.txt", io.BytesIO(b"nope"), "text/plain")),
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    payload = response.json()
    assert payload["n_images"] == 1
    assert payload["n_failed"] == 1
    assert payload["failures"][0]["filename"] == "bad.txt"


@pytest.mark.slow
def test_predict_batch_all_invalid_returns_400(client) -> None:  # noqa: ANN001
    files = [("files", ("bad.txt", io.BytesIO(b"nope"), "text/plain"))]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 400


@pytest.mark.slow
def test_predict_batch_enforces_limit(client, jpeg_bytes: bytes, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    from ppe_detection import api

    monkeypatch.setattr(api, "MAX_BATCH_FILES", 2)
    files = [
        ("files", (f"img{i}.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")) for i in range(3)
    ]
    response = client.post("/predict/batch", files=files)
    assert response.status_code == 413


# --------------------------------------------------------------------------- #
# Documentation
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_openapi_schema_is_exposed(client) -> None:  # noqa: ANN001
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    for route in ("/health", "/model-info", "/predict/image", "/predict/batch"):
        assert route in schema["paths"]
