"""API REST FastAPI pour la detection d'EPI.

Lancement :

    uvicorn ppe_detection.api:app --host 127.0.0.1 --port 8000

Points d'entree :

* ``GET  /health``         — etat du service et disponibilite du modele
* ``GET  /model-info``     — metadonnees du modele charge
* ``POST /predict/image``  — inference sur une image
* ``POST /predict/batch``  — inference sur plusieurs images

Choix de conception lies a la securite :

* les fichiers recus sont traites **entierement en memoire** : aucun contenu
  fourni par un client n'est ecrit sur disque, ce qui elimine tout risque
  d'ecriture arbitraire via un nom de fichier hostile ;
* le nom de fichier renvoye dans la reponse est neutralise ;
* les journaux ne contiennent ni le contenu des images ni le nom brut soumis ;
* le type et la taille des fichiers sont valides avant tout decodage.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .config import ComplianceConfig, InferenceConfig, load_compliance_config, load_inference_config
from .predict import PPEDetector, PredictionError
from .utils import get_logger, project_root, safe_filename, setup_logging

LOGGER = get_logger(__name__)

try:
    from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "L'API necessite FastAPI. Installez les dependances :\n"
        "  pip install fastapi 'uvicorn[standard]' python-multipart"
    ) from exc


# --------------------------------------------------------------------------- #
# Configuration par variables d'environnement
# --------------------------------------------------------------------------- #
def _env_float(name: str, default: float) -> float:
    """Lit une variable d'environnement numerique, avec repli sur la valeur par defaut."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        LOGGER.warning("%s=%r invalide — valeur par defaut %s utilisee.", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    """Lit une variable d'environnement entiere, avec repli sur la valeur par defaut."""
    return int(_env_float(name, float(default)))


def _env_bool(name: str, default: bool = False) -> bool:
    """Lit une variable d'environnement booleenne."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "oui"}


# Starlette a renomme HTTP_413_REQUEST_ENTITY_TOO_LARGE en HTTP_413_CONTENT_TOO_LARGE.
# L'ancien nom emet un avertissement de depreciation des qu'on y accede : on ne
# le consulte donc que si le nouveau est absent.
HTTP_413_TOO_LARGE: int = (
    status.HTTP_413_CONTENT_TOO_LARGE
    if hasattr(status, "HTTP_413_CONTENT_TOO_LARGE")
    else getattr(status, "HTTP_413_REQUEST_ENTITY_TOO_LARGE", 413)
)

ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp", "image/tiff"}
)
"""Types MIME acceptes en entree."""

MAX_FILE_MB = _env_float("PPE_API_MAX_FILE_MB", 10.0)
MAX_BATCH_FILES = _env_int("PPE_API_MAX_BATCH", 16)
API_TITLE = os.environ.get("PPE_API_TITLE", "PPE Detection API")


class ServiceState:
    """Etat partage du service : le modele n'est charge qu'une fois au demarrage."""

    def __init__(self) -> None:
        self.detector: PPEDetector | None = None
        self.error: str | None = None
        self.started_at: datetime | None = None
        self.requests_served: int = 0

    def load(self) -> None:
        """Charge le detecteur en memorisant l'erreur eventuelle sans lever d'exception."""
        config_path = project_root() / "configs" / "inference.yaml"
        overrides: dict[str, Any] = {}
        if os.environ.get("PPE_API_WEIGHTS"):
            overrides["weights"] = os.environ["PPE_API_WEIGHTS"]
        if os.environ.get("PPE_API_DEVICE"):
            overrides["device"] = os.environ["PPE_API_DEVICE"]
        overrides["conf"] = _env_float("PPE_API_CONF", 0.25)
        overrides["iou"] = _env_float("PPE_API_IOU", 0.45)

        try:
            inference_config: InferenceConfig = load_inference_config(config_path, **overrides)
            compliance_config: ComplianceConfig = load_compliance_config(config_path)
            if _env_bool("PPE_API_COMPLIANCE", compliance_config.enabled):
                compliance_config.enabled = True
            self.detector = PPEDetector(inference_config, compliance=compliance_config)
            self.error = None
            LOGGER.info("Modele charge : %s", inference_config.weights)
        except (PredictionError, ValueError, OSError) as exc:
            self.detector = None
            self.error = str(exc)
            LOGGER.error("Modele indisponible : %s", exc)
        self.started_at = datetime.now(timezone.utc)


STATE = ServiceState()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Charge le modele au demarrage et le libere a l'arret."""
    setup_logging(os.environ.get("PPE_API_LOG_LEVEL", "INFO"))
    LOGGER.info("Demarrage de l'API — chargement du modele...")
    STATE.load()
    yield
    LOGGER.info("Arret de l'API.")
    STATE.detector = None


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Detection(BaseModel):
    """Un objet detecte."""

    class_id: int = Field(..., description="Identifiant numerique de la classe.")
    class_name: str = Field(..., description="Nom lisible de la classe.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Score de confiance.")
    bbox_xyxy: list[float] = Field(..., description="Boite en pixels [x1, y1, x2, y2].")


class ImageSize(BaseModel):
    """Dimensions de l'image analysee."""

    width: int
    height: int


class PredictionResponse(BaseModel):
    """Reponse d'une inference sur une image."""

    request_id: str
    filename: str | None = None
    image: ImageSize
    detections: list[Detection]
    compliance: list[dict[str, Any]] = Field(default_factory=list)
    compliance_summary: dict[str, Any] | None = None
    timing_ms: dict[str, float]


class BatchPredictionResponse(BaseModel):
    """Reponse d'une inference par lot."""

    request_id: str
    n_images: int
    n_failed: int
    results: list[PredictionResponse]
    failures: list[dict[str, str]] = Field(default_factory=list)
    timing_ms: dict[str, float]


class HealthResponse(BaseModel):
    """Etat de sante du service."""

    status: str = Field(..., description="ok | degraded")
    model_loaded: bool
    detail: str | None = None
    uptime_seconds: float | None = None
    requests_served: int = 0
    version: str


class ModelInfoResponse(BaseModel):
    """Metadonnees du modele charge."""

    weights_name: str
    device: str
    task: str | None
    num_classes: int
    class_names: list[str]
    parameters: int | None
    imgsz: int
    conf: float
    iou: float
    compliance_enabled: bool
    weights_size_bytes: int | None


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
app = FastAPI(
    title=API_TITLE,
    version="1.0.0",
    description=(
        "Detection d'equipements de protection individuelle (EPI) sur images.\n\n"
        "**Avertissement** : le champ `compliance` repose sur une heuristique "
        "geometrique associant les EPI aux personnes detectees. Un statut "
        "`non conforme` doit etre traite comme une alerte a verifier, jamais "
        "comme un constat automatique."
    ),
    lifespan=lifespan,
)


def get_detector() -> PPEDetector:
    """Dependance FastAPI : retourne le detecteur ou renvoie 503.

    Raises:
        HTTPException: 503 si le modele n'a pas pu etre charge.
    """
    if STATE.detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Modele indisponible. "
                f"{STATE.error or 'Aucun poids charge.'} "
                "Definissez PPE_API_WEIGHTS vers un fichier .pt valide et redemarrez le service."
            ),
        )
    return STATE.detector


async def _read_validated_image(file: UploadFile) -> tuple[np.ndarray, str]:
    """Valide et decode une image envoyee par un client.

    Args:
        file: Fichier recu.

    Returns:
        ``(image_bgr, nom_de_fichier_neutralise)``.

    Raises:
        HTTPException: 415 si le type est refuse, 413 si la taille depasse la
            limite, 400 si le contenu n'est pas une image decodable.
    """
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Type de fichier non supporte : {content_type or 'inconnu'}. "
                f"Types acceptes : {', '.join(sorted(ALLOWED_CONTENT_TYPES))}."
            ),
        )

    payload = await file.read()
    size_mb = len(payload) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=HTTP_413_TOO_LARGE,
            detail=f"Fichier trop volumineux ({size_mb:.1f} Mo). Limite : {MAX_FILE_MB} Mo.",
        )
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Fichier vide."
        )

    import cv2

    # Decodage en memoire : le contenu client n'atteint jamais le disque.
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image illisible ou corrompue : le contenu n'a pas pu etre decode.",
        )
    return image, safe_filename(file.filename or "image")


def _to_response(
    prediction: Any, request_id: str, filename: str | None
) -> PredictionResponse:
    """Convertit une prediction interne en reponse API."""
    payload = prediction.to_dict()
    return PredictionResponse(
        request_id=request_id,
        filename=filename,
        image=ImageSize(width=payload["image"]["width"], height=payload["image"]["height"]),
        detections=[Detection(**d) for d in payload["detections"]],
        compliance=payload.get("compliance", []),
        compliance_summary=payload.get("compliance_summary"),
        timing_ms=payload["timing_ms"],
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["service"])
async def health() -> HealthResponse:
    """Etat du service.

    Retourne toujours 200 : ``status`` vaut ``ok`` si le modele est charge,
    ``degraded`` sinon, ce qui permet a une sonde de distinguer un service
    injoignable d'un service demarre sans modele.
    """
    uptime = None
    if STATE.started_at is not None:
        uptime = (datetime.now(timezone.utc) - STATE.started_at).total_seconds()
    from . import __version__

    return HealthResponse(
        status="ok" if STATE.detector is not None else "degraded",
        model_loaded=STATE.detector is not None,
        detail=STATE.error,
        uptime_seconds=round(uptime, 1) if uptime is not None else None,
        requests_served=STATE.requests_served,
        version=__version__,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["service"])
async def model_info(detector: PPEDetector = Depends(get_detector)) -> ModelInfoResponse:
    """Metadonnees du modele actuellement charge."""
    info = detector.model_info()
    return ModelInfoResponse(
        weights_name=info["weights_name"],
        device=info["device"],
        task=info["task"],
        num_classes=info["num_classes"],
        class_names=info["class_names"],
        parameters=info["parameters"],
        imgsz=info["imgsz"],
        conf=info["conf"],
        iou=info["iou"],
        compliance_enabled=info["compliance_enabled"],
        weights_size_bytes=info["weights_size_bytes"],
    )


@app.post("/predict/image", response_model=PredictionResponse, tags=["inference"])
async def predict_image(
    file: UploadFile = File(..., description="Image a analyser (JPEG, PNG, BMP, WebP, TIFF)."),
    detector: PPEDetector = Depends(get_detector),
) -> PredictionResponse:
    """Detecte les EPI sur une image."""
    request_id = str(uuid.uuid4())
    image, filename = await _read_validated_image(file)
    started = time.perf_counter()
    try:
        prediction = detector.predict_array(image, source_name=filename)
    except PredictionError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Echec de l'inference : {exc}"
        ) from exc
    elapsed = (time.perf_counter() - started) * 1000.0
    STATE.requests_served += 1
    LOGGER.info("[%s] image %dx%d, %d detection(s), %.1f ms",
                request_id, prediction.width, prediction.height, len(prediction.detections), elapsed)

    response = _to_response(prediction, request_id, filename)
    response.timing_ms = {**response.timing_ms, "total_request": round(elapsed, 2)}
    return response


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
async def predict_batch(
    files: list[UploadFile] = File(..., description="Images a analyser."),
    detector: PPEDetector = Depends(get_detector),
) -> BatchPredictionResponse:
    """Detecte les EPI sur plusieurs images.

    Une image invalide n'interrompt pas le lot : elle est reportee dans
    ``failures`` avec la raison du rejet.
    """
    request_id = str(uuid.uuid4())
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Aucun fichier fourni."
        )
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=HTTP_413_TOO_LARGE,
            detail=f"Lot trop volumineux ({len(files)} fichiers). Limite : {MAX_BATCH_FILES}.",
        )

    started = time.perf_counter()
    results: list[PredictionResponse] = []
    failures: list[dict[str, str]] = []

    for file in files:
        name = safe_filename(file.filename or "image")
        try:
            image, filename = await _read_validated_image(file)
        except HTTPException as exc:
            failures.append({"filename": name, "error": str(exc.detail), "status": str(exc.status_code)})
            continue
        try:
            prediction = detector.predict_array(image, source_name=filename)
        except PredictionError as exc:
            failures.append({"filename": name, "error": str(exc), "status": "500"})
            continue
        results.append(_to_response(prediction, request_id, filename))

    elapsed = (time.perf_counter() - started) * 1000.0
    STATE.requests_served += len(results)
    LOGGER.info(
        "[%s] lot de %d fichier(s) : %d traite(s), %d echec(s), %.1f ms",
        request_id, len(files), len(results), len(failures), elapsed,
    )

    if not results and failures:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Aucune image du lot n'a pu etre traitee. Premiere erreur : {failures[0]['error']}",
        )

    return BatchPredictionResponse(
        request_id=request_id,
        n_images=len(results),
        n_failed=len(failures),
        results=results,
        failures=failures,
        timing_ms={
            "total_request": round(elapsed, 2),
            "mean_per_image": round(elapsed / len(results), 2) if results else 0.0,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
    """Filet de securite : renvoie une erreur JSON sans divulguer de trace interne."""
    error_id = str(uuid.uuid4())
    LOGGER.exception("[%s] Erreur non geree sur %s", error_id, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Erreur interne du serveur.",
            "error_id": error_id,
        },
    )


def main() -> int:  # pragma: no cover - lance un serveur bloquant
    """Lance le serveur uvicorn (equivalent au script ``run_api.ps1``)."""
    import uvicorn

    uvicorn.run(
        "ppe_detection.api:app",
        host=os.environ.get("PPE_API_HOST", "127.0.0.1"),
        port=_env_int("PPE_API_PORT", 8000),
        reload=_env_bool("PPE_API_RELOAD", False),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
