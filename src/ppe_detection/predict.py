"""Interface d'inference unifiee : image, dossier, video, webcam, flux RTSP.

Usage :

    python -m ppe_detection.predict --weights artifacts/models/best.pt \\
        --source chemin_source --conf 0.25 --iou 0.45 --device auto \\
        --save --save-txt --save-json

Toutes les sources passent par le meme detecteur : le modele n'est charge
qu'une fois et les resultats sont retournes dans une structure homogene.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .compliance import evaluate_compliance, summarise_compliance
from .config import ComplianceConfig, InferenceConfig, load_compliance_config, load_inference_config
from .utils import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ensure_dir,
    get_logger,
    is_image,
    is_video,
    project_root,
    resolve_device,
    resolve_path,
    setup_logging,
    write_json,
)
from .visualization import draw_compliance, draw_detections

LOGGER = get_logger(__name__)

STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "tcp://", "udp://")


def _detect_quantize_support() -> bool:
    """True si l'Ultralytics installe expose ``quantize`` (8.4+) plutot que ``half``."""
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        return "quantize" in DEFAULT_CFG_DICT
    except ImportError:  # pragma: no cover - Ultralytics absent
        return False


_SUPPORTS_QUANTIZE = _detect_quantize_support()


class PredictionError(RuntimeError):
    """Erreur bloquante durant l'inference."""


@dataclass
class ImagePrediction:
    """Resultat structure de l'inference sur une image."""

    source: str
    width: int
    height: int
    detections: list[dict[str, Any]] = field(default_factory=list)
    timing_ms: dict[str, float] = field(default_factory=dict)
    compliance: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable du resultat."""
        payload: dict[str, Any] = {
            "source": self.source,
            "image": {"width": self.width, "height": self.height},
            "detections": self.detections,
            "timing_ms": {k: round(v, 2) for k, v in self.timing_ms.items()},
        }
        if self.compliance:
            payload["compliance"] = self.compliance
            payload["compliance_summary"] = summarise_compliance(self.compliance)
        return payload


def is_stream_source(source: str) -> bool:
    """True si la source est une URL de flux (RTSP/HTTP...)."""
    return str(source).lower().startswith(STREAM_PREFIXES)


def is_webcam_source(source: str) -> bool:
    """True si la source designe une camera locale (index numerique)."""
    text = str(source).strip()
    return text.isdigit() or text.lower() in {"webcam", "camera"}


def classify_source(source: str) -> str:
    """Determine la nature d'une source d'inference.

    Returns:
        ``"webcam"``, ``"stream"``, ``"video"``, ``"image"`` ou ``"directory"``.

    Raises:
        PredictionError: Si le chemin n'existe pas ou si le type est inconnu.
    """
    if is_webcam_source(source):
        return "webcam"
    if is_stream_source(source):
        return "stream"
    path = Path(source)
    if not path.exists():
        raise PredictionError(
            f"Source introuvable : {source}\n"
            f"Fournissez un fichier image, un fichier video, un dossier, "
            f"un index de webcam (0) ou une URL rtsp://."
        )
    if path.is_dir():
        return "directory"
    if is_video(path):
        return "video"
    if is_image(path):
        return "image"
    raise PredictionError(
        f"Type de fichier non supporte : {path.suffix!r}\n"
        f"Images acceptees : {', '.join(sorted(IMAGE_EXTENSIONS))}\n"
        f"Videos acceptees : {', '.join(sorted(VIDEO_EXTENSIONS))}"
    )


class PPEDetector:
    """Detecteur EPI : le modele n'est charge qu'une seule fois.

    Cette classe est reutilisee par la CLI, l'API REST et l'interface Streamlit
    afin de garantir un comportement identique quel que soit le point d'entree.
    """

    def __init__(
        self,
        config: InferenceConfig,
        *,
        compliance: ComplianceConfig | None = None,
    ) -> None:
        """Charge les poids et prepare le detecteur.

        Args:
            config: Parametres d'inference (poids, seuils, device).
            compliance: Regles de conformite optionnelles.

        Raises:
            PredictionError: Si les poids sont introuvables ou illisibles.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise PredictionError("Ultralytics n'est pas installe : pip install ultralytics") from exc

        self.config = config
        self.compliance_config = compliance
        # Estimateur de pose optionnel, attache via attach_pose_estimator().
        self.pose_estimator: Any = None
        weights_path = resolve_path(config.weights)
        if not weights_path.is_file():
            raise PredictionError(
                f"Poids introuvables : {weights_path}\n"
                f"Entrainez d'abord un modele :\n"
                f"  python -m ppe_detection.train --config configs/train.yaml\n"
                f"ou indiquez d'autres poids via --weights."
            )

        self.device = resolve_device(config.device)
        LOGGER.info("Chargement des poids %s sur %s", weights_path.name, self.device)
        try:
            self.model = YOLO(str(weights_path))
        except Exception as exc:  # noqa: BLE001 - torch leve des types varies
            raise PredictionError(f"Chargement du modele impossible ({weights_path}) : {exc}") from exc

        names = getattr(self.model, "names", {}) or {}
        self.class_names: dict[int, str] = {int(k): str(v) for k, v in names.items()}
        self.weights_path = weights_path

    # ------------------------------------------------------------------ #
    def _precision_kwargs(self) -> dict[str, Any]:
        """Traduit la demande FP16 dans le vocabulaire de l'Ultralytics installe.

        Ultralytics 8.4 a remplace ``half=True`` par ``quantize=16``; les
        versions anterieures ne connaissent que ``half``. On n'emet le parametre
        que lorsque la demi-precision est reellement demandee et utilisable
        (elle n'a pas de sens sur CPU).
        """
        if not (self.config.half and self.device != "cpu"):
            return {}
        return {"quantize": 16} if _SUPPORTS_QUANTIZE else {"half": True}

    def _predict_raw(self, source: Any, **overrides: Any) -> list[Any]:
        """Appelle Ultralytics avec les parametres d'inference resolus."""
        kwargs: dict[str, Any] = {
            "conf": self.config.conf,
            "iou": self.config.iou,
            "imgsz": self.config.imgsz,
            "device": self.device,
            "max_det": self.config.max_det,
            "agnostic_nms": self.config.agnostic_nms,
            "verbose": False,
        }
        kwargs.update(self._precision_kwargs())
        kwargs.update(overrides)
        # Les annotations d'Ultralytics declarent une union Iterator/list ;
        # en mode non-stream, predict() retourne toujours une liste de Results.
        return list(self.model.predict(source=source, **kwargs))  # type: ignore[arg-type]

    def _track_raw(self, source: Any, **overrides: Any) -> list[Any]:
        """Comme :meth:`_predict_raw`, mais avec suivi d'objets persistant."""
        kwargs: dict[str, Any] = {
            "conf": self.config.conf,
            "iou": self.config.iou,
            "imgsz": self.config.imgsz,
            "device": self.device,
            "max_det": self.config.max_det,
            "verbose": False,
            "persist": True,
            "tracker": self.config.tracker,
        }
        kwargs.update(self._precision_kwargs())
        kwargs.update(overrides)
        try:
            return list(self.model.track(source=source, **kwargs))  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - Ultralytics leve des types varies
            raise PredictionError(
                f"Suivi d'objets impossible ({self.config.tracker}) : {exc}\n"
                f"Trackers disponibles : bytetrack.yaml, botsort.yaml."
            ) from exc

    def _build_detections(self, result: Any) -> list[dict[str, Any]]:
        """Convertit un resultat Ultralytics en detections structurees et filtrees."""
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        # Present uniquement en mode suivi ; None sinon.
        raw_ids = getattr(boxes, "id", None)
        track_ids = raw_ids.cpu().numpy().astype(int) if raw_ids is not None else None

        for index, (box, confidence, class_id) in enumerate(
            zip(xyxy, confidences, class_ids, strict=True)
        ):
            class_name = self.class_names.get(int(class_id), f"unknown_{int(class_id)}")
            # Les seuils par classe ne peuvent que durcir le filtrage global.
            if float(confidence) < self.config.threshold_for(class_name):
                continue
            detection: dict[str, Any] = {
                "class_id": int(class_id),
                "class_name": class_name,
                "confidence": round(float(confidence), 4),
                "bbox_xyxy": [round(float(v), 2) for v in box],
            }
            if track_ids is not None and index < len(track_ids):
                detection["track_id"] = int(track_ids[index])
            detections.append(detection)
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections

    def _apply_compliance(
        self,
        detections: list[dict[str, Any]],
        image_size: tuple[int, int] | None = None,
        image: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Applique la couche de conformite si elle est activee.

        Lorsqu'un estimateur de pose est attache, les detections de personnes
        sont d'abord enrichies de regions issues des points cles du corps, qui
        remplacent le decoupage par fractions.
        """
        if self.compliance_config is None or not self.compliance_config.enabled:
            return []
        if self.pose_estimator is not None and image is not None:
            try:
                self.pose_estimator.annotate_detections(
                    image, detections, self.compliance_config.person_class
                )
            except Exception as exc:  # noqa: BLE001 - la pose ne doit jamais casser l'inference
                LOGGER.warning(
                    "Estimation de pose ignoree pour cette image (%s). "
                    "Repli sur le decoupage par fractions.",
                    exc,
                )
        return evaluate_compliance(detections, self.compliance_config, image_size=image_size)

    # ------------------------------------------------------------------ #
    def track_array(self, image: np.ndarray, *, source_name: str = "array") -> ImagePrediction:
        """Infere avec suivi d'objets : chaque detection recoit un ``track_id`` stable.

        Utilise le tracker configure d'Ultralytics (ByteTrack par defaut) avec
        ``persist=True``, ce qui conserve l'etat entre les appels successifs.
        A n'employer que sur une sequence continue : appeler cette methode sur
        des images sans rapport ferait deriver les identifiants.

        Args:
            image: Tableau NumPy BGR.
            source_name: Libelle de la source.

        Returns:
            Le resultat structure, detections enrichies de ``track_id``.
        """
        return self.predict_array(image, source_name=source_name, track=True)

    def predict_array(
        self, image: np.ndarray, *, source_name: str = "array", track: bool = False
    ) -> ImagePrediction:
        """Infere sur une image deja chargee en memoire (BGR).

        Args:
            image: Tableau NumPy BGR.
            source_name: Libelle de la source pour la tracabilite.
            track: Active le suivi d'objets (identifiants persistants).

        Returns:
            Le resultat structure.
        """
        started = time.perf_counter()
        results = self._track_raw(image) if track else self._predict_raw(image)
        elapsed = (time.perf_counter() - started) * 1000.0
        if not results:
            height, width = image.shape[:2]
            return ImagePrediction(source=source_name, width=width, height=height)

        result = results[0]
        detections = self._build_detections(result)
        height, width = image.shape[:2]
        prediction = ImagePrediction(
            source=source_name,
            width=width,
            height=height,
            detections=detections,
            timing_ms=_extract_speed(result, elapsed),
        )
        prediction.compliance = self._apply_compliance(
            detections, image_size=(width, height), image=image
        )
        return prediction

    def attach_pose_estimator(
        self,
        weights: str = "",
        *,
        device: str | None = None,
        min_keypoint_score: float = 0.5,
    ) -> None:
        """Active l'association EPI/personne fondee sur les points cles du corps.

        Le decoupage par fractions suppose une personne debout vue de face ;
        les points cles donnent la position reelle de la tete, du torse et des
        pieds quelle que soit la posture. Le repli reste automatique lorsque la
        pose est indisponible pour une personne donnee.

        Args:
            weights: Poids du modele de pose (defaut : ``yolo26n-pose.pt``).
            device: Peripherique; par defaut celui du detecteur principal.
            min_keypoint_score: Confiance minimale d'un point cle.

        Raises:
            PredictionError: Si le modele de pose ne peut pas etre charge.
        """
        from .pose import DEFAULT_POSE_MODEL, PoseError, PoseEstimator

        try:
            self.pose_estimator = PoseEstimator(
                weights or DEFAULT_POSE_MODEL,
                device=device or str(self.device),
                imgsz=self.config.imgsz,
                min_keypoint_score=min_keypoint_score,
            )
        except PoseError as exc:
            raise PredictionError(str(exc)) from exc

    def predict_image(self, path: str | Path) -> ImagePrediction:
        """Infere sur un fichier image.

        Raises:
            PredictionError: Si l'image est illisible.
        """
        import cv2

        image_path = Path(path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise PredictionError(
                f"Image illisible : {image_path}\n"
                f"Le fichier est peut-etre corrompu ou dans un format non supporte."
            )
        return self.predict_array(image, source_name=str(image_path))

    def predict_images(self, paths: Sequence[Path]) -> list[ImagePrediction]:
        """Infere sur une liste d'images, en poursuivant malgre les erreurs unitaires."""
        predictions: list[ImagePrediction] = []
        for index, path in enumerate(paths, start=1):
            try:
                predictions.append(self.predict_image(path))
            except PredictionError as exc:
                LOGGER.warning("Image ignoree (%s) : %s", path.name, exc)
                continue
            if index % 50 == 0 or index == len(paths):
                LOGGER.info("Progression : %d/%d images", index, len(paths))
        return predictions

    # ------------------------------------------------------------------ #
    def annotate(self, image: np.ndarray, prediction: ImagePrediction) -> np.ndarray:
        """Dessine detections et, le cas echeant, statuts de conformite."""
        canvas = draw_detections(
            image,
            prediction.detections,
            show_labels=self.config.show_labels,
            show_conf=self.config.show_conf,
            line_width=self.config.line_width,
            font_scale=self.config.font_scale,
        )
        if prediction.compliance:
            canvas = draw_compliance(
                canvas,
                prediction.compliance,
                line_width=self.config.line_width,
                font_scale=self.config.font_scale,
                copy=False,
            )
        return canvas

    def model_info(self) -> dict[str, Any]:
        """Metadonnees du modele charge (pour l'API et l'interface)."""
        parameters = None
        with contextlib.suppress(AttributeError, TypeError):  # pragma: no cover
            # Ultralytics annote `.model` comme `str | None` alors qu'il s'agit
            # d'un nn.Module une fois les poids charges.
            module = self.model.model
            parameters = int(sum(p.numel() for p in module.parameters()))  # type: ignore[union-attr]
        return {
            "weights": str(self.weights_path),
            "weights_name": self.weights_path.name,
            "weights_size_bytes": self.weights_path.stat().st_size if self.weights_path.is_file() else None,
            "device": self.device,
            "task": getattr(self.model, "task", None),
            "num_classes": len(self.class_names),
            "class_names": [self.class_names[i] for i in sorted(self.class_names)],
            "parameters": parameters,
            "imgsz": self.config.imgsz,
            "conf": self.config.conf,
            "iou": self.config.iou,
            "compliance_enabled": bool(self.compliance_config and self.compliance_config.enabled),
        }


def _extract_speed(result: Any, fallback_ms: float) -> dict[str, float]:
    """Recupere les temps de traitement rapportes par Ultralytics."""
    speed = getattr(result, "speed", None)
    if isinstance(speed, dict) and speed:
        timing = {str(key): float(value) for key, value in speed.items()}
        timing.setdefault("total", sum(timing.values()))
        return timing
    return {"inference": fallback_ms, "total": fallback_ms}


# --------------------------------------------------------------------------- #
# Sorties
# --------------------------------------------------------------------------- #
def save_predictions_json(predictions: Sequence[ImagePrediction], path: Path) -> Path:
    """Ecrit les predictions au format JSON."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_sources": len(predictions),
        "results": [prediction.to_dict() for prediction in predictions],
    }
    return write_json(path, payload)


def save_predictions_csv(predictions: Sequence[ImagePrediction], path: Path) -> Path:
    """Ecrit les detections a plat au format CSV (une ligne par detection)."""
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source", "image_width", "image_height", "class_id", "class_name",
             "confidence", "x1", "y1", "x2", "y2"]
        )
        for prediction in predictions:
            for detection in prediction.detections:
                x1, y1, x2, y2 = detection["bbox_xyxy"]
                writer.writerow(
                    [
                        prediction.source,
                        prediction.width,
                        prediction.height,
                        detection["class_id"],
                        detection["class_name"],
                        detection["confidence"],
                        x1, y1, x2, y2,
                    ]
                )
    return path


def save_yolo_txt(prediction: ImagePrediction, path: Path) -> Path:
    """Ecrit les detections au format YOLO normalise (class cx cy w h conf)."""
    ensure_dir(path.parent)
    lines: list[str] = []
    for detection in prediction.detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        cx = ((x1 + x2) / 2.0) / prediction.width
        cy = ((y1 + y2) / 2.0) / prediction.height
        box_w = (x2 - x1) / prediction.width
        box_h = (y2 - y1) / prediction.height
        lines.append(
            f"{detection['class_id']} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f} "
            f"{detection['confidence']:.4f}"
        )
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return path


def collect_images(directory: Path, *, recursive: bool = False) -> list[Path]:
    """Liste triee des images d'un dossier."""
    iterator: Iterable[Path] = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in iterator if p.is_file() and is_image(p))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments de la commande d'inference."""
    parser = argparse.ArgumentParser(
        prog="python -m ppe_detection.predict",
        description="Inference EPI sur image, dossier, video, webcam ou flux.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default=None, help="Chemin des poids (.pt).")
    parser.add_argument("--source", required=True, help="Image, dossier, video, index webcam (0) ou URL rtsp://.")
    parser.add_argument("--config", default=None, help="configs/inference.yaml.")
    parser.add_argument("--conf", type=float, default=None, help="Seuil de confiance global.")
    parser.add_argument("--iou", type=float, default=None, help="Seuil IoU pour la NMS.")
    parser.add_argument("--imgsz", type=int, default=None, help="Taille d'inference.")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda | 0")
    parser.add_argument(
        "--max-det",
        type=int,
        dest="max_det",
        default=None,
        help="Detections maximales par image.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=None,
        help="Inference en FP16 (GPU uniquement).",
    )
    parser.add_argument("--output", default="artifacts/predictions", help="Repertoire de sortie.")
    parser.add_argument(
        "--name",
        default=None,
        help="Nom du sous-repertoire de sortie (defaut : horodatage).",
    )
    parser.add_argument("--save", action="store_true", help="Sauvegarde les images/videos annotees.")
    parser.add_argument(
        "--save-txt",
        action="store_true",
        dest="save_txt",
        help="Sauvegarde les labels YOLO.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        dest="save_json",
        help="Sauvegarde un rapport JSON.",
    )
    parser.add_argument("--save-csv", action="store_true", dest="save_csv", help="Sauvegarde un rapport CSV.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Parcourt les sous-dossiers d'un repertoire source.",
    )
    parser.add_argument("--hide-labels", action="store_true", help="Masque les noms de classes.")
    parser.add_argument("--hide-conf", action="store_true", help="Masque les scores de confiance.")
    parser.add_argument("--compliance", action="store_true", help="Active la logique de conformite EPI.")
    parser.add_argument(
        "--required-ppe",
        nargs="*",
        default=None,
        help="EPI obligatoires (ex. --required-ppe \"Safety Helmet\" \"Safety Vest\").",
    )
    # Options video / webcam
    parser.add_argument(
        "--pose",
        action="store_true",
        help=(
            "Associe les EPI aux personnes via les points cles du corps plutot que "
            "par decoupage de la boite en fractions. Plus robuste aux postures "
            "accroupies et aux prises de vue en plongee. Necessite un second modele."
        ),
    )
    parser.add_argument(
        "--pose-weights",
        default="",
        help="Poids du modele de pose (defaut : yolo26n-pose.pt, telecharge au besoin).",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help=(
            "Active le suivi d'objets et le lissage temporel des verdicts de "
            "conformite (une alerte par personne, pas par frame)."
        ),
    )
    parser.add_argument(
        "--tracker",
        default=None,
        help="Tracker Ultralytics : bytetrack.yaml (rapide) ou botsort.yaml (robuste aux occlusions).",
    )
    parser.add_argument("--show", action="store_true", help="Affiche une fenetre temps reel (video/webcam).")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        dest="frame_skip",
        help="Nombre de frames sautees entre deux inferences.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        dest="max_frames",
        help="Nombre maximal de frames traitees (0 = illimite).",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        dest="no_save_video",
        help="Ne produit pas de video annotee.",
    )
    parser.add_argument("--log-level", default="INFO", help="Niveau de log.")
    return parser


def _resolve_output_dir(base: str, name: str | None) -> Path:
    """Cree et retourne le repertoire de sortie de l'execution."""
    stamp = name or datetime.now().strftime("predict_%Y%m%d_%H%M%S")
    return ensure_dir(resolve_path(base) / stamp)


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entree CLI. Retourne le code de sortie du processus."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=project_root() / "artifacts" / "logs" / "predict.log")

    config_path = args.config or (project_root() / "configs" / "inference.yaml")
    overrides = {
        "weights": args.weights,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "max_det": args.max_det,
        "half": args.half,
        "tracker": args.tracker,
    }
    if args.hide_labels:
        overrides["show_labels"] = False
    if args.hide_conf:
        overrides["show_conf"] = False

    try:
        inference_config = load_inference_config(config_path, **overrides)
        compliance_config = load_compliance_config(config_path)
        if args.compliance:
            compliance_config.enabled = True
        if args.required_ppe is not None:
            compliance_config.required_ppe = list(args.required_ppe)

        kind = classify_source(args.source)
        LOGGER.info("Source detectee : %s (%s)", args.source, kind)
        detector = PPEDetector(inference_config, compliance=compliance_config)
        if args.pose:
            if not compliance_config.enabled:
                LOGGER.warning(
                    "--pose est sans effet sans --compliance : les points cles ne servent "
                    "qu'a associer les EPI aux personnes."
                )
            else:
                detector.attach_pose_estimator(args.pose_weights)
    except (PredictionError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2

    output_dir = _resolve_output_dir(args.output, args.name)
    LOGGER.info("Sorties : %s", output_dir)

    if kind in {"video", "webcam", "stream"}:
        from .video import run_video_inference

        try:
            summary = run_video_inference(
                detector,
                args.source,
                output_dir=output_dir,
                save_video=args.save and not args.no_save_video,
                show=args.show,
                frame_skip=max(0, args.frame_skip),
                max_frames=max(0, args.max_frames),
                save_json=args.save_json,
                track=args.track,
            )
        except PredictionError as exc:
            LOGGER.error("%s", exc)
            return 2
        LOGGER.info(
            "Video terminee : %d frames traitees, %.1f FPS moyens, %d detections.",
            summary["frames_processed"],
            summary["average_fps"],
            summary["total_detections"],
        )
        tracked = summary.get("tracked_compliance")
        if tracked:
            LOGGER.info(
                "Conformite par personne suivie : %d personne(s) — %d conforme(s), "
                "%d non conforme(s), %d indetermine(s). %d evenement(s) d'alerte, "
                "%d personne(s) en alerte a la fin.",
                tracked["tracked_persons"],
                tracked["compliant"],
                tracked["non_compliant"],
                tracked["indeterminate"],
                summary.get("n_alerts", 0),
                tracked["persons_currently_alerted"],
            )
        return 0

    # Images / dossier
    if kind == "image":
        paths = [Path(args.source)]
    else:
        paths = collect_images(Path(args.source), recursive=args.recursive)
        if not paths:
            LOGGER.error("Aucune image trouvee dans %s", args.source)
            return 2
        LOGGER.info("%d image(s) a traiter.", len(paths))

    predictions = detector.predict_images(paths)
    if not predictions:
        LOGGER.error("Aucune image n'a pu etre traitee.")
        return 1

    if args.save or args.save_txt:
        import cv2

        for prediction in predictions:
            source_path = Path(prediction.source)
            if args.save:
                image = cv2.imread(str(source_path))
                if image is not None:
                    annotated = detector.annotate(image, prediction)
                    ensure_dir(output_dir / "images")
                    cv2.imwrite(str(output_dir / "images" / source_path.name), annotated)
            if args.save_txt:
                save_yolo_txt(prediction, output_dir / "labels" / f"{source_path.stem}.txt")

    if args.save_json:
        LOGGER.info("JSON : %s", save_predictions_json(predictions, output_dir / "predictions.json"))
    if args.save_csv:
        LOGGER.info("CSV : %s", save_predictions_csv(predictions, output_dir / "predictions.csv"))

    total_detections = sum(len(p.detections) for p in predictions)
    mean_inference = (
        sum(p.timing_ms.get("inference", 0.0) for p in predictions) / len(predictions)
        if predictions
        else 0.0
    )
    LOGGER.info(
        "Termine : %d image(s), %d detection(s), %.1f ms d'inference en moyenne.",
        len(predictions),
        total_detections,
        mean_inference,
    )

    if compliance_config.enabled:
        all_persons = [person for p in predictions for person in p.compliance]
        summary = summarise_compliance(all_persons)
        LOGGER.info(
            "Conformite (heuristique) : %d personne(s) — %d conforme(s), %d non conforme(s), "
            "%d indetermine(s).",
            summary["persons_detected"],
            summary["compliant"],
            summary["non_compliant"],
            summary["indeterminate"],
        )
        if summary["indeterminate"]:
            LOGGER.info(
                "Les cas indetermines correspondent a des personnes tronquees par le bord du "
                "cadre ou trop petites : aucune conclusion n'en est tiree."
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
