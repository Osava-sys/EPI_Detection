"""Evaluation independante d'un modele de detection EPI.

Usage :

    python -m ppe_detection.evaluate \\
        --weights artifacts/models/best.pt \\
        --data artifacts/dataset_detection/data.yaml \\
        --split test

Produit metriques globales et par classe, matrice de confusion, vitesses de
traitement, ainsi qu'une analyse d'erreurs (faux positifs, faux negatifs,
confusions entre classes) et la liste des meilleurs/pires exemples.

Rappel methodologique : les hyperparametres et seuils doivent etre choisis sur
le split de **validation**. Le split de test ne sert qu'une fois les choix
arretes, pour estimer la performance finale sans biais.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .annotations import parse_label_text
from .config import ConfigError, load_dataset_config
from .utils import (
    describe_environment,
    ensure_dir,
    get_logger,
    human_bytes,
    markdown_table,
    project_root,
    resolve_device,
    resolve_path,
    setup_logging,
    write_json,
    write_text,
)

LOGGER = get_logger(__name__)

SPLIT_TO_YAML_KEY = {"train": "train", "valid": "val", "val": "val", "test": "test"}


class EvaluationError(RuntimeError):
    """Erreur bloquante durant l'evaluation."""


def _iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Matrice d'IoU entre boites predites et boites de reference (format xyxy)."""
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=float)
    x1 = np.maximum(pred_boxes[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(pred_boxes[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(pred_boxes[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(pred_boxes[:, None, 3], gt_boxes[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_pred = np.clip(pred_boxes[:, 2] - pred_boxes[:, 0], 0, None) * np.clip(
        pred_boxes[:, 3] - pred_boxes[:, 1], 0, None
    )
    area_gt = np.clip(gt_boxes[:, 2] - gt_boxes[:, 0], 0, None) * np.clip(
        gt_boxes[:, 3] - gt_boxes[:, 1], 0, None
    )
    union = area_pred[:, None] + area_gt[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


def _load_ground_truth(
    label_path: Path, width: int, height: int, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Charge un fichier de labels YOLO et le convertit en boites pixel xyxy."""
    if not label_path.is_file():
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    text = label_path.read_text(encoding="utf-8", errors="replace")
    boxes: list[list[float]] = []
    classes: list[int] = []
    for _, parsed in parse_label_text(text, num_classes=num_classes):
        if parsed.is_error or parsed.box is None or parsed.class_id is None:
            continue
        x1, y1, x2, y2 = parsed.box.xyxy
        boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
        classes.append(parsed.class_id)
    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    return np.asarray(boxes, dtype=float), np.asarray(classes, dtype=int)


def analyse_errors(
    detector: Any,
    images_dir: Path,
    labels_dir: Path,
    class_names: Sequence[str],
    *,
    iou_threshold: float = 0.5,
    limit: int = 0,
    max_examples: int = 15,
) -> dict[str, Any]:
    """Analyse les erreurs du modele image par image.

    Apparie chaque prediction a la boite de reference de meilleur IoU (par ordre
    de confiance decroissante) pour classer les detections en vrais positifs,
    faux positifs, faux negatifs et confusions de classe.

    Args:
        detector: :class:`~ppe_detection.predict.PPEDetector` initialise.
        images_dir: Repertoire des images du split.
        labels_dir: Repertoire des labels correspondants.
        class_names: Noms de classes indexes par identifiant.
        iou_threshold: IoU minimal pour considerer un appariement.
        limit: Nombre maximal d'images analysees (0 = toutes).
        max_examples: Nombre d'exemples conserves par categorie.

    Returns:
        Un dictionnaire d'analyse d'erreurs serialisable.
    """
    import cv2

    from .utils import is_image

    image_paths = sorted(p for p in images_dir.iterdir() if p.is_file() and is_image(p))
    if limit:
        image_paths = image_paths[:limit]

    num_classes = len(class_names)
    true_positives = np.zeros(num_classes, dtype=int)
    false_positives = np.zeros(num_classes, dtype=int)
    false_negatives = np.zeros(num_classes, dtype=int)
    confusion = np.zeros((num_classes + 1, num_classes + 1), dtype=int)  # +1 = "fond"
    background_index = num_classes

    per_image: list[dict[str, Any]] = []

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        prediction = detector.predict_array(image, source_name=str(image_path))

        gt_boxes, gt_classes = _load_ground_truth(
            labels_dir / f"{image_path.stem}.txt", width, height, num_classes
        )
        pred_boxes = np.asarray(
            [d["bbox_xyxy"] for d in prediction.detections], dtype=float
        ).reshape(-1, 4)
        pred_classes = np.asarray([d["class_id"] for d in prediction.detections], dtype=int)

        ious = _iou_matrix(pred_boxes, gt_boxes)
        matched_gt: set[int] = set()
        image_tp = image_fp = 0

        for pred_index in range(len(pred_boxes)):
            if gt_boxes.size == 0:
                false_positives[pred_classes[pred_index]] += 1
                confusion[pred_classes[pred_index], background_index] += 1
                image_fp += 1
                continue
            candidates = [
                (ious[pred_index, gt_index], gt_index)
                for gt_index in range(len(gt_boxes))
                if gt_index not in matched_gt and ious[pred_index, gt_index] >= iou_threshold
            ]
            if not candidates:
                false_positives[pred_classes[pred_index]] += 1
                confusion[pred_classes[pred_index], background_index] += 1
                image_fp += 1
                continue
            _, best_gt = max(candidates)
            matched_gt.add(best_gt)
            predicted_class = int(pred_classes[pred_index])
            actual_class = int(gt_classes[best_gt])
            confusion[predicted_class, actual_class] += 1
            if predicted_class == actual_class:
                true_positives[predicted_class] += 1
                image_tp += 1
            else:
                false_positives[predicted_class] += 1
                false_negatives[actual_class] += 1
                image_fp += 1

        for gt_index in range(len(gt_boxes)):
            if gt_index not in matched_gt:
                false_negatives[gt_classes[gt_index]] += 1
                confusion[background_index, gt_classes[gt_index]] += 1

        n_gt = len(gt_boxes)
        image_fn = n_gt - len(matched_gt)
        score = image_tp / max(n_gt + image_fp, 1)
        per_image.append(
            {
                "image": image_path.name,
                "n_ground_truth": n_gt,
                "n_predictions": len(pred_boxes),
                "true_positives": image_tp,
                "false_positives": image_fp,
                "false_negatives": image_fn,
                "score": round(score, 4),
            }
        )
        if index % 100 == 0:
            LOGGER.info("Analyse d'erreurs : %d/%d images", index, len(image_paths))

    scored = [item for item in per_image if item["n_ground_truth"] > 0]
    scored.sort(key=lambda item: (-item["score"], item["image"]))
    best = scored[:max_examples]
    worst = sorted(scored, key=lambda item: (item["score"], item["image"]))[:max_examples]

    confusions: list[dict[str, Any]] = []
    for predicted in range(num_classes):
        for actual in range(num_classes):
            if predicted != actual and confusion[predicted, actual] > 0:
                confusions.append(
                    {
                        "predicted": class_names[predicted],
                        "actual": class_names[actual],
                        "count": int(confusion[predicted, actual]),
                    }
                )
    confusions.sort(key=lambda item: -item["count"])

    per_class: dict[str, Any] = {}
    for class_id, name in enumerate(class_names):
        tp = int(true_positives[class_id])
        fp = int(false_positives[class_id])
        fn = int(false_negatives[class_id])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return {
        "iou_threshold": iou_threshold,
        "conf_threshold": detector.config.conf,
        "images_analysed": len(per_image),
        "per_class": per_class,
        "totals": {
            "true_positives": int(true_positives.sum()),
            "false_positives": int(false_positives.sum()),
            "false_negatives": int(false_negatives.sum()),
        },
        "class_confusions": confusions[:25],
        "best_examples": best,
        "worst_examples": worst,
        "confusion_matrix": {
            "labels": [*class_names, "background"],
            "matrix": confusion.tolist(),
            "orientation": "matrix[predite][reelle]",
        },
    }


def evaluate(
    weights: str | Path,
    data: str | Path,
    *,
    split: str = "test",
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = "auto",
    error_analysis: bool = True,
    error_conf: float = 0.25,
    error_limit: int = 0,
    output_dir: str | Path = "artifacts/reports",
    name: str | None = None,
) -> dict[str, Any]:
    """Evalue un modele sur un split et produit un rapport complet.

    Args:
        weights: Poids ``.pt`` a evaluer.
        data: ``data.yaml`` du dataset.
        split: ``train``, ``valid`` ou ``test``.
        imgsz: Taille d'inference.
        batch: Taille de lot pour la validation.
        conf: Seuil de confiance pour le calcul de la mAP (bas volontairement,
            conformement au protocole COCO).
        iou: Seuil IoU de la NMS durant la validation.
        device: Device d'execution.
        error_analysis: Active l'analyse d'erreurs detaillee.
        error_conf: Seuil de confiance **operationnel** pour l'analyse d'erreurs.
        error_limit: Limite d'images pour l'analyse d'erreurs (0 = toutes).
        output_dir: Repertoire des rapports.
        name: Nom de base des fichiers de rapport.

    Returns:
        Le rapport d'evaluation.

    Raises:
        EvaluationError: Poids/dataset introuvables ou echec de la validation.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise EvaluationError("Ultralytics n'est pas installe : pip install ultralytics") from exc

    weights_path = resolve_path(weights)
    if not weights_path.is_file():
        raise EvaluationError(
            f"Poids introuvables : {weights_path}\n"
            f"Entrainez d'abord un modele ou corrigez --weights."
        )
    data_path = resolve_path(data)
    if not data_path.is_file():
        raise EvaluationError(f"data.yaml introuvable : {data_path}")

    dataset = load_dataset_config(data_path)
    split_key = SPLIT_TO_YAML_KEY.get(split)
    if split_key is None:
        raise EvaluationError(f"Split inconnu : {split!r}. Utilisez train, valid ou test.")

    resolved_device = resolve_device(device)
    LOGGER.info("Evaluation de %s sur le split '%s' (device=%s)", weights_path.name, split, resolved_device)

    model = YOLO(str(weights_path))
    # Sans 'project', Ultralytics ecrit ses courbes dans ./runs/detect/val,
    # hors de l'arborescence artifacts/ du projet.
    val_project = ensure_dir(resolve_path(output_dir).parent / "runs" / "val")
    try:
        metrics = model.val(
            data=str(data_path),
            split=split_key,
            imgsz=imgsz,
            batch=batch,
            conf=conf,
            iou=iou,
            device=resolved_device,
            plots=True,
            verbose=False,
            project=str(val_project),
            name=name or f"evaluation_{split}",
            exist_ok=True,
        )
    except (RuntimeError, MemoryError) as exc:
        message = str(exc).lower()
        if "out of memory" in message:
            raise EvaluationError(
                f"Memoire GPU insuffisante durant l'evaluation : {exc}\n"
                f"Reduisez --batch (par exemple 8) ou --imgsz."
            ) from exc
        raise EvaluationError(f"Echec de la validation Ultralytics : {exc}") from exc

    class_names = [dataset.class_name(i) for i in range(dataset.num_classes)]
    box = getattr(metrics, "box", None)
    global_metrics = {
        "precision": float(getattr(box, "mp", float("nan"))),
        "recall": float(getattr(box, "mr", float("nan"))),
        "mAP50": float(getattr(box, "map50", float("nan"))),
        "mAP50_95": float(getattr(box, "map", float("nan"))),
        "mAP75": float(getattr(box, "map75", float("nan"))),
    }

    per_class_metrics: dict[str, Any] = {}
    if box is not None:
        maps = getattr(box, "maps", None)
        ap50 = getattr(box, "ap50", None)
        precisions = getattr(box, "p", None)
        recalls = getattr(box, "r", None)
        # ap_class_index est un tableau NumPy : `or []` declencherait une
        # evaluation de verite ambigue, d'ou la normalisation explicite.
        raw_indices = getattr(box, "ap_class_index", None)
        indices = [] if raw_indices is None else list(np.atleast_1d(raw_indices))
        for position, class_id in enumerate(indices):
            class_name = (
                class_names[int(class_id)] if int(class_id) < len(class_names) else str(class_id)
            )
            entry: dict[str, float] = {}
            if maps is not None and int(class_id) < len(maps):
                entry["mAP50_95"] = round(float(maps[int(class_id)]), 4)
            for label, source in (("mAP50", ap50), ("precision", precisions), ("recall", recalls)):
                if source is not None and position < len(source):
                    entry[label] = round(float(source[position]), 4)
            per_class_metrics[class_name] = entry

    speed = getattr(metrics, "speed", {}) or {}
    speed_ms = {str(k): round(float(v), 3) for k, v in speed.items()}
    total_ms = sum(speed_ms.values())

    parameters = None
    with contextlib.suppress(AttributeError, TypeError):  # pragma: no cover
        # Ultralytics annote `.model` comme `str | None` alors qu'il s'agit
        # d'un nn.Module une fois les poids charges.
        module = model.model
        parameters = int(sum(p.numel() for p in module.parameters()))  # type: ignore[union-attr]

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": describe_environment(),
        "model": {
            "weights": str(weights_path),
            "weights_size_bytes": weights_path.stat().st_size,
            "weights_size_human": human_bytes(weights_path.stat().st_size),
            "parameters": parameters,
            "class_names": class_names,
        },
        "evaluation": {
            "data_yaml": str(data_path),
            "split": split,
            "imgsz": imgsz,
            "batch": batch,
            "conf": conf,
            "iou": iou,
            "device": resolved_device,
        },
        "metrics": {k: (round(v, 4) if v == v else None) for k, v in global_metrics.items()},
        "per_class": per_class_metrics,
        "speed_ms_per_image": speed_ms,
        "throughput_fps": round(1000.0 / total_ms, 2) if total_ms > 0 else None,
        "ultralytics_save_dir": str(getattr(metrics, "save_dir", "") or ""),
    }

    if error_analysis:
        from .config import InferenceConfig
        from .predict import PPEDetector

        LOGGER.info("Analyse d'erreurs au seuil operationnel conf=%.2f...", error_conf)
        detector = PPEDetector(
            InferenceConfig(
                weights=str(weights_path),
                conf=error_conf,
                iou=0.45,
                imgsz=imgsz,
                device=device,
            )
        )
        split_dir_name = "valid" if split in {"valid", "val"} else split
        if split_dir_name not in dataset.splits:
            LOGGER.warning("Split '%s' absent du dataset — analyse d'erreurs ignoree.", split_dir_name)
        else:
            report["error_analysis"] = analyse_errors(
                detector,
                dataset.splits[split_dir_name],
                dataset.labels_dir(split_dir_name),
                class_names,
                limit=error_limit,
            )

    report["known_limitations"] = _known_limitations(report)

    base_name = name or f"evaluation_{split}"
    out_dir = ensure_dir(resolve_path(output_dir))
    write_json(out_dir / f"{base_name}.json", report)
    write_text(out_dir / f"{base_name}.md", render_evaluation_markdown(report))
    LOGGER.info("Rapport ecrit : %s", out_dir / f"{base_name}.md")
    return report


def _known_limitations(report: dict[str, Any]) -> list[str]:
    """Deduit du rapport une liste de limites connues, en francais."""
    limitations = [
        "Le split de test provient du meme export Roboflow que l'entrainement : "
        "l'audit a identifie des images issues d'une meme photo source reparties "
        "entre plusieurs splits, ce qui rend les metriques optimistes par rapport "
        "a un deploiement sur un chantier inconnu.",
        "Les metriques sont calculees sur des annotations dont 349 lignes ont ete "
        "converties depuis des polygones de segmentation : la boite englobante d'un "
        "polygone est systematiquement au moins aussi grande que l'objet reel.",
    ]
    per_class = report.get("per_class", {})
    weak = [
        name
        for name, values in per_class.items()
        if isinstance(values.get("mAP50"), (int, float)) and values["mAP50"] < 0.5
    ]
    if weak:
        limitations.append(
            "Classes dont la mAP@0.50 reste inferieure a 0.50 : "
            + ", ".join(sorted(weak))
            + ". Ces classes demandent davantage d'exemples annotes."
        )
    errors = report.get("error_analysis", {})
    confusions = errors.get("class_confusions", [])
    if confusions:
        top = confusions[0]
        limitations.append(
            f"Confusion la plus frequente : {top['count']} objet(s) de classe "
            f"'{top['actual']}' predits comme '{top['predicted']}'."
        )
    return limitations


def render_evaluation_markdown(report: dict[str, Any]) -> str:
    """Genere le rapport Markdown d'evaluation."""
    lines: list[str] = []
    evaluation = report["evaluation"]
    model = report["model"]
    metrics = report["metrics"]

    lines.append(f"# Rapport d'evaluation — split `{evaluation['split']}`")
    lines.append("")
    lines.append(f"- **Genere le** : {report['generated_at']}")
    lines.append(f"- **Poids** : `{model['weights']}` ({model['weights_size_human']})")
    lines.append(f"- **Parametres** : {model['parameters'] or 'n/a'}")
    lines.append(f"- **Dataset** : `{evaluation['data_yaml']}`")
    lines.append(
        f"- **Parametres d'evaluation** : imgsz={evaluation['imgsz']}, batch={evaluation['batch']}, "
        f"conf={evaluation['conf']}, iou={evaluation['iou']}, device={evaluation['device']}"
    )
    lines.append("")

    lines.append("## Metriques globales")
    lines.append("")
    lines.append(
        markdown_table(
            ["Metrique", "Valeur"],
            [
                ["Precision (moyenne)", metrics.get("precision")],
                ["Rappel (moyen)", metrics.get("recall")],
                ["**mAP@0.50**", metrics.get("mAP50")],
                ["**mAP@0.50:0.95**", metrics.get("mAP50_95")],
                ["mAP@0.75", metrics.get("mAP75")],
            ],
        )
    )
    lines.append("")

    lines.append("## Metriques par classe")
    lines.append("")
    per_class = report.get("per_class", {})
    if per_class:
        rows = [
            [
                name,
                values.get("precision", "-"),
                values.get("recall", "-"),
                values.get("mAP50", "-"),
                values.get("mAP50_95", "-"),
            ]
            for name, values in sorted(per_class.items())
        ]
        lines.append(markdown_table(["Classe", "Precision", "Rappel", "mAP@0.50", "mAP@0.50:0.95"], rows))
    else:
        lines.append("Aucune metrique par classe disponible.")
    lines.append("")

    lines.append("## Vitesse")
    lines.append("")
    speed = report.get("speed_ms_per_image", {})
    rows = [[key, f"{value} ms"] for key, value in speed.items()]
    if report.get("throughput_fps"):
        rows.append(["**Debit**", f"{report['throughput_fps']} images/s"])
    lines.append(markdown_table(["Etape", "Temps par image"], rows))
    lines.append("")

    errors = report.get("error_analysis")
    if errors:
        lines.append("## Analyse d'erreurs")
        lines.append("")
        lines.append(
            f"Realisee au seuil operationnel `conf={errors['conf_threshold']}` et "
            f"`IoU>={errors['iou_threshold']}` sur {errors['images_analysed']} image(s)."
        )
        lines.append("")
        totals = errors["totals"]
        lines.append(
            markdown_table(
                ["Indicateur", "Total"],
                [
                    ["Vrais positifs", totals["true_positives"]],
                    ["Faux positifs", totals["false_positives"]],
                    ["Faux negatifs", totals["false_negatives"]],
                ],
            )
        )
        lines.append("")
        lines.append("### Detail par classe")
        lines.append("")
        lines.append(
            markdown_table(
                ["Classe", "VP", "FP", "FN", "Precision", "Rappel", "F1"],
                [
                    [
                        name,
                        values["true_positives"],
                        values["false_positives"],
                        values["false_negatives"],
                        values["precision"],
                        values["recall"],
                        values["f1"],
                    ]
                    for name, values in sorted(errors["per_class"].items())
                ],
            )
        )
        lines.append("")
        if errors["class_confusions"]:
            lines.append("### Confusions entre classes")
            lines.append("")
            lines.append(
                markdown_table(
                    ["Classe reelle", "Predite comme", "Occurrences"],
                    [
                        [item["actual"], item["predicted"], item["count"]]
                        for item in errors["class_confusions"][:15]
                    ],
                )
            )
            lines.append("")
        lines.append("### Pires exemples (a inspecter en priorite)")
        lines.append("")
        lines.append(
            markdown_table(
                ["Image", "Reference", "Predictions", "VP", "FP", "FN", "Score"],
                [
                    [
                        item["image"],
                        item["n_ground_truth"],
                        item["n_predictions"],
                        item["true_positives"],
                        item["false_positives"],
                        item["false_negatives"],
                        item["score"],
                    ]
                    for item in errors["worst_examples"][:10]
                ],
            )
        )
        lines.append("")
        lines.append("### Meilleurs exemples")
        lines.append("")
        lines.append(
            markdown_table(
                ["Image", "Reference", "Predictions", "VP", "FP", "FN", "Score"],
                [
                    [
                        item["image"],
                        item["n_ground_truth"],
                        item["n_predictions"],
                        item["true_positives"],
                        item["false_positives"],
                        item["false_negatives"],
                        item["score"],
                    ]
                    for item in errors["best_examples"][:10]
                ],
            )
        )
        lines.append("")

    lines.append("## Limites connues")
    lines.append("")
    for item in report.get("known_limitations", []):
        lines.append(f"- {item}")
    lines.append("")

    if report.get("ultralytics_save_dir"):
        lines.append(
            f"> Les courbes et la matrice de confusion generees par Ultralytics se trouvent "
            f"dans `{report['ultralytics_save_dir']}`."
        )
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments de la commande d'evaluation."""
    parser = argparse.ArgumentParser(
        prog="python -m ppe_detection.evaluate",
        description="Evalue un modele de detection EPI sur un split donne.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default="artifacts/models/best.pt", help="Poids a evaluer.")
    parser.add_argument(
        "--data",
        default="artifacts/dataset_detection/data.yaml",
        help="data.yaml du dataset.",
    )
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"], help="Split evalue.")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'inference.")
    parser.add_argument("--batch", type=int, default=16, help="Taille de lot.")
    parser.add_argument("--conf", type=float, default=0.001, help="Seuil de confiance pour la mAP.")
    parser.add_argument("--iou", type=float, default=0.6, help="Seuil IoU de la NMS.")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | 0")
    parser.add_argument("--no-error-analysis", action="store_true", help="Desactive l'analyse d'erreurs.")
    parser.add_argument(
        "--error-conf",
        type=float,
        default=0.25,
        help="Seuil operationnel de l'analyse d'erreurs.",
    )
    parser.add_argument("--error-limit", type=int, default=0, help="Limite d'images analysees (0 = toutes).")
    parser.add_argument("--output", default="artifacts/reports", help="Repertoire des rapports.")
    parser.add_argument("--name", default=None, help="Nom de base des fichiers de rapport.")
    parser.add_argument("--log-level", default="INFO", help="Niveau de log.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entree CLI. Retourne le code de sortie du processus."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=project_root() / "artifacts" / "logs" / "evaluate.log")

    try:
        report = evaluate(
            args.weights,
            args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            error_analysis=not args.no_error_analysis,
            error_conf=args.error_conf,
            error_limit=args.error_limit,
            output_dir=args.output,
            name=args.name,
        )
    except (EvaluationError, ConfigError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        return 2

    metrics = report["metrics"]
    LOGGER.info(
        "mAP@0.50=%s | mAP@0.50:0.95=%s | precision=%s | rappel=%s",
        metrics.get("mAP50"),
        metrics.get("mAP50_95"),
        metrics.get("precision"),
        metrics.get("recall"),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
