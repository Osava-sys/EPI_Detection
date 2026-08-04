"""Calibration des seuils de confiance par classe.

Un seuil de confiance unique pour toutes les classes est un compromis
mediocre : chaque classe a sa propre distribution de scores. Sur ce projet,
`Safety Helmet` produit beaucoup de faux positifs a 0.25 tandis que
`Safety Gloves` manque des objets a ce meme seuil.

Ce module balaie les seuils possibles pour chaque classe et retient celui qui
maximise le score F1, ou la precision sous contrainte de rappel minimal.

Protocole
---------
La calibration s'effectue **exclusivement sur le split de validation**. Choisir
un seuil sur le test reviendrait a ajuster le modele sur les donnees censees
l'evaluer, et l'estimation finale ne vaudrait plus rien. Le module refuse donc
explicitement le split ``test`` sauf mention contraire assumee.

Usage
-----
    python -m ppe_detection.calibrate --weights artifacts/models/best.pt \\
        --data artifacts/dataset_detection/data.yaml --split valid --apply
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import ConfigError, load_dataset_config, load_inference_config
from .evaluate import EvaluationError, _iou_matrix, _load_ground_truth
from .utils import (
    ensure_dir,
    get_logger,
    is_image,
    read_yaml,
    resolve_path,
    setup_logging,
    write_json,
    write_text,
    write_yaml,
)

LOGGER = get_logger(__name__)

DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(
    round(float(v), 2) for v in np.arange(0.05, 0.96, 0.05)
)
"""Grille de seuils balayee par defaut."""

SCAN_CONF = 0.01
"""Confiance d'inference tres basse : on collecte tout, on filtre ensuite."""


@dataclass
class ClassCalibration:
    """Resultat de calibration pour une classe."""

    class_id: int
    class_name: str
    n_ground_truth: int
    best_threshold: float
    best_f1: float
    precision_at_best: float
    recall_at_best: float
    f1_at_baseline: float
    baseline_threshold: float
    curve: list[dict[str, float]] = field(default_factory=list)

    @property
    def gain(self) -> float:
        """Gain de F1 par rapport au seuil uniforme de reference."""
        return self.best_f1 - self.f1_at_baseline

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "n_ground_truth": self.n_ground_truth,
            "best_threshold": round(self.best_threshold, 3),
            "best_f1": round(self.best_f1, 4),
            "precision_at_best": round(self.precision_at_best, 4),
            "recall_at_best": round(self.recall_at_best, 4),
            "baseline_threshold": round(self.baseline_threshold, 3),
            "f1_at_baseline": round(self.f1_at_baseline, 4),
            "f1_gain": round(self.gain, 4),
            "curve": self.curve,
        }


def collect_matches(
    detector: Any,
    images_dir: Path,
    labels_dir: Path,
    num_classes: int,
    *,
    iou_threshold: float = 0.5,
    limit: int = 0,
) -> tuple[list[tuple[int, float, bool]], np.ndarray]:
    """Collecte toutes les predictions et leur statut d'appariement.

    Chaque prediction est appariee a la boite de reference de meilleur IoU et de
    meme classe, par ordre de confiance decroissante — la meme regle que
    l'analyse d'erreurs, afin que les deux rapports restent comparables.

    Args:
        detector: Detecteur initialise, configure a une confiance tres basse.
        images_dir: Images du split.
        labels_dir: Labels correspondants.
        num_classes: Nombre de classes.
        iou_threshold: IoU minimal d'appariement.
        limit: Nombre maximal d'images (0 = toutes).

    Returns:
        ``(predictions, gt_counts)`` ou ``predictions`` est une liste de
        ``(class_id, confiance, est_vrai_positif)`` et ``gt_counts`` le nombre
        d'objets de reference par classe.
    """
    import cv2

    image_paths = sorted(p for p in images_dir.iterdir() if p.is_file() and is_image(p))
    if limit:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise EvaluationError(f"Aucune image exploitable dans {images_dir}.")

    predictions: list[tuple[int, float, bool]] = []
    gt_counts = np.zeros(num_classes, dtype=int)

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning("Image illisible ignoree : %s", image_path.name)
            continue
        height, width = image.shape[:2]

        gt_boxes, gt_classes = _load_ground_truth(
            labels_dir / f"{image_path.stem}.txt", width, height, num_classes
        )
        for class_id in gt_classes:
            if 0 <= class_id < num_classes:
                gt_counts[class_id] += 1

        prediction = detector.predict_array(image, source_name=image_path.name)
        detections = prediction.detections
        if not detections:
            continue

        pred_boxes = np.asarray([d["bbox_xyxy"] for d in detections], dtype=float)
        pred_classes = np.asarray([d["class_id"] for d in detections], dtype=int)
        pred_confs = np.asarray([d["confidence"] for d in detections], dtype=float)

        order = np.argsort(-pred_confs)
        ious = _iou_matrix(pred_boxes, gt_boxes)
        matched: set[int] = set()

        for pred_index in order:
            class_id = int(pred_classes[pred_index])
            confidence = float(pred_confs[pred_index])
            is_tp = False
            if gt_boxes.size:
                candidates = [
                    (ious[pred_index, gt_index], gt_index)
                    for gt_index in range(len(gt_boxes))
                    if gt_index not in matched
                    and gt_classes[gt_index] == class_id
                    and ious[pred_index, gt_index] >= iou_threshold
                ]
                if candidates:
                    _, best_gt = max(candidates, key=lambda item: item[0])
                    matched.add(best_gt)
                    is_tp = True
            predictions.append((class_id, confidence, is_tp))

        if index % 200 == 0:
            LOGGER.info("Calibration : %d/%d images", index, len(image_paths))

    return predictions, gt_counts


def calibrate_class(
    class_id: int,
    class_name: str,
    predictions: Sequence[tuple[int, float, bool]],
    n_ground_truth: int,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    baseline: float = 0.25,
    min_recall: float = 0.0,
) -> ClassCalibration:
    """Determine le seuil optimal d'une classe par balayage.

    Args:
        class_id: Identifiant de la classe.
        class_name: Nom de la classe.
        predictions: Toutes les predictions collectees, toutes classes confondues.
        n_ground_truth: Nombre d'objets de reference de cette classe.
        thresholds: Grille de seuils a evaluer.
        baseline: Seuil uniforme servant de reference pour mesurer le gain.
        min_recall: Rappel minimal exige; les seuils qui l'enfreignent sont
            ecartes sauf si aucun ne le respecte.

    Returns:
        Le resultat de calibration de la classe.
    """
    scores = np.asarray(
        [(conf, tp) for cid, conf, tp in predictions if cid == class_id], dtype=float
    )
    curve: list[dict[str, float]] = []

    def metrics_at(threshold: float) -> tuple[float, float, float]:
        """Precision, rappel et F1 au seuil donne."""
        if scores.size == 0 or n_ground_truth == 0:
            return 0.0, 0.0, 0.0
        kept = scores[scores[:, 0] >= threshold]
        tp = float(kept[:, 1].sum())
        fp = float(len(kept) - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / n_ground_truth
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    best_threshold = baseline
    best_f1 = -1.0
    best_precision = best_recall = 0.0

    for threshold in thresholds:
        precision, recall, f1 = metrics_at(float(threshold))
        curve.append(
            {
                "threshold": round(float(threshold), 3),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
        if recall < min_recall:
            continue
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(threshold)
            best_precision, best_recall = precision, recall

    if best_f1 < 0:  # aucun seuil ne respecte min_recall : on relache la contrainte
        LOGGER.warning(
            "Classe '%s' : aucun seuil n'atteint le rappel minimal de %.2f. "
            "Selection sans contrainte.",
            class_name,
            min_recall,
        )
        for entry in curve:
            if entry["f1"] > best_f1:
                best_f1 = entry["f1"]
                best_threshold = entry["threshold"]
                best_precision, best_recall = entry["precision"], entry["recall"]

    _, _, f1_baseline = metrics_at(baseline)

    return ClassCalibration(
        class_id=class_id,
        class_name=class_name,
        n_ground_truth=int(n_ground_truth),
        best_threshold=best_threshold,
        best_f1=max(best_f1, 0.0),
        precision_at_best=best_precision,
        recall_at_best=best_recall,
        f1_at_baseline=f1_baseline,
        baseline_threshold=baseline,
        curve=curve,
    )


def render_markdown(report: dict[str, Any]) -> str:
    """Rapport lisible de la calibration."""
    lines = [
        "# Calibration des seuils de confiance par classe",
        "",
        f"- **Genere le** : {report['generated_at']}",
        f"- **Poids** : `{report['weights']}`",
        f"- **Split** : `{report['split']}` ({report['n_images']} images)",
        f"- **Seuil de reference** : {report['baseline_threshold']}",
        f"- **IoU d'appariement** : {report['iou_threshold']}",
        "",
        "> Les seuils sont choisis sur la **validation** uniquement. Les appliquer",
        "> puis mesurer sur le test reste un protocole valide ; les choisir sur le",
        "> test ne le serait pas.",
        "",
        "## Seuils retenus",
        "",
        "| Classe | Objets | Seuil retenu | F1 obtenu | F1 au seuil uniforme | Gain | Precision | Rappel |",
        "|--------|--------|--------------|-----------|----------------------|------|-----------|--------|",
    ]
    for entry in report["classes"]:
        lines.append(
            f"| {entry['class_name']} | {entry['n_ground_truth']} | "
            f"**{entry['best_threshold']}** | {entry['best_f1']:.4f} | "
            f"{entry['f1_at_baseline']:.4f} | {entry['f1_gain']:+.4f} | "
            f"{entry['precision_at_best']:.4f} | {entry['recall_at_best']:.4f} |"
        )

    summary = report["summary"]
    lines += [
        "",
        "## Synthese",
        "",
        f"- F1 macro au seuil uniforme {report['baseline_threshold']} : "
        f"**{summary['macro_f1_baseline']:.4f}**",
        f"- F1 macro avec seuils calibres : **{summary['macro_f1_calibrated']:.4f}**",
        f"- Gain macro : **{summary['macro_f1_gain']:+.4f}**",
        "",
        "## Extrait YAML a reporter dans `configs/inference.yaml`",
        "",
        "```yaml",
        "inference:",
        "  class_conf:",
    ]
    for entry in report["classes"]:
        lines.append(f"    {entry['class_name']}: {entry['best_threshold']}")
    lines += ["```", ""]
    return "\n".join(lines)


def apply_to_config(config_path: Path, calibrations: Sequence[ClassCalibration]) -> None:
    """Ecrit les seuils calibres dans la section ``inference.class_conf``.

    Le reste du fichier est preserve. Les commentaires YAML, eux, ne survivent
    pas a la reecriture : c'est pourquoi l'application reste optionnelle et que
    le rapport fournit toujours l'extrait a recopier manuellement.
    """
    data = read_yaml(config_path) if config_path.is_file() else {}
    if not isinstance(data, dict):
        raise ConfigError(f"{config_path} ne contient pas un mapping YAML.")
    section = data.setdefault("inference", {})
    if not isinstance(section, dict):
        raise ConfigError(f"La cle 'inference' de {config_path} n'est pas un mapping.")
    section["class_conf"] = {c.class_name: round(c.best_threshold, 3) for c in calibrations}
    write_yaml(config_path, data)
    LOGGER.info("Seuils ecrits dans %s (section inference.class_conf).", config_path)


def calibrate(
    weights: str | Path,
    data: str | Path,
    *,
    split: str = "valid",
    iou_threshold: float = 0.5,
    baseline: float = 0.25,
    min_recall: float = 0.0,
    limit: int = 0,
    device: str = "auto",
    imgsz: int = 640,
    output_dir: str | Path = "artifacts/reports",
    name: str = "threshold_calibration",
    apply: bool = False,
    config_path: str | Path = "configs/inference.yaml",
    allow_test_split: bool = False,
) -> dict[str, Any]:
    """Calibre les seuils par classe et produit un rapport.

    Args:
        weights: Poids du modele.
        data: ``data.yaml`` du dataset.
        split: Split de calibration (``valid`` recommande).
        iou_threshold: IoU d'appariement.
        baseline: Seuil uniforme de reference.
        min_recall: Rappel minimal exige par classe.
        limit: Nombre maximal d'images.
        device: Peripherique d'inference.
        imgsz: Taille d'inference.
        output_dir: Repertoire des rapports.
        name: Nom de base des fichiers produits.
        apply: Ecrit les seuils dans ``config_path``.
        config_path: Fichier de configuration d'inference.
        allow_test_split: Autorise explicitement la calibration sur le test.

    Returns:
        Le rapport de calibration.

    Raises:
        EvaluationError: Si le split est absent ou si la calibration sur le test
            est demandee sans l'autoriser explicitement.
    """
    from datetime import datetime, timezone

    from .predict import PPEDetector

    if split == "test" and not allow_test_split:
        raise EvaluationError(
            "Calibrer les seuils sur le split 'test' invaliderait l'evaluation finale : "
            "les seuils seraient ajustes sur les donnees censees mesurer la generalisation.\n"
            "Utilisez --split valid, ou --allow-test-split si vous assumez ce choix."
        )

    dataset = load_dataset_config(data)
    if split not in dataset.splits:
        raise EvaluationError(
            f"Split '{split}' absent de {data}. Disponibles : {', '.join(dataset.splits)}."
        )
    images_dir = dataset.splits[split]
    labels_dir = dataset.labels_dir(split)
    class_names = list(dataset.names)

    # Confiance tres basse : on collecte l'ensemble des candidats une seule fois,
    # le balayage de seuils se fait ensuite hors ligne, sans reinference.
    inference_config = load_inference_config(
        None, weights=str(weights), conf=SCAN_CONF, device=device, imgsz=imgsz
    )
    detector = PPEDetector(inference_config)

    LOGGER.info("Collecte des predictions sur le split '%s'...", split)
    predictions, gt_counts = collect_matches(
        detector,
        images_dir,
        labels_dir,
        len(class_names),
        iou_threshold=iou_threshold,
        limit=limit,
    )

    calibrations = [
        calibrate_class(
            class_id,
            class_name,
            predictions,
            int(gt_counts[class_id]),
            baseline=baseline,
            min_recall=min_recall,
        )
        for class_id, class_name in enumerate(class_names)
    ]

    macro_baseline = float(np.mean([c.f1_at_baseline for c in calibrations]))
    macro_calibrated = float(np.mean([c.best_f1 for c in calibrations]))

    n_images = len([p for p in images_dir.iterdir() if p.is_file() and is_image(p)])
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(resolve_path(weights)),
        "data": str(resolve_path(data)),
        "split": split,
        "n_images": limit or n_images,
        "iou_threshold": iou_threshold,
        "baseline_threshold": baseline,
        "min_recall": min_recall,
        "classes": [c.to_dict() for c in calibrations],
        "summary": {
            "macro_f1_baseline": round(macro_baseline, 4),
            "macro_f1_calibrated": round(macro_calibrated, 4),
            "macro_f1_gain": round(macro_calibrated - macro_baseline, 4),
        },
        "class_conf": {c.class_name: round(c.best_threshold, 3) for c in calibrations},
    }

    out_dir = ensure_dir(resolve_path(output_dir))
    write_json(out_dir / f"{name}.json", report)
    write_text(out_dir / f"{name}.md", render_markdown(report))
    LOGGER.info("Rapport ecrit : %s", out_dir / f"{name}.md")

    if apply:
        apply_to_config(resolve_path(config_path), calibrations)

    return report


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m ppe_detection.calibrate",
        description="Calibre les seuils de confiance par classe sur la validation.",
    )
    parser.add_argument("--weights", default="artifacts/models/best.pt", help="Poids a calibrer.")
    parser.add_argument(
        "--data", default="artifacts/dataset_detection/data.yaml", help="data.yaml du dataset."
    )
    parser.add_argument("--split", default="valid", help="Split de calibration (valid recommande).")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU d'appariement.")
    parser.add_argument(
        "--baseline", type=float, default=0.25, help="Seuil uniforme servant de reference."
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.0,
        help="Rappel minimal exige par classe (contrainte metier).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Nombre maximal d'images (0 = toutes).")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | 0")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'inference.")
    parser.add_argument("--output", default="artifacts/reports", help="Repertoire des rapports.")
    parser.add_argument("--name", default="threshold_calibration", help="Nom de base des rapports.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Ecrit les seuils retenus dans configs/inference.yaml.",
    )
    parser.add_argument(
        "--config", default="configs/inference.yaml", help="Configuration d'inference a mettre a jour."
    )
    parser.add_argument(
        "--allow-test-split",
        action="store_true",
        help="Autorise la calibration sur le test (invalide l'evaluation finale).",
    )
    parser.add_argument("--log-level", default="INFO", help="Niveau de log.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entree CLI."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        report = calibrate(
            args.weights,
            args.data,
            split=args.split,
            iou_threshold=args.iou,
            baseline=args.baseline,
            min_recall=args.min_recall,
            limit=args.limit,
            device=args.device,
            imgsz=args.imgsz,
            output_dir=args.output,
            name=args.name,
            apply=args.apply,
            config_path=args.config,
            allow_test_split=args.allow_test_split,
        )
    except (EvaluationError, ConfigError, FileNotFoundError) as exc:
        LOGGER.error("Calibration impossible : %s", exc)
        return 2

    summary = report["summary"]
    LOGGER.info(
        "F1 macro : %.4f (uniforme) -> %.4f (calibre) | gain %+.4f",
        summary["macro_f1_baseline"],
        summary["macro_f1_calibrated"],
        summary["macro_f1_gain"],
    )
    for entry in report["classes"]:
        LOGGER.info(
            "  %-15s seuil %.2f  F1 %.4f (%+.4f)",
            entry["class_name"],
            entry["best_threshold"],
            entry["best_f1"],
            entry["f1_gain"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
