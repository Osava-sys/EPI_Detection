"""Rendu des detections : couleurs deterministes, boites, etiquettes, graphiques.

Les couleurs sont derivees du *nom* de classe par hachage stable, ce qui garantit
qu'une classe conserve la meme couleur d'une execution a l'autre et d'un jeu de
poids a l'autre.
"""

from __future__ import annotations

import colorsys
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .utils import ensure_dir, get_logger

LOGGER = get_logger(__name__)

# Palette fixe pour les 7 classes EPI du projet (BGR, compatible OpenCV).
PPE_PALETTE_BGR: dict[str, tuple[int, int, int]] = {
    "Face Mask": (255, 191, 0),      # cyan-bleu
    "Person": (60, 180, 75),         # vert
    "Safety Gloves": (180, 120, 255),  # rose
    "Safety Harness": (0, 165, 255),   # orange
    "Safety Helmet": (0, 0, 255),      # rouge
    "Safety Shoes": (255, 0, 200),     # magenta
    "Safety Vest": (0, 255, 255),      # jaune
}

COMPLIANT_COLOR_BGR: tuple[int, int, int] = (60, 200, 60)      # vert
NON_COMPLIANT_COLOR_BGR: tuple[int, int, int] = (0, 0, 230)    # rouge
INDETERMINATE_COLOR_BGR: tuple[int, int, int] = (0, 190, 255)  # ambre


def color_for_class(class_name: str) -> tuple[int, int, int]:
    """Couleur BGR deterministe pour un nom de classe.

    Utilise la palette EPI si le nom y figure, sinon derive une teinte stable
    depuis un hachage SHA-1 du nom.
    """
    if class_name in PPE_PALETTE_BGR:
        return PPE_PALETTE_BGR[class_name]
    digest = hashlib.sha1(class_name.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(blue * 255), int(green * 255), int(red * 255))


def _auto_line_width(image_shape: Sequence[int]) -> int:
    """Epaisseur de trait proportionnelle a la taille de l'image."""
    return max(1, round(0.002 * (image_shape[0] + image_shape[1]) / 2))


def draw_box(
    image: np.ndarray,
    xyxy: Sequence[float],
    *,
    label: str | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    line_width: int | None = None,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Dessine une boite (et son etiquette) **en place** sur l'image.

    Args:
        image: Image BGR modifiee sur place.
        xyxy: Coordonnees pixel ``(x1, y1, x2, y2)``.
        label: Texte optionnel affiche au-dessus de la boite.
        color: Couleur BGR.
        line_width: Epaisseur; deduite de la taille de l'image si ``None``.
        font_scale: Echelle de police OpenCV.

    Returns:
        L'image annotee (meme objet que ``image``).
    """
    thickness = line_width or _auto_line_width(image.shape)
    x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
    height, width = image.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_thickness = max(1, thickness - 1)
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        box_top = y1 - text_h - baseline - 2
        outside = box_top >= 0
        top = box_top if outside else y1 + 2
        cv2.rectangle(
            image,
            (x1, top),
            (x1 + text_w + 4, top + text_h + baseline + 2),
            color,
            -1,
            lineType=cv2.LINE_AA,
        )
        luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        text_color = (0, 0, 0) if luminance > 140 else (255, 255, 255)
        cv2.putText(
            image,
            label,
            (x1 + 2, top + text_h + 1),
            font,
            font_scale,
            text_color,
            text_thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def draw_detections(
    image: np.ndarray,
    detections: Iterable[Mapping[str, Any]],
    *,
    show_labels: bool = True,
    show_conf: bool = True,
    line_width: int | None = None,
    font_scale: float = 0.5,
    copy: bool = True,
) -> np.ndarray:
    """Dessine une liste de detections structurees sur une image.

    Args:
        image: Image BGR.
        detections: Elements exposant ``class_name``, ``confidence`` et
            ``bbox_xyxy`` (voir :mod:`ppe_detection.predict`).
        show_labels: Affiche le nom de classe.
        show_conf: Affiche le score de confiance.
        line_width: Epaisseur de trait.
        font_scale: Echelle de police.
        copy: Travaille sur une copie plutot que sur l'image d'origine.

    Returns:
        L'image annotee.
    """
    canvas = image.copy() if copy else image
    for detection in detections:
        class_name = str(detection.get("class_name", "?"))
        bbox = detection.get("bbox_xyxy")
        if bbox is None:
            continue
        parts: list[str] = []
        if show_labels:
            parts.append(class_name)
        if show_conf:
            confidence = detection.get("confidence")
            if confidence is not None:
                parts.append(f"{float(confidence):.2f}")
        label = " ".join(parts) if parts else None
        draw_box(
            canvas,
            list(bbox),  # type: ignore[arg-type]
            label=label,
            color=color_for_class(class_name),
            line_width=line_width,
            font_scale=font_scale,
        )
    return canvas


def draw_compliance(
    image: np.ndarray,
    persons: Iterable[Mapping[str, Any]],
    *,
    line_width: int | None = None,
    font_scale: float = 0.5,
    copy: bool = True,
) -> np.ndarray:
    """Surligne les personnes selon leur statut de conformite EPI.

    Args:
        image: Image BGR.
        persons: Enregistrements produits par :func:`ppe_detection.compliance.evaluate_compliance`.
        line_width: Epaisseur de trait.
        font_scale: Echelle de police.
        copy: Travaille sur une copie.

    Returns:
        L'image annotee.
    """
    canvas = image.copy() if copy else image
    thickness = line_width or _auto_line_width(canvas.shape)
    for person in persons:
        bbox = person.get("bbox_xyxy")
        if bbox is None:
            continue

        # Le statut lisse (video suivie) prime sur le verdict instantane.
        status = str(person.get("smoothed_status") or person.get("status") or "")
        if not status:
            status = "compliant" if person.get("compliant") else "non_compliant"

        prefix = ""
        track_id = person.get("track_id")
        if track_id is not None:
            prefix = f"#{track_id} "

        if status == "compliant":
            color = COMPLIANT_COLOR_BGR
            label = f"{prefix}CONFORME"
        elif status == "indeterminate":
            color = INDETERMINATE_COLOR_BGR
            unknown = [str(item) for item in (person.get("indeterminate_ppe") or [])]
            label = f"{prefix}INDETERMINE" + (f": {', '.join(unknown)}" if unknown else "")
        else:
            color = NON_COMPLIANT_COLOR_BGR
            missing_list = [str(item) for item in (person.get("missing_ppe") or [])]
            label = f"{prefix}NON CONFORME: " + (", ".join(missing_list) if missing_list else "?")

        draw_box(
            canvas,
            list(bbox),
            label=label,
            color=color,
            line_width=thickness + 1,
            font_scale=font_scale,
        )
    return canvas


def draw_yolo_labels(
    image: np.ndarray,
    labels: Iterable[tuple[int, float, float, float, float]],
    class_names: Sequence[str],
    *,
    line_width: int | None = None,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Dessine des annotations YOLO normalisees (verification visuelle du dataset).

    Args:
        image: Image BGR.
        labels: Tuples ``(class_id, cx, cy, w, h)`` normalises.
        class_names: Noms de classes indexes par identifiant.
        line_width: Epaisseur de trait.
        font_scale: Echelle de police.

    Returns:
        Une copie annotee de l'image.
    """
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for class_id, cx, cy, box_w, box_h in labels:
        x1 = (cx - box_w / 2.0) * width
        y1 = (cy - box_h / 2.0) * height
        x2 = (cx + box_w / 2.0) * width
        y2 = (cy + box_h / 2.0) * height
        name = class_names[class_id] if 0 <= class_id < len(class_names) else f"id{class_id}"
        draw_box(
            canvas,
            (x1, y1, x2, y2),
            label=name,
            color=color_for_class(name),
            line_width=line_width,
            font_scale=font_scale,
        )
    return canvas


def plot_class_distribution(
    counts_by_split: Mapping[str, Mapping[str, int]],
    class_names: Sequence[str],
    output_path: Path,
    *,
    title: str = "Distribution des classes par split",
) -> Path | None:
    """Trace un diagramme en barres groupees de la frequence des classes.

    Args:
        counts_by_split: ``{split: {nom_classe: effectif}}``.
        class_names: Ordre d'affichage des classes.
        output_path: Fichier PNG de sortie.
        title: Titre du graphique.

    Returns:
        Le chemin du PNG, ou ``None`` si matplotlib est indisponible.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib indisponible — graphique de distribution non genere.")
        return None

    splits = list(counts_by_split)
    if not splits or not class_names:
        return None

    indices = np.arange(len(class_names))
    bar_width = 0.8 / max(len(splits), 1)
    fig, axis = plt.subplots(figsize=(max(8, len(class_names) * 1.4), 5.5))

    for offset, split in enumerate(splits):
        values = [counts_by_split[split].get(name, 0) for name in class_names]
        positions = indices + offset * bar_width - 0.4 + bar_width / 2
        bars = axis.bar(positions, values, bar_width, label=split)
        for rect, value in zip(bars, values, strict=True):
            if value:
                axis.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height(),
                    str(value),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    axis.set_xticks(indices)
    axis.set_xticklabels(class_names, rotation=20, ha="right")
    axis.set_ylabel("Nombre d'instances")
    axis.set_title(title)
    axis.legend(title="Split")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path


def plot_box_size_distribution(
    areas_by_split: Mapping[str, Sequence[float]],
    output_path: Path,
    *,
    title: str = "Distribution de la taille relative des boites",
) -> Path | None:
    """Trace un histogramme de la racine carree de l'aire normalisee des boites.

    Args:
        areas_by_split: ``{split: [aire_normalisee, ...]}``.
        output_path: Fichier PNG de sortie.
        title: Titre du graphique.

    Returns:
        Le chemin du PNG, ou ``None`` si matplotlib est indisponible ou s'il n'y
        a aucune donnee.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib indisponible — histogramme des tailles non genere.")
        return None

    usable = {split: values for split, values in areas_by_split.items() if values}
    if not usable:
        return None

    fig, axis = plt.subplots(figsize=(9, 5))
    bins = [i / 50.0 for i in range(51)]
    for split, areas in usable.items():
        scales = np.sqrt(np.clip(np.asarray(areas, dtype=float), 0.0, 1.0))
        axis.hist(scales, bins=bins, alpha=0.55, label=f"{split} (n={len(areas)})")
    axis.set_xlabel("sqrt(aire normalisee) — proxy de l'echelle de l'objet")
    axis.set_ylabel("Nombre de boites")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path
