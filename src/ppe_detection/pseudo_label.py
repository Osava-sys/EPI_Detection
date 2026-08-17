"""Pre-annotation d'une classe absente d'un dataset externe.

Pourquoi c'est necessaire
=========================
Les datasets publics n'annotent que ce qui interesse leurs auteurs. Le dataset
``Voxel51/hard-hat-detection`` annote 18 966 casques et 5 785 tetes, mais
seulement 751 personnes sur 5 000 images : les personnes y sont **visibles mais
non annotees**.

Fusionner tel quel apprend au modele qu'il n'y a *rien a detecter* la ou une
personne se trouve pourtant. Ce phenomene a deja ete constate sur ce projet :
apres l'import d'Open Images, le modele avait cesse de detecter les personnes
sur l'imagerie sportive, precisement parce que ces images en contenaient sans
les annoter.

Ce module comble le trou en pre-annotant la classe manquante avec un modele
existant, avant la fusion.

Limites, a garder en tete
=========================
Une pre-annotation n'est pas une verite terrain : elle **propage les erreurs du
modele qui la produit**. Elle ne se justifie que lorsque

* la classe pre-annotee est deja bien maitrisee par le modele source ;
* un seuil de confiance eleve est applique, pour privilegier la precision au
  rappel — une boite manquante est moins nuisible qu'une boite fausse ;
* les annotations produites sont tracables, ce que fait le rapport JSON.

Ne jamais pre-annoter une classe faible : cela figerait ses erreurs dans les
donnees d'entrainement.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .config import ConfigError, load_inference_config
from .dataset_merge import resolve_split_dirs
from .predict import PPEDetector, PredictionError
from .taxonomy import ClassSchema, extended_schema
from .utils import IMAGE_EXTENSIONS, get_logger, resolve_path, setup_logging, write_json

LOGGER = get_logger(__name__)

SPLITS: tuple[str, ...] = ("train", "valid", "test")


class PseudoLabelError(RuntimeError):
    """Erreur bloquante durant la pre-annotation."""


def _to_yolo_line(box: list[float], width: int, height: int, class_id: int) -> str | None:
    """Convertit une boite xyxy en ligne YOLO normalisee."""
    x1, y1, x2, y2 = (float(v) for v in box)
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    if bw <= 0 or bh <= 0:
        return None
    # Les boites debordant legerement du cadre sont ramenees dans [0, 1].
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    bw, bh = min(bw, 1.0), min(bh, 1.0)
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def pseudo_label_dataset(
    dataset_dir: str | Path,
    detector: PPEDetector,
    *,
    class_name: str = "Person",
    source_class_id: int,
    conf: float = 0.50,
    splits: tuple[str, ...] = SPLITS,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ajoute les detections d'une classe aux annotations d'un dataset.

    Les annotations existantes sont **preservees** : les nouvelles lignes sont
    ajoutees a la suite. Un fichier deja pourvu d'une annotation de cette classe
    est laisse intact, ce qui rend l'operation idempotente.

    Args:
        dataset_dir: Racine du dataset a completer.
        detector: Detecteur servant de source d'annotations.
        class_name: Classe a pre-annoter.
        source_class_id: Identifiant que prendra cette classe dans le dataset.
        conf: Seuil de confiance. Volontairement eleve : mieux vaut manquer une
            personne que d'en inventer une.
        splits: Splits a traiter.
        dry_run: Analyse sans rien ecrire.

    Returns:
        Un bilan de l'operation.
    """
    root = resolve_path(dataset_dir)
    if not root.is_dir():
        raise PseudoLabelError(f"Dataset introuvable : {root}")

    stats: Counter[str] = Counter()
    per_split: dict[str, dict[str, int]] = {}

    for split in splits:
        resolved = resolve_split_dirs(root, split)
        if resolved is None:
            continue
        images_dir, labels_dir = resolved
        added = images = skipped = 0

        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            images += 1
            label_path = labels_dir / f"{image_path.stem}.txt"
            existing = (
                label_path.read_text(encoding="utf-8").splitlines()
                if label_path.is_file()
                else []
            )
            # Idempotence : ne pas re-annoter un fichier deja complete.
            if any(line.split()[:1] == [str(source_class_id)] for line in existing if line.split()):
                skipped += 1
                continue

            import cv2

            image = cv2.imread(str(image_path))
            if image is None:
                LOGGER.warning("Image illisible, ignoree : %s", image_path)
                continue

            prediction = detector.predict_array(image, source_name=image_path.name)
            height, width = image.shape[:2]
            new_lines: list[str] = []
            for detection in prediction.detections:
                if detection.get("class_name") != class_name:
                    continue
                if float(detection.get("confidence", 0.0)) < conf:
                    continue
                line = _to_yolo_line(detection["bbox_xyxy"], width, height, source_class_id)
                if line:
                    new_lines.append(line)

            if new_lines and not dry_run:
                merged = [*(line for line in existing if line.strip()), *new_lines]
                label_path.write_text("\n".join(merged) + "\n", encoding="utf-8")
            added += len(new_lines)

        per_split[split] = {"images": images, "added": added, "already_labelled": skipped}
        stats["images"] += images
        stats["added"] += added
        stats["skipped"] += skipped
        LOGGER.info(
            "[%s] %d image(s), %d annotation(s) '%s' ajoutee(s), %d deja pourvue(s).",
            split,
            images,
            added,
            class_name,
            skipped,
        )

    return {
        "dataset": str(root),
        "class_name": class_name,
        "class_id": source_class_id,
        "conf": conf,
        "weights": str(detector.weights_path),
        "dry_run": dry_run,
        "totals": dict(stats),
        "per_split": per_split,
        "caveat": (
            "Annotations produites par un modele, non verifiees humainement. "
            "Elles propagent les erreurs de ce modele."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m ppe_detection.pseudo_label",
        description="Pre-annote une classe absente d'un dataset externe.",
    )
    parser.add_argument("--dataset", required=True, help="Racine du dataset a completer.")
    parser.add_argument("--weights", required=True, help="Poids servant de source d'annotations.")
    parser.add_argument("--class-name", default="Person", help="Classe a pre-annoter.")
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Identifiant cible (defaut : celui de la classe dans le schema etendu).",
    )
    parser.add_argument("--conf", type=float, default=0.50, help="Seuil de confiance.")
    parser.add_argument("--device", default="auto", help="Device d'inference.")
    parser.add_argument("--dry-run", action="store_true", help="Analyse sans rien ecrire.")
    parser.add_argument(
        "--report", default="artifacts/reports/pseudo_label.json", help="Rapport JSON."
    )
    parser.add_argument("--log-level", default="INFO", help="Niveau de log.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entree CLI."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    schema: ClassSchema = extended_schema()
    try:
        class_id = args.class_id if args.class_id is not None else schema.id_of(args.class_name)
    except KeyError as exc:
        LOGGER.error("%s", exc)
        return 2

    try:
        config = load_inference_config(None, weights=args.weights, device=args.device)
        detector = PPEDetector(config)
        report = pseudo_label_dataset(
            args.dataset,
            detector,
            class_name=args.class_name,
            source_class_id=class_id,
            conf=args.conf,
            dry_run=args.dry_run,
        )
    except (PseudoLabelError, PredictionError, ConfigError) as exc:
        LOGGER.error("%s", exc)
        return 2

    if not args.dry_run:
        write_json(resolve_path(args.report), report)
        LOGGER.info("Rapport : %s", args.report)
    LOGGER.info(
        "Termine : %d annotation(s) '%s' ajoutee(s) sur %d image(s).",
        report["totals"].get("added", 0),
        args.class_name,
        report["totals"].get("images", 0),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
