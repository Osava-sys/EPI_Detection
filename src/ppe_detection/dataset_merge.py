"""Fusion de datasets YOLO externes avec remappage de classes.

Pourquoi cet outil
==================
Aucun dataset public ne separe casque de chantier et casque de velo : la
recherche sur Roboflow Universe et Open Images le confirme. En revanche, les
briques existent separement — un dataset de casques de chantier ici, un dataset
de casques de velo la, les couvre-chefs d'Open Images ailleurs.

Les assembler suppose de resoudre trois problemes :

1. **Les identifiants de classes different.** La classe 0 d'un dataset de velo
   n'a rien a voir avec la classe 0 du votre. Il faut traduire.
2. **Certaines classes source n'ont pas d'equivalent** et doivent etre
   ignorees, sans pour autant jeter l'image : une image de cycliste garde son
   interet meme si la classe « velo » ne nous concerne pas.
3. **Les noms de fichiers peuvent entrer en collision** entre sources.

Ce module fait les trois, sans jamais modifier les datasets sources.

Images negatives
================
Une image dont toutes les annotations ont ete ecartees devient une **image
negative** : aucun objet a detecter. Loin d'etre inutile, elle apprend au modele
a ne rien signaler dans ce contexte, ce qui reduit les faux positifs. YOLO
accepte un fichier d'annotations vide. L'option ``--keep-empty`` controle ce
comportement, actif par defaut.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ConfigError
from .taxonomy import ClassSchema, build_class_mapping, extended_schema
from .utils import (
    IMAGE_EXTENSIONS,
    ensure_dir,
    get_logger,
    read_yaml,
    resolve_path,
    setup_logging,
    write_json,
    write_text,
    write_yaml,
)

LOGGER = get_logger(__name__)

SPLITS: tuple[str, ...] = ("train", "valid", "test")

# Les datasets YOLO circulent sous deux dispositions, et les exports FiftyOne
# emploient la seconde. Un outil de fusion qui n'en gere qu'une est inutilisable
# sur la moitie des sources publiques.
SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "valid": ("valid", "val", "validation"),
    "test": ("test",),
}


def resolve_split_dirs(root: Path, split: str) -> tuple[Path, Path] | None:
    """Localise les repertoires d'images et d'annotations d'un split.

    Essaie les deux dispositions courantes et les alias de nommage :

    * ``<racine>/<split>/images`` et ``<racine>/<split>/labels`` (Roboflow) ;
    * ``<racine>/images/<split>`` et ``<racine>/labels/<split>`` (FiftyOne, YOLOv5).

    Args:
        root: Racine du dataset (repertoire contenant ``data.yaml``).
        split: Split canonique (``train``, ``valid`` ou ``test``).

    Returns:
        ``(images, labels)``, ou ``None`` si le split est absent.
    """
    for alias in SPLIT_ALIASES.get(split, (split,)):
        candidates = (
            (root / alias / "images", root / alias / "labels"),
            (root / "images" / alias, root / "labels" / alias),
        )
        for images, labels in candidates:
            if images.is_dir():
                return images, labels
    return None


class MergeError(RuntimeError):
    """Erreur de fusion, avec message actionnable."""


@dataclass
class SourceStats:
    """Bilan d'une source fusionnee."""

    name: str
    images_copied: int = 0
    images_skipped: int = 0
    labels_written: int = 0
    annotations_kept: int = 0
    annotations_dropped: int = 0
    empty_labels: int = 0
    per_class: Counter[str] = field(default_factory=Counter)
    dropped_classes: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable."""
        return {
            "name": self.name,
            "images_copied": self.images_copied,
            "images_skipped": self.images_skipped,
            "labels_written": self.labels_written,
            "annotations_kept": self.annotations_kept,
            "annotations_dropped": self.annotations_dropped,
            "empty_labels": self.empty_labels,
            "per_class": dict(self.per_class.most_common()),
            "dropped_classes": dict(self.dropped_classes.most_common()),
        }


def _read_source_names(data_yaml: Path) -> list[str]:
    """Extrait la liste ordonnee des classes d'un ``data.yaml``."""
    raw = read_yaml(data_yaml)
    names = raw.get("names")
    if isinstance(names, dict):
        # Format {0: 'a', 1: 'b'} : l'ordre des cles fait foi.
        return [str(names[key]) for key in sorted(names, key=lambda k: int(k))]
    if isinstance(names, list):
        return [str(n) for n in names]
    raise MergeError(
        f"Impossible de lire les classes depuis {data_yaml}. "
        f"Attendu une cle 'names' contenant une liste ou un dictionnaire."
    )


def _unique_stem(source_name: str, original: Path) -> str:
    """Prefixe deterministe evitant les collisions entre sources.

    Le hachage court du chemin d'origine garantit qu'une meme image importee
    deux fois produit le meme nom, ce qui rend la fusion idempotente.
    """
    digest = hashlib.sha1(str(original).encode("utf-8")).hexdigest()[:8]
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in source_name)[:24]
    return f"{slug}__{original.stem}__{digest}"


def _remap_label_file(
    label_path: Path, mapping: dict[int, int], schema: ClassSchema, stats: SourceStats
) -> list[str] | None:
    """Traduit un fichier d'annotations YOLO vers le schema cible.

    Returns:
        Les lignes traduites, ou ``None`` si le fichier est illisible.
    """
    try:
        content = label_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        LOGGER.warning("Annotations illisibles, ignorees (%s) : %s", label_path, exc)
        return None

    output: list[str] = []
    for line in content.splitlines():
        parts = line.split()
        if len(parts) != 5:
            # Les polygones ne sont pas geres ici : passer d'abord le dataset
            # source par dataset_cleaner, qui les convertit en boites.
            stats.annotations_dropped += 1
            continue
        try:
            source_id = int(parts[0])
        except ValueError:
            stats.annotations_dropped += 1
            continue

        if source_id not in mapping:
            stats.annotations_dropped += 1
            stats.dropped_classes[str(source_id)] += 1
            continue

        target_id = mapping[source_id]
        stats.annotations_kept += 1
        stats.per_class[schema.name_of(target_id)] += 1
        output.append(f"{target_id} {' '.join(parts[1:])}")
    return output


def merge_dataset(
    source_yaml: str | Path,
    target_dir: str | Path,
    class_mapping: dict[str, str],
    *,
    source_name: str | None = None,
    schema: ClassSchema | None = None,
    splits: tuple[str, ...] = SPLITS,
    keep_empty: bool = True,
    dry_run: bool = False,
) -> SourceStats:
    """Fusionne un dataset YOLO externe dans un dataset cible.

    Args:
        source_yaml: ``data.yaml`` du dataset a importer.
        target_dir: Dataset de destination (cree si absent).
        class_mapping: ``{nom_source: nom_cible}``. Les classes absentes de ce
            dictionnaire sont ignorees.
        source_name: Etiquette de la source, utilisee pour prefixer les fichiers.
        schema: Schema cible (par defaut, le schema etendu a 10 classes).
        splits: Splits a traiter.
        keep_empty: Conserve les images dont toutes les annotations ont ete
            ecartees, sous forme d'images negatives.
        dry_run: Analyse sans rien ecrire.

    Returns:
        Le bilan de la fusion.

    Raises:
        MergeError: Si le dataset source est illisible ou le mapping invalide.
    """
    source_path = resolve_path(source_yaml)
    if not source_path.is_file():
        raise MergeError(f"data.yaml introuvable : {source_path}")

    target = resolve_path(target_dir)
    active_schema = schema or extended_schema()
    label = source_name or source_path.parent.name
    stats = SourceStats(name=label)

    source_names = _read_source_names(source_path)
    unknown = [n for n in class_mapping if n not in source_names]
    if unknown:
        raise MergeError(
            f"Classes absentes du dataset source : {unknown}.\n"
            f"Classes disponibles : {', '.join(source_names)}"
        )

    mapping = build_class_mapping(source_names, active_schema, class_mapping)
    if not mapping:
        raise MergeError(
            "Le mapping ne conserve aucune classe. Verifiez les noms : "
            f"source={source_names}, mapping={class_mapping}"
        )

    LOGGER.info(
        "Fusion de '%s' : %d classe(s) conservee(s) sur %d.",
        label,
        len(mapping),
        len(source_names),
    )

    root = source_path.parent
    for split in splits:
        resolved = resolve_split_dirs(root, split)
        if resolved is None:
            LOGGER.debug("Split absent de la source, ignore : %s", split)
            continue
        images_dir, labels_dir = resolved

        out_images = target / split / "images"
        out_labels = target / split / "labels"
        if not dry_run:
            ensure_dir(out_images)
            ensure_dir(out_labels)

        for image in sorted(images_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_file = labels_dir / f"{image.stem}.txt"
            lines: list[str] = []
            if label_file.is_file():
                remapped = _remap_label_file(label_file, mapping, active_schema, stats)
                if remapped is None:
                    stats.images_skipped += 1
                    continue
                lines = remapped

            if not lines and not keep_empty:
                stats.images_skipped += 1
                continue
            if not lines:
                stats.empty_labels += 1

            stem = _unique_stem(label, image)
            if not dry_run:
                shutil.copy2(image, out_images / f"{stem}{image.suffix.lower()}")
                (out_labels / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
            stats.images_copied += 1
            stats.labels_written += 1

    return stats


def write_target_yaml(target_dir: str | Path, schema: ClassSchema) -> Path:
    """Ecrit le ``data.yaml`` du dataset fusionne."""
    target = resolve_path(target_dir)
    payload: dict[str, Any] = {
        "path": str(target),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        **schema.to_data_yaml(),
    }
    destination = target / "data.yaml"
    write_yaml(destination, payload)
    return destination


def render_merge_report(stats: list[SourceStats], schema: ClassSchema) -> str:
    """Rapport Markdown recapitulant la fusion."""
    total_images = sum(s.images_copied for s in stats)
    total_kept = sum(s.annotations_kept for s in stats)
    total_dropped = sum(s.annotations_dropped for s in stats)

    lines = [
        "# Rapport de fusion de datasets",
        "",
        f"- **Sources fusionnees** : {len(stats)}",
        f"- **Images**  : {total_images}",
        f"- **Annotations conservees** : {total_kept}",
        f"- **Annotations ecartees**   : {total_dropped}",
        f"- **Schema cible** : {schema.size} classes",
        "",
        "## Par source",
        "",
        "| Source | Images | Annotations gardees | Ecartees | Images negatives |",
        "|--------|--------|---------------------|----------|------------------|",
    ]
    for item in stats:
        lines.append(
            f"| {item.name} | {item.images_copied} | {item.annotations_kept} "
            f"| {item.annotations_dropped} | {item.empty_labels} |"
        )

    combined: Counter[str] = Counter()
    for item in stats:
        combined.update(item.per_class)

    lines += [
        "",
        "## Instances par classe",
        "",
        "| Classe | Instances | Part |",
        "|--------|-----------|------|",
    ]
    grand_total = sum(combined.values()) or 1
    for name in schema.names:
        count = combined.get(name, 0)
        lines.append(f"| {name} | {count} | {count / grand_total:.1%} |")

    absent = [n for n in schema.names if combined.get(n, 0) == 0]
    if absent:
        lines += [
            "",
            "> **Classes sans aucune instance** : " + ", ".join(absent) + ".",
            "> Un modele entraine ainsi ne pourra jamais les predire.",
        ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m ppe_detection.dataset_merge",
        description="Fusionne un dataset YOLO externe en remappant ses classes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemple :\n"
            "  python -m ppe_detection.dataset_merge \\\n"
            "    --source datasets/bike-helmets/data.yaml \\\n"
            "    --target artifacts/dataset_extended \\\n"
            '    --map "With Helmet=Non-Safety Headwear" \\\n'
            "    --name bike-helmets\n"
        ),
    )
    parser.add_argument("--source", required=True, help="data.yaml du dataset a importer.")
    parser.add_argument("--target", required=True, help="Dataset de destination.")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="SOURCE=CIBLE",
        help="Correspondance de classes, repetable. Les classes non citees sont ignorees.",
    )
    parser.add_argument("--name", default=None, help="Etiquette de la source.")
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Ecarte les images sans annotation retenue (conservees par defaut comme negatives).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyse sans rien ecrire.")
    parser.add_argument(
        "--report", default="artifacts/reports/dataset_merge.md", help="Rapport Markdown."
    )
    parser.add_argument("--log-level", default="INFO", help="Niveau de log.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entree CLI."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    mapping: dict[str, str] = {}
    for item in args.map:
        if "=" not in item:
            LOGGER.error("Correspondance invalide : %r. Attendu SOURCE=CIBLE.", item)
            return 2
        source_name, target_name = item.split("=", 1)
        mapping[source_name.strip()] = target_name.strip()

    if not mapping:
        LOGGER.error("Aucune correspondance fournie. Utilisez --map \"SOURCE=CIBLE\".")
        return 2

    schema = extended_schema()
    try:
        stats = merge_dataset(
            args.source,
            args.target,
            mapping,
            source_name=args.name,
            schema=schema,
            keep_empty=not args.drop_empty,
            dry_run=args.dry_run,
        )
    except (MergeError, ConfigError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        return 2

    if not args.dry_run:
        yaml_path = write_target_yaml(args.target, schema)
        LOGGER.info("data.yaml ecrit : %s", yaml_path)
        report = render_merge_report([stats], schema)
        write_text(resolve_path(args.report), report)
        write_json(resolve_path(args.report).with_suffix(".json"), stats.to_dict())
        LOGGER.info("Rapport : %s", args.report)

    LOGGER.info(
        "%s : %d image(s), %d annotation(s) conservee(s), %d ecartee(s), %d negative(s).",
        stats.name,
        stats.images_copied,
        stats.annotations_kept,
        stats.annotations_dropped,
        stats.empty_labels,
    )
    for name, count in stats.per_class.most_common():
        LOGGER.info("   %-24s %d", name, count)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
