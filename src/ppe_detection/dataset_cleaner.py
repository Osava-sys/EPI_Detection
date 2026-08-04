"""Construction d'un dataset de detection normalise a partir de l'export original.

Le dataset source n'est **jamais** modifie : une copie derivee est produite dans
un repertoire distinct, ne contenant que des lignes YOLO detection a 5 champs.

Usage :

    python -m ppe_detection.dataset_cleaner \\
        --source data.yaml \\
        --output artifacts/dataset_detection \\
        --mode copy

Chaque conversion, correction ou exclusion est journalisee dans un rapport
JSON/Markdown, et l'audit est automatiquement relance sur le dataset derive.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .annotations import (
    DEFAULT_CLAMP_TOLERANCE,
    DEFAULT_MIN_BOX_SIDE,
    IssueCode,
    LineKind,
    format_detection_line,
    parse_label_text,
)
from .config import ConfigError, DatasetConfig, load_dataset_config
from .dataset_audit import roboflow_source_stem, run_audit
from .utils import (
    describe_environment,
    ensure_dir,
    get_logger,
    markdown_table,
    project_root,
    resolve_path,
    setup_logging,
    write_json,
    write_text,
    write_yaml,
)

LOGGER = get_logger(__name__)

MAX_LOG_ENTRIES = 500
"""Nombre maximum d'evenements detailles conserves dans le rapport."""


@dataclass
class CleanStats:
    """Compteurs et journal de la normalisation."""

    images_copied: int = 0
    images_skipped: int = 0
    label_files_written: int = 0
    lines_read: int = 0
    lines_kept: int = 0
    lines_converted: int = 0
    lines_clamped: int = 0
    lines_clipped: int = 0
    lines_dropped: int = 0
    empty_label_files: int = 0
    issue_counts: Counter[str] = field(default_factory=Counter)
    class_counts: Counter[str] = field(default_factory=Counter)
    events: list[dict[str, Any]] = field(default_factory=list)

    def log_event(self, kind: str, split: str, file: str, line: int, detail: str) -> None:
        """Journalise un evenement, en bornant la taille du journal."""
        self.issue_counts[kind] += 1
        if len(self.events) < MAX_LOG_ENTRIES:
            self.events.append(
                {"kind": kind, "split": split, "file": file, "line": line, "detail": detail}
            )


class CleaningError(RuntimeError):
    """Erreur bloquante durant la normalisation."""


# --------------------------------------------------------------------------- #
# Copie des images
# --------------------------------------------------------------------------- #
def _link_or_copy(source: Path, target: Path, mode: str) -> bool:
    """Materialise une image dans le dataset derive.

    Args:
        source: Fichier source.
        target: Destination.
        mode: ``"copy"`` (copie binaire) ou ``"symlink"`` (lien symbolique, avec
            repli automatique sur la copie si le systeme le refuse — sous
            Windows, un lien symbolique exige le mode developpeur ou des droits
            administrateur).

    Returns:
        True si le fichier a ete cree, False s'il existait deja.
    """
    if target.exists() or target.is_symlink():
        return False
    ensure_dir(target.parent)
    if mode == "symlink":
        try:
            target.symlink_to(source)
            return True
        except (OSError, NotImplementedError) as exc:
            LOGGER.debug("Lien symbolique impossible (%s) — copie utilisee.", exc)
    shutil.copy2(source, target)
    return True


# --------------------------------------------------------------------------- #
# Regroupement anti-fuite
# --------------------------------------------------------------------------- #
def plan_split_assignment(
    dataset: DatasetConfig, *, regroup: bool, seed: int = 42
) -> tuple[dict[str, str], dict[str, Any]]:
    """Determine le split de destination de chaque image.

    Par defaut la repartition d'origine est conservee a l'identique. Avec
    ``regroup=True``, toutes les images partageant la meme image source Roboflow
    (prefixe avant ``.rf.``) sont regroupees dans un seul split, ce qui supprime
    la fuite constatee lors de l'audit.

    Le split retenu pour un groupe est celui ou reside deja la majorite de ses
    fichiers; en cas d'egalite, un hachage stable du prefixe tranche de maniere
    deterministe (donc reproductible d'une execution a l'autre).

    Args:
        dataset: Dataset source.
        regroup: Active le regroupement anti-fuite.
        seed: Graine melangee au hachage pour departager les egalites.

    Returns:
        ``({"split/nom_fichier": split_cible}, rapport)``.
    """
    assignment: dict[str, str] = {}
    stems: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for split, images_dir in dataset.splits.items():
        for path in sorted(images_dir.iterdir()):
            if not path.is_file():
                continue
            key = f"{split}/{path.name}"
            assignment[key] = split
            stems[roboflow_source_stem(path.name)].append((split, path.name))

    report: dict[str, Any] = {
        "regroup_applied": regroup,
        "groups_spanning_splits": 0,
        "files_reassigned": 0,
        "moves": [],
    }

    multi = {stem: items for stem, items in stems.items() if len({s for s, _ in items}) > 1}
    report["groups_spanning_splits"] = len(multi)

    if not regroup:
        return assignment, report

    for stem, items in sorted(multi.items()):
        counts = Counter(split for split, _ in items)
        best = max(counts.values())
        contenders = sorted(split for split, count in counts.items() if count == best)
        if len(contenders) == 1:
            target = contenders[0]
        else:
            digest = hashlib.sha256(f"{seed}:{stem}".encode()).digest()
            target = contenders[digest[0] % len(contenders)]
        for split, name in items:
            if split != target:
                assignment[f"{split}/{name}"] = target
                report["files_reassigned"] += 1
                if len(report["moves"]) < MAX_LOG_ENTRIES:
                    report["moves"].append(
                        {"file": name, "from": split, "to": target, "source_stem": stem}
                    )
    return assignment, report


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def clean_dataset(
    source_yaml: str | Path,
    output_dir: str | Path,
    *,
    mode: str = "copy",
    overwrite: bool = False,
    dry_run: bool = False,
    strict: bool = False,
    regroup_by_source: bool = True,
    clamp_tolerance: float = DEFAULT_CLAMP_TOLERANCE,
    min_box_side: float = DEFAULT_MIN_BOX_SIDE,
    seed: int = 42,
) -> dict[str, Any]:
    """Construit le dataset de detection derive.

    Args:
        source_yaml: ``data.yaml`` du dataset original (jamais modifie).
        output_dir: Repertoire du dataset derive.
        mode: ``"copy"`` ou ``"symlink"`` pour les images.
        overwrite: Autorise l'ecrasement d'un dataset derive existant.
        dry_run: Analyse sans rien ecrire sur disque.
        strict: Echoue si au moins une ligne doit etre exclue.
        regroup_by_source: Regroupe les variantes d'une meme image source dans
            un seul split (supprime la fuite inter-splits).
        clamp_tolerance: Tolerance de derive numerique.
        min_box_side: Cote normalise minimal d'une boite.
        seed: Graine pour le regroupement deterministe.

    Returns:
        Le rapport de nettoyage.

    Raises:
        CleaningError: Si la destination existe sans ``overwrite``, si le mode
            est inconnu, ou si ``strict`` est actif et des lignes sont exclues.
    """
    if mode not in {"copy", "symlink"}:
        raise CleaningError(f"Mode inconnu : {mode!r}. Utilisez 'copy' ou 'symlink'.")

    dataset = load_dataset_config(source_yaml)
    target_root = resolve_path(output_dir)

    if target_root.exists() and any(target_root.iterdir()):
        if not overwrite and not dry_run:
            raise CleaningError(
                f"Le repertoire de destination existe deja et n'est pas vide : {target_root}\n"
                f"Relancez avec --overwrite pour le remplacer explicitement, "
                f"ou choisissez un autre --output."
            )
        if overwrite and not dry_run:
            LOGGER.warning("--overwrite : suppression du contenu de %s", target_root)
            for split in ("train", "valid", "test"):
                shutil.rmtree(target_root / split, ignore_errors=True)

    # Le dataset derive peut vivre sous la racine du projet (artifacts/...), mais
    # jamais ecraser la racine du dataset source ni l'un de ses splits.
    try:
        resolved_target = target_root.resolve()
        forbidden = {dataset.root.resolve()}
        for images_dir in dataset.splits.values():
            forbidden.add(images_dir.resolve())
            forbidden.add(images_dir.resolve().parent)
        if resolved_target in forbidden:
            raise CleaningError(
                f"La destination ({target_root}) est identique au dataset source. "
                f"Le dataset original doit rester intact : choisissez un autre --output."
            )
    except OSError:  # pragma: no cover - chemins pathologiques Windows
        LOGGER.debug("Verification de non-recouvrement impossible pour %s", target_root)

    assignment, regroup_report = plan_split_assignment(
        dataset, regroup=regroup_by_source, seed=seed
    )

    stats = CleanStats()
    per_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    written_images: dict[str, int] = Counter()

    LOGGER.info(
        "Normalisation de %s -> %s (mode=%s, dry_run=%s, regroup=%s)",
        dataset.yaml_path,
        target_root,
        mode,
        dry_run,
        regroup_by_source,
    )

    for split, images_dir in dataset.splits.items():
        labels_dir = dataset.labels_dir(split)
        image_paths = sorted(p for p in images_dir.iterdir() if p.is_file())
        LOGGER.info("[%s] traitement de %d images...", split, len(image_paths))

        for image_path in image_paths:
            key = f"{split}/{image_path.name}"
            target_split = assignment.get(key, split)
            label_path = labels_dir / f"{image_path.stem}.txt"

            if not label_path.is_file():
                stats.images_skipped += 1
                stats.log_event(
                    "missing_label", split, image_path.name, 0, "aucun fichier de label associe"
                )
                continue

            text = label_path.read_text(encoding="utf-8", errors="replace")
            parsed_lines = parse_label_text(
                text,
                num_classes=dataset.num_classes,
                clamp_tolerance=clamp_tolerance,
                min_box_side=min_box_side,
            )

            kept: list[str] = []
            for line_no, parsed in parsed_lines:
                stats.lines_read += 1

                if parsed.is_error:
                    stats.lines_dropped += 1
                    codes = ",".join(code.value for code in parsed.errors)
                    stats.log_event(
                        f"dropped:{codes}", split, label_path.name, line_no, parsed.detail
                    )
                    continue

                if parsed.box is None or parsed.class_id is None:
                    stats.lines_dropped += 1
                    stats.log_event("dropped:no_box", split, label_path.name, line_no, parsed.detail)
                    continue

                if parsed.kind is LineKind.POLYGON:
                    stats.lines_converted += 1
                    stats.log_event(
                        "converted_polygon",
                        split,
                        label_path.name,
                        line_no,
                        f"{parsed.n_points} sommets -> boite",
                    )
                if IssueCode.CLAMPED_MINOR in parsed.issues:
                    stats.lines_clamped += 1
                if IssueCode.CLIPPED_TO_BOUNDS in parsed.issues:
                    stats.lines_clipped += 1
                    stats.log_event(
                        "clipped_to_bounds", split, label_path.name, line_no, parsed.detail
                    )

                kept.append(format_detection_line(parsed.class_id, parsed.box))
                stats.lines_kept += 1
                class_name = dataset.class_name(parsed.class_id)
                stats.class_counts[class_name] += 1
                per_split_counts[target_split][class_name] += 1

            if not kept:
                stats.empty_label_files += 1

            if not dry_run:
                target_image = target_root / target_split / "images" / image_path.name
                if _link_or_copy(image_path, target_image, mode):
                    stats.images_copied += 1
                target_label = target_root / target_split / "labels" / f"{image_path.stem}.txt"
                ensure_dir(target_label.parent)
                target_label.write_text(
                    ("\n".join(kept) + "\n") if kept else "", encoding="utf-8"
                )
                stats.label_files_written += 1
            else:
                stats.images_copied += 1
                stats.label_files_written += 1
            written_images[target_split] += 1

    data_yaml_path = target_root / "data.yaml"
    if not dry_run:
        write_yaml(
            data_yaml_path,
            {
                "path": str(target_root),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "nc": dataset.num_classes,
                "names": list(dataset.names),
            },
        )
        LOGGER.info("data.yaml derive ecrit : %s", data_yaml_path)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": describe_environment(),
        "source": {
            "data_yaml": str(dataset.yaml_path),
            "root": str(dataset.root),
            "splits": {name: str(path) for name, path in dataset.splits.items()},
        },
        "output": {
            "root": str(target_root),
            "data_yaml": str(data_yaml_path),
            "mode": mode,
            "dry_run": dry_run,
            "images_per_split": dict(written_images),
        },
        "parameters": {
            "clamp_tolerance": clamp_tolerance,
            "min_box_side": min_box_side,
            "strict": strict,
            "regroup_by_source": regroup_by_source,
            "seed": seed,
        },
        "statistics": {
            "images_copied": stats.images_copied,
            "images_skipped_no_label": stats.images_skipped,
            "label_files_written": stats.label_files_written,
            "lines_read": stats.lines_read,
            "lines_kept": stats.lines_kept,
            "lines_converted_from_polygon": stats.lines_converted,
            "lines_clamped_minor": stats.lines_clamped,
            "lines_clipped_to_bounds": stats.lines_clipped,
            "lines_dropped": stats.lines_dropped,
            "empty_label_files": stats.empty_label_files,
        },
        "class_counts": dict(stats.class_counts),
        "class_counts_per_split": {s: dict(c) for s, c in per_split_counts.items()},
        "issue_counts": dict(stats.issue_counts),
        "events": stats.events,
        "regrouping": regroup_report,
    }

    if strict and stats.lines_dropped:
        raise CleaningError(
            f"Mode --strict : {stats.lines_dropped} ligne(s) ont du etre exclues. "
            f"Consultez le rapport de nettoyage pour le detail des lignes concernees."
        )
    return report


def render_clean_markdown(report: dict[str, Any]) -> str:
    """Genere le rapport Markdown de la normalisation."""
    stats = report["statistics"]
    lines: list[str] = []
    lines.append("# Rapport de normalisation du dataset (detection)")
    lines.append("")
    lines.append(f"- **Genere le** : {report['generated_at']}")
    lines.append(f"- **Source** : `{report['source']['data_yaml']}` (non modifiee)")
    lines.append(f"- **Destination** : `{report['output']['root']}`")
    lines.append(f"- **Mode de copie** : {report['output']['mode']}")
    lines.append(f"- **Simulation (dry-run)** : {'oui' if report['output']['dry_run'] else 'non'}")
    lines.append("")

    lines.append("## Statistiques de conversion")
    lines.append("")
    lines.append(
        markdown_table(
            ["Indicateur", "Valeur"],
            [
                ["Images ecrites", stats["images_copied"]],
                ["Images ignorees (label absent)", stats["images_skipped_no_label"]],
                ["Fichiers de labels ecrits", stats["label_files_written"]],
                ["Lignes lues", stats["lines_read"]],
                ["Lignes conservees", stats["lines_kept"]],
                ["**Lignes converties depuis un polygone**", stats["lines_converted_from_polygon"]],
                ["Lignes avec derive numerique corrigee", stats["lines_clamped_minor"]],
                ["Boites rognees sur les bords de l'image", stats["lines_clipped_to_bounds"]],
                ["Lignes exclues", stats["lines_dropped"]],
                ["Fichiers de labels vides (negatifs)", stats["empty_label_files"]],
            ],
        )
    )
    lines.append("")

    lines.append("## Repartition des images produites")
    lines.append("")
    lines.append(
        markdown_table(
            ["Split", "Images"],
            [[split, count] for split, count in sorted(report["output"]["images_per_split"].items())],
        )
    )
    lines.append("")

    lines.append("## Instances par classe et par split")
    lines.append("")
    per_split = report["class_counts_per_split"]
    splits = sorted(per_split)
    class_names = sorted(report["class_counts"])
    if class_names:
        rows = [
            [name, *[per_split.get(s, {}).get(name, 0) for s in splits], report["class_counts"][name]]
            for name in class_names
        ]
        lines.append(markdown_table(["Classe", *splits, "Total"], rows))
    lines.append("")

    regrouping = report.get("regrouping", {})
    lines.append("## Regroupement anti-fuite")
    lines.append("")
    if regrouping.get("regroup_applied"):
        lines.append(
            f"Le regroupement par image source a ete applique : "
            f"**{regrouping['files_reassigned']} fichier(s)** deplace(s) pour que toutes les "
            f"variantes d'une meme photo source restent dans un seul split "
            f"({regrouping['groups_spanning_splits']} groupe(s) concerne(s))."
        )
    else:
        lines.append(
            f"Le regroupement n'a **pas** ete applique : la repartition train/valid/test d'origine "
            f"est conservee a l'identique. L'audit a identifie "
            f"**{regrouping.get('groups_spanning_splits', 0)} groupe(s)** d'images partageant la meme "
            f"image source repartis entre plusieurs splits. Relancez avec `--regroup-by-source` "
            f"pour supprimer cette fuite."
        )
    lines.append("")

    if report["issue_counts"]:
        issue_rows: list[tuple[str, int]] = sorted(
            report["issue_counts"].items(), key=lambda kv: -int(kv[1])
        )
        lines.append("## Evenements par type")
        lines.append("")
        lines.append(markdown_table(["Evenement", "Occurrences"], issue_rows))
        lines.append("")

    if report["events"]:
        lines.append(f"## Journal detaille (premiers {len(report['events'])} evenements)")
        lines.append("")
        lines.append(
            markdown_table(
                ["Type", "Split", "Fichier", "Ligne", "Detail"],
                [
                    [e["kind"], e["split"], e["file"], e["line"], e["detail"][:70]]
                    for e in report["events"][:120]
                ],
            )
        )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments de la commande de nettoyage."""
    parser = argparse.ArgumentParser(
        prog="python -m ppe_detection.dataset_cleaner",
        description=(
            "Construit un dataset de detection normalise (labels YOLO a 5 champs) "
            "a partir de l'export original, sans jamais modifier ce dernier."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default="data.yaml", help="data.yaml du dataset original.")
    parser.add_argument(
        "--output", default="artifacts/dataset_detection", help="Repertoire du dataset derive."
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Copie binaire des images ou lien symbolique (economise l'espace disque).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Remplace explicitement un dataset derive existant."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Analyse et rapporte sans rien ecrire sur disque."
    )
    parser.add_argument(
        "--strict", action="store_true", help="Echoue si au moins une ligne doit etre exclue."
    )
    # Le regroupement est ACTIF PAR DEFAUT : sans lui, l'export Roboflow place
    # des variantes augmentees d'une meme photo dans train et dans test, ce qui
    # gonfle artificiellement les metriques (mesure : +0.033 de mAP@0.50).
    # Le drapeau historique reste accepte pour ne pas casser les commandes
    # existantes, mais il n'a plus d'effet puisqu'il decrit le defaut.
    parser.add_argument(
        "--regroup-by-source",
        action="store_true",
        help="Sans effet : le regroupement anti-fuite est desormais applique par defaut.",
    )
    parser.add_argument(
        "--allow-source-leak",
        action="store_true",
        help=(
            "Desactive le regroupement anti-fuite et reproduit la repartition "
            "train/valid/test exacte de l'export Roboflow. A n'utiliser que pour "
            "comparer des resultats a ceux publies sur le decoupage d'origine : "
            "les metriques obtenues seront optimistes."
        ),
    )
    parser.add_argument(
        "--clamp-tolerance",
        type=float,
        default=DEFAULT_CLAMP_TOLERANCE,
        help="Excursion hors [0,1] consideree comme une derive numerique.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Graine du regroupement deterministe.")
    parser.add_argument(
        "--report",
        default="artifacts/reports/dataset_cleaning.json",
        help="Chemin du rapport de nettoyage JSON.",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="N'execute pas l'audit de verification sur le dataset derive.",
    )
    parser.add_argument("--log-level", default="INFO", help="Niveau de log.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entree CLI. Retourne le code de sortie du processus."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=project_root() / "artifacts" / "logs" / "cleaning.log")

    try:
        report = clean_dataset(
            args.source,
            args.output,
            mode=args.mode,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            strict=args.strict,
            regroup_by_source=not args.allow_source_leak,
            clamp_tolerance=args.clamp_tolerance,
            seed=args.seed,
        )
    except (CleaningError, ConfigError, FileNotFoundError) as exc:
        LOGGER.error("Normalisation impossible : %s", exc)
        return 2

    report_path = resolve_path(args.report)
    write_json(report_path, report)
    write_text(report_path.with_suffix(".md"), render_clean_markdown(report))
    LOGGER.info("Rapport de nettoyage : %s", report_path)

    stats = report["statistics"]
    LOGGER.info(
        "Termine : %d lignes conservees, %d converties depuis un polygone, %d exclues.",
        stats["lines_kept"],
        stats["lines_converted_from_polygon"],
        stats["lines_dropped"],
    )

    if args.dry_run:
        LOGGER.info("Mode --dry-run : aucun fichier n'a ete ecrit.")
        return 0

    if args.no_audit:
        return 0

    # Verification : le dataset derive ne doit plus contenir aucun polygone.
    LOGGER.info("Audit de verification du dataset derive...")
    audit_report = run_audit(
        report["output"]["data_yaml"],
        resolve_path("artifacts/reports/dataset_audit_detection.json"),
        workers=8,
        compute_phash=False,
        generate_plots=True,
    )
    totals = audit_report["totals"]
    if totals["polygon_lines"] or totals["malformed_lines"]:
        LOGGER.error(
            "Le dataset derive contient encore %d ligne(s) polygonale(s) et %d ligne(s) malformee(s).",
            totals["polygon_lines"],
            totals["malformed_lines"],
        )
        return 1
    if audit_report["conclusion"]["n_errors"]:
        LOGGER.error(
            "L'audit du dataset derive signale %d erreur(s) bloquante(s).",
            audit_report["conclusion"]["n_errors"],
        )
        return 1
    LOGGER.info(
        "Verification reussie : %d annotations, 100 %% au format detection a 5 champs.",
        totals["annotations"],
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
