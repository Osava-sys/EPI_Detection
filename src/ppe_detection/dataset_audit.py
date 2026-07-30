"""Audit complet et reproductible d'un dataset de detection YOLO.

Usage :

    python -m ppe_detection.dataset_audit --data data.yaml \\
        --output artifacts/reports/dataset_audit.json

Produit un rapport JSON exploitable par une machine, un rapport Markdown
lisible, des graphiques de distribution et des exemples d'annotations rendus.

Un point important : une ligne polygonale n'est **pas** comptee comme invalide.
Elle est identifiee comme une annotation de segmentation qui doit etre convertie
en boite englobante pour une tache de detection.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .annotations import (
    DEFAULT_CLAMP_TOLERANCE,
    DEFAULT_MIN_BOX_SIDE,
    IssueCode,
    LineKind,
    ParsedLine,
    parse_label_text,
)
from .config import ConfigError, DatasetConfig, load_dataset_config
from .utils import (
    IMAGE_EXTENSIONS,
    describe_environment,
    ensure_dir,
    get_logger,
    human_bytes,
    markdown_table,
    project_root,
    resolve_path,
    setup_logging,
    write_json,
    write_text,
)
from .visualization import plot_box_size_distribution, plot_class_distribution

LOGGER = get_logger(__name__)

MAX_EXAMPLES = 25
"""Nombre maximum d'exemples conserves par categorie de probleme."""

ROBOFLOW_MARKER = ".rf."
"""Separateur insere par Roboflow entre le nom source et le hash d'export."""


# --------------------------------------------------------------------------- #
# Structures de donnees
# --------------------------------------------------------------------------- #
@dataclass
class ImageInfo:
    """Metadonnees d'un fichier image."""

    path: Path
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    sha256: str = ""
    phash: str | None = None
    readable: bool = True
    error: str = ""

    @property
    def aspect_ratio(self) -> float:
        """Rapport largeur/hauteur (0.0 si la hauteur est inconnue)."""
        return self.width / self.height if self.height else 0.0


@dataclass
class SplitAudit:
    """Resultat de l'audit d'un split."""

    name: str
    images_dir: Path
    labels_dir: Path
    n_images: int = 0
    n_labels: int = 0
    n_annotations: int = 0
    n_detection_lines: int = 0
    n_polygon_lines: int = 0
    n_malformed_lines: int = 0
    n_empty_label_files: int = 0
    n_images_without_objects: int = 0
    class_counts: Counter[str] = field(default_factory=Counter)
    issue_counts: Counter[str] = field(default_factory=Counter)
    extension_counts: Counter[str] = field(default_factory=Counter)
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    unreadable_images: list[dict[str, str]] = field(default_factory=list)
    problems: list[dict[str, Any]] = field(default_factory=list)
    box_areas: list[float] = field(default_factory=list)
    dimensions: Counter[str] = field(default_factory=Counter)
    min_dimension: tuple[int, int] | None = None
    max_dimension: tuple[int, int] | None = None
    aspect_ratio_min: float = 0.0
    aspect_ratio_max: float = 0.0
    total_bytes: int = 0

    def add_problem(self, category: str, path: Path, line_no: int, detail: str) -> None:
        """Enregistre un probleme, en bornant le nombre d'exemples conserves."""
        self.issue_counts[category] += 1
        if sum(1 for p in self.problems if p["category"] == category) < MAX_EXAMPLES:
            self.problems.append(
                {
                    "category": category,
                    "file": path.name,
                    "line": line_no,
                    "detail": detail,
                }
            )


# --------------------------------------------------------------------------- #
# Lecture des images
# --------------------------------------------------------------------------- #
def _sha256_of(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Hash SHA-256 du contenu binaire d'un fichier."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path, *, compute_phash: bool = True) -> ImageInfo:
    """Ouvre une image, verifie son integrite et calcule ses empreintes.

    Args:
        path: Chemin de l'image.
        compute_phash: Calcule le hash perceptuel (necessite ``imagehash``).

    Returns:
        Un :class:`ImageInfo`; ``readable=False`` si l'image est illisible ou
        corrompue (aucune exception n'est propagee).
    """
    info = ImageInfo(path=path)
    try:
        info.size_bytes = path.stat().st_size
        info.sha256 = _sha256_of(path)
    except OSError as exc:
        info.readable = False
        info.error = f"lecture impossible : {exc}"
        return info

    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()  # detecte les fichiers tronques/corrompus
        with Image.open(path) as image:
            info.width, info.height = image.size
            if compute_phash:
                try:
                    import imagehash

                    info.phash = str(imagehash.phash(image.convert("RGB")))
                except ImportError:
                    info.phash = None
                except (OSError, ValueError) as exc:  # decodage partiel
                    LOGGER.debug("phash impossible pour %s : %s", path.name, exc)
                    info.phash = None
    except Exception as exc:  # noqa: BLE001 - PIL leve des types tres varies
        info.readable = False
        info.error = f"{type(exc).__name__}: {exc}"
    return info


def scan_images(paths: Sequence[Path], *, workers: int, compute_phash: bool) -> list[ImageInfo]:
    """Inspecte un lot d'images en parallele (I/O + decodage)."""
    if not paths:
        return []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda p: inspect_image(p, compute_phash=compute_phash), paths))


# --------------------------------------------------------------------------- #
# Detection de doublons
# --------------------------------------------------------------------------- #
def _hex_to_int(value: str) -> int:
    """Convertit un hash hexadecimal en entier."""
    return int(value, 16)


def find_near_duplicates(
    entries: Sequence[tuple[str, str]],
    *,
    max_distance: int = 3,
    hash_bits: int = 64,
) -> list[tuple[str, str, int]]:
    """Recherche les paires d'images perceptuellement proches.

    Utilise le *multi-index hashing* : le hash est decoupe en ``max_distance+1``
    bandes. Par le principe des tiroirs, deux hashes a distance de Hamming
    <= ``max_distance`` partagent forcement au moins une bande identique, ce qui
    evite la comparaison quadratique de toutes les paires.

    Args:
        entries: Paires ``(identifiant, hash_hexadecimal)``.
        max_distance: Distance de Hamming maximale pour declarer un doublon.
        hash_bits: Taille du hash en bits (64 pour ``imagehash.phash``).

    Returns:
        Liste de ``(id_a, id_b, distance)`` triee par distance croissante.
    """
    if not entries or max_distance < 0:
        return []

    n_bands = max_distance + 1
    band_bits = max(1, hash_bits // n_bands)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    values: list[int] = []

    for index, (_, hex_hash) in enumerate(entries):
        try:
            value = _hex_to_int(hex_hash)
        except ValueError:
            values.append(-1)
            continue
        values.append(value)
        for band in range(n_bands):
            shift = band * band_bits
            key = (band, (value >> shift) & ((1 << band_bits) - 1))
            buckets[key].append(index)

    seen: set[tuple[int, int]] = set()
    results: list[tuple[str, str, int]] = []
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        for position, first in enumerate(indices):
            for second in indices[position + 1 :]:
                pair = (first, second) if first < second else (second, first)
                if pair in seen:
                    continue
                seen.add(pair)
                if values[first] < 0 or values[second] < 0:
                    continue
                distance = bin(values[first] ^ values[second]).count("1")
                if distance <= max_distance:
                    results.append((entries[first][0], entries[second][0], distance))
    results.sort(key=lambda item: item[2])
    return results


class _UnionFind:
    """Union-find minimaliste pour regrouper les images en clusters de doublons."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        """Retourne le representant du groupe de ``item`` (avec compression de chemin)."""
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, first: str, second: str) -> None:
        """Fusionne les groupes de ``first`` et ``second``."""
        root_a, root_b = self.find(first), self.find(second)
        if root_a != root_b:
            self._parent[root_b] = root_a

    def groups(self) -> list[list[str]]:
        """Retourne les clusters de taille >= 2, tries par taille decroissante."""
        clusters: dict[str, list[str]] = defaultdict(list)
        for item in self._parent:
            clusters[self.find(item)].append(item)
        result = [sorted(members) for members in clusters.values() if len(members) > 1]
        result.sort(key=len, reverse=True)
        return result


def cluster_pairs(pairs: Iterable[tuple[str, str, int]]) -> list[list[str]]:
    """Regroupe des paires de quasi-doublons en composantes connexes.

    Le nombre de *paires* est un indicateur trompeur : un groupe de N images
    quasi identiques genere N*(N-1)/2 paires. Les clusters refletent bien mieux
    la realite (par exemple une sequence de frames video).
    """
    union_find = _UnionFind()
    for first, second, _ in pairs:
        union_find.union(first, second)
    return union_find.groups()


SEQUENCE_PATTERN = re.compile(
    r"^(?P<prefix>.*?)(?P<num>\d{3,})_(?:jpg|jpeg|png|bmp|webp)\.rf\.", re.IGNORECASE
)
"""Reconnait les noms de type ``frame_000324_jpg.rf.<hash>.jpg`` (sequences video)."""


def sequence_prefix(filename: str) -> str | None:
    """Retourne le prefixe de sequence d'un nom de fichier, ou ``None``.

    Les datasets construits a partir de videos contiennent des series
    ``frame_000324``, ``frame_000325``... dont les images consecutives sont
    quasi identiques. Repartir une telle serie entre plusieurs splits constitue
    une fuite de donnees.
    """
    match = SEQUENCE_PATTERN.match(filename)
    if match and match.group("prefix"):
        return match.group("prefix")
    return None


def roboflow_source_stem(filename: str) -> str:
    """Extrait le nom de l'image source d'un fichier exporte par Roboflow.

    Roboflow renomme ``photo.jpg`` en ``photo_jpg.rf.<hash>.jpg``. Deux fichiers
    partageant le meme prefixe proviennent donc de la **meme image source** : les
    retrouver dans deux splits differents constitue une fuite de donnees.

    Args:
        filename: Nom de fichier exporte.

    Returns:
        Le prefixe source, ou le nom sans extension si le marqueur est absent.
    """
    if ROBOFLOW_MARKER in filename:
        return filename.split(ROBOFLOW_MARKER, 1)[0]
    return Path(filename).stem


# --------------------------------------------------------------------------- #
# Audit d'un split
# --------------------------------------------------------------------------- #
def audit_split(
    dataset: DatasetConfig,
    split: str,
    *,
    workers: int = 8,
    compute_phash: bool = True,
    clamp_tolerance: float = DEFAULT_CLAMP_TOLERANCE,
    min_box_side: float = DEFAULT_MIN_BOX_SIDE,
    tiny_box_area: float = 1e-5,
) -> tuple[SplitAudit, dict[str, ImageInfo]]:
    """Audite un split : appariement image/label, integrite, geometrie, classes.

    Args:
        dataset: Configuration dataset resolue.
        split: Nom du split (``train`` / ``valid`` / ``test``).
        workers: Threads pour l'inspection des images.
        compute_phash: Calcule les hashes perceptuels.
        clamp_tolerance: Tolerance de derive numerique sur [0, 1].
        min_box_side: Cote normalise minimal avant de declarer une boite degeneree.
        tiny_box_area: Aire normalisee sous laquelle une boite est signalee.

    Returns:
        Le :class:`SplitAudit` et la table ``nom de fichier -> ImageInfo``.
    """
    images_dir = dataset.splits[split]
    labels_dir = dataset.labels_dir(split)
    audit = SplitAudit(name=split, images_dir=images_dir, labels_dir=labels_dir)

    image_paths = sorted(p for p in images_dir.iterdir() if p.is_file()) if images_dir.is_dir() else []
    supported = [p for p in image_paths if p.suffix.lower() in IMAGE_EXTENSIONS]
    for path in image_paths:
        audit.extension_counts[path.suffix.lower()] += 1
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            audit.add_problem("unsupported_extension", path, 0, f"extension {path.suffix!r}")

    audit.n_images = len(supported)
    LOGGER.info("[%s] inspection de %d images...", split, len(supported))
    infos = scan_images(supported, workers=workers, compute_phash=compute_phash)
    info_by_stem: dict[str, ImageInfo] = {}

    for info in infos:
        info_by_stem[info.path.stem] = info
        audit.total_bytes += info.size_bytes
        if not info.readable:
            audit.unreadable_images.append({"file": info.path.name, "error": info.error})
            audit.add_problem("unreadable_image", info.path, 0, info.error)
            continue
        audit.dimensions[f"{info.width}x{info.height}"] += 1
        current = (info.width, info.height)
        pixels = current[0] * current[1]
        if audit.min_dimension is None or pixels < audit.min_dimension[0] * audit.min_dimension[1]:
            audit.min_dimension = current
        if audit.max_dimension is None or pixels > audit.max_dimension[0] * audit.max_dimension[1]:
            audit.max_dimension = current
        ratio = info.aspect_ratio
        if ratio:
            audit.aspect_ratio_min = (
                ratio if audit.aspect_ratio_min == 0.0 else min(audit.aspect_ratio_min, ratio)
            )
            audit.aspect_ratio_max = max(audit.aspect_ratio_max, ratio)

    label_paths = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
    audit.n_labels = len(label_paths)
    label_stems = {p.stem: p for p in label_paths}

    for stem in sorted(info_by_stem):
        if stem not in label_stems:
            audit.missing_labels.append(info_by_stem[stem].path.name)
    for stem in sorted(label_stems):
        if stem not in info_by_stem:
            audit.orphan_labels.append(label_stems[stem].name)

    LOGGER.info("[%s] analyse de %d fichiers de labels...", split, len(label_paths))
    for _stem, label_path in sorted(label_stems.items()):
        try:
            text = label_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            audit.add_problem("unreadable_label", label_path, 0, str(exc))
            continue

        if not text.strip():
            audit.n_empty_label_files += 1
            audit.n_images_without_objects += 1
            audit.add_problem("empty_label_file", label_path, 0, "aucune annotation")
            continue

        parsed_lines = parse_label_text(
            text,
            num_classes=dataset.num_classes,
            clamp_tolerance=clamp_tolerance,
            min_box_side=min_box_side,
        )
        if not parsed_lines:
            audit.n_empty_label_files += 1
            audit.n_images_without_objects += 1
            continue

        for line_no, parsed in parsed_lines:
            audit.n_annotations += 1
            _tally_line(audit, dataset, label_path, line_no, parsed, tiny_box_area)

    return audit, {info.path.name: info for info in infos}


def _tally_line(
    audit: SplitAudit,
    dataset: DatasetConfig,
    label_path: Path,
    line_no: int,
    parsed: ParsedLine,
    tiny_box_area: float,
) -> None:
    """Comptabilise une ligne analysee dans le rapport de split."""
    if parsed.kind is LineKind.DETECTION:
        audit.n_detection_lines += 1
    elif parsed.kind is LineKind.POLYGON:
        audit.n_polygon_lines += 1
    else:
        audit.n_malformed_lines += 1

    for code in parsed.issues:
        # Une conversion polygone est un fait structurel, pas un probleme :
        # elle est deja comptee via n_polygon_lines.
        if code is IssueCode.CONVERTED_FROM_POLYGON:
            continue
        audit.add_problem(code.value, label_path, line_no, parsed.detail)

    if parsed.is_error:
        return

    if parsed.class_id is not None and parsed.box is not None:
        audit.class_counts[dataset.class_name(parsed.class_id)] += 1
        area = parsed.box.area
        audit.box_areas.append(area)
        if area < tiny_box_area:
            audit.add_problem("very_small_box", label_path, line_no, f"aire={area:.8f}")


# --------------------------------------------------------------------------- #
# Rapport global
# --------------------------------------------------------------------------- #
def _imbalance_stats(class_counts: Counter[str], class_names: Sequence[str]) -> dict[str, Any]:
    """Calcule les indicateurs de desequilibre des classes."""
    total = sum(class_counts.values())
    per_class = {name: class_counts.get(name, 0) for name in class_names}
    present = {name: count for name, count in per_class.items() if count > 0}
    max_count = max(present.values()) if present else 0
    min_count = min(present.values()) if present else 0
    return {
        "total_instances": total,
        "per_class": per_class,
        "share_percent": {
            name: round(100.0 * count / total, 2) if total else 0.0 for name, count in per_class.items()
        },
        "absent_classes": [name for name, count in per_class.items() if count == 0],
        "max_over_min_ratio": round(max_count / min_count, 2) if min_count else None,
        "majority_class": max(per_class, key=lambda k: per_class[k]) if per_class else None,
        "minority_class": min(present, key=lambda k: present[k]) if present else None,
    }


def audit_dataset(
    data_yaml: str | Path,
    *,
    workers: int = 8,
    compute_phash: bool = True,
    near_duplicate_distance: int = 3,
    clamp_tolerance: float = DEFAULT_CLAMP_TOLERANCE,
    min_box_side: float = DEFAULT_MIN_BOX_SIDE,
) -> dict[str, Any]:
    """Audite l'ensemble des splits d'un dataset et agrege les resultats.

    Args:
        data_yaml: Chemin du ``data.yaml``.
        workers: Threads pour l'inspection des images.
        compute_phash: Active la detection de quasi-doublons visuels.
        near_duplicate_distance: Distance de Hamming maximale entre phashes.
        clamp_tolerance: Tolerance de derive numerique.
        min_box_side: Cote minimal avant declaration de boite degeneree.

    Returns:
        Le rapport d'audit complet sous forme de dictionnaire serialisable.
    """
    dataset = load_dataset_config(data_yaml)
    LOGGER.info("Dataset : %s (%d classes)", dataset.yaml_path, dataset.num_classes)

    split_audits: dict[str, SplitAudit] = {}
    all_images: dict[str, dict[str, ImageInfo]] = {}
    for split in dataset.splits:
        audit, infos = audit_split(
            dataset,
            split,
            workers=workers,
            compute_phash=compute_phash,
            clamp_tolerance=clamp_tolerance,
            min_box_side=min_box_side,
        )
        split_audits[split] = audit
        all_images[split] = infos

    duplicates = _analyse_duplicates(
        all_images,
        compute_phash=compute_phash,
        near_duplicate_distance=near_duplicate_distance,
    )

    totals = _aggregate_totals(split_audits, dataset)
    problems = _collect_blocking_problems(split_audits, duplicates)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": describe_environment(),
        "dataset": {
            "data_yaml": str(dataset.yaml_path),
            "root": str(dataset.root),
            "num_classes": dataset.num_classes,
            "class_names": dataset.names,
            "splits_found": sorted(dataset.splits),
            "splits_expected": sorted({"train", "valid", "test"}),
            "missing_splits": sorted({"train", "valid", "test"} - set(dataset.splits)),
        },
        "splits": {name: _split_to_dict(audit, dataset) for name, audit in split_audits.items()},
        "totals": totals,
        "duplicates": duplicates,
        "problems": problems,
        "conclusion": _build_conclusion(totals, problems, duplicates),
    }


def _split_to_dict(audit: SplitAudit, dataset: DatasetConfig) -> dict[str, Any]:
    """Serialise un :class:`SplitAudit`."""
    return {
        "images_dir": str(audit.images_dir),
        "labels_dir": str(audit.labels_dir),
        "counts": {
            "images": audit.n_images,
            "label_files": audit.n_labels,
            "annotations": audit.n_annotations,
            "detection_lines": audit.n_detection_lines,
            "polygon_lines": audit.n_polygon_lines,
            "malformed_lines": audit.n_malformed_lines,
            "empty_label_files": audit.n_empty_label_files,
            "images_without_objects": audit.n_images_without_objects,
            "total_size": human_bytes(audit.total_bytes),
        },
        "pairing": {
            "images_missing_label": len(audit.missing_labels),
            "labels_without_image": len(audit.orphan_labels),
            "missing_label_examples": audit.missing_labels[:MAX_EXAMPLES],
            "orphan_label_examples": audit.orphan_labels[:MAX_EXAMPLES],
            "perfect_match": not audit.missing_labels and not audit.orphan_labels,
        },
        "images": {
            "extensions": dict(audit.extension_counts),
            "unreadable": audit.unreadable_images,
            "distinct_resolutions": len(audit.dimensions),
            "most_common_resolutions": audit.dimensions.most_common(5),
            "min_resolution": list(audit.min_dimension) if audit.min_dimension else None,
            "max_resolution": list(audit.max_dimension) if audit.max_dimension else None,
            "aspect_ratio_min": round(audit.aspect_ratio_min, 4),
            "aspect_ratio_max": round(audit.aspect_ratio_max, 4),
        },
        "classes": _imbalance_stats(audit.class_counts, dataset.names),
        "issues": dict(audit.issue_counts),
        "issue_examples": audit.problems,
        "box_stats": _box_stats(audit.box_areas),
    }


def _box_stats(areas: Sequence[float]) -> dict[str, Any]:
    """Statistiques descriptives sur l'aire normalisee des boites."""
    if not areas:
        return {"count": 0}
    ordered = sorted(areas)
    count = len(ordered)

    def percentile(fraction: float) -> float:
        position = min(count - 1, max(0, int(round(fraction * (count - 1)))))
        return round(ordered[position], 6)

    return {
        "count": count,
        "area_min": round(ordered[0], 8),
        "area_p05": percentile(0.05),
        "area_median": percentile(0.50),
        "area_p95": percentile(0.95),
        "area_max": round(ordered[-1], 6),
        "small_objects_share_percent": round(
            100.0 * sum(1 for a in ordered if a < 0.01) / count, 2
        ),
    }


def _analyse_duplicates(
    all_images: dict[str, dict[str, ImageInfo]],
    *,
    compute_phash: bool,
    near_duplicate_distance: int,
) -> dict[str, Any]:
    """Detecte doublons binaires, quasi-doublons visuels et fuites inter-splits."""
    by_sha: dict[str, list[str]] = defaultdict(list)
    phash_entries: list[tuple[str, str]] = []
    by_source_stem: dict[str, list[str]] = defaultdict(list)

    for split, infos in all_images.items():
        for name, info in infos.items():
            key = f"{split}/{name}"
            if info.sha256:
                by_sha[info.sha256].append(key)
            if info.phash:
                phash_entries.append((key, info.phash))
            by_source_stem[roboflow_source_stem(name)].append(key)

    exact_groups = [sorted(group) for group in by_sha.values() if len(group) > 1]
    exact_groups.sort()
    exact_cross_split = [
        group for group in exact_groups if len({item.split("/", 1)[0] for item in group}) > 1
    ]

    near_pairs: list[tuple[str, str, int]] = []
    if compute_phash and phash_entries:
        LOGGER.info("Recherche de quasi-doublons sur %d empreintes perceptuelles...", len(phash_entries))
        near_pairs = find_near_duplicates(phash_entries, max_distance=near_duplicate_distance)
    near_cross_split = [
        pair for pair in near_pairs if pair[0].split("/", 1)[0] != pair[1].split("/", 1)[0]
    ]

    # Le nombre de paires explose de facon quadratique a l'interieur d'un
    # cluster : on raisonne donc en composantes connexes, bien plus lisible.
    near_clusters = cluster_pairs(near_pairs)
    cross_split_clusters = [
        cluster for cluster in near_clusters if len({m.split("/", 1)[0] for m in cluster}) > 1
    ]
    images_in_clusters = sum(len(cluster) for cluster in near_clusters)
    images_in_cross_clusters = sum(len(cluster) for cluster in cross_split_clusters)

    source_leaks = [
        {"source_stem": stem, "files": sorted(files)}
        for stem, files in sorted(by_source_stem.items())
        if len({item.split("/", 1)[0] for item in files}) > 1
    ]
    files_in_source_leaks = sum(len(item["files"]) for item in source_leaks)

    sequence_map: dict[str, set[str]] = defaultdict(set)
    sequence_sizes: Counter[str] = Counter()
    for split, infos in all_images.items():
        for name in infos:
            prefix = sequence_prefix(name)
            if prefix:
                sequence_map[prefix].add(split)
                sequence_sizes[prefix] += 1
    crossing_sequences = sorted(p for p, splits in sequence_map.items() if len(splits) > 1)

    return {
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_files": sum(len(g) - 1 for g in exact_groups),
        "exact_duplicate_examples": exact_groups[:MAX_EXAMPLES],
        "exact_cross_split_groups": len(exact_cross_split),
        "exact_cross_split_examples": exact_cross_split[:MAX_EXAMPLES],
        "perceptual_hash_available": bool(phash_entries),
        "near_duplicate_distance": near_duplicate_distance,
        "near_duplicate_pairs": len(near_pairs),
        "near_duplicate_clusters": len(near_clusters),
        "near_duplicate_images": images_in_clusters,
        "near_duplicate_largest_cluster": len(near_clusters[0]) if near_clusters else 0,
        "near_duplicate_cross_split_pairs": len(near_cross_split),
        "near_duplicate_cross_split_clusters": len(cross_split_clusters),
        "near_duplicate_cross_split_images": images_in_cross_clusters,
        "near_duplicate_cluster_examples": [cluster[:8] for cluster in near_clusters[:10]],
        "shared_source_stem_groups": len(source_leaks),
        "shared_source_stem_files": files_in_source_leaks,
        "shared_source_stem_examples": source_leaks[:MAX_EXAMPLES],
        "sequence_prefixes": len(sequence_map),
        "sequence_prefixes_crossing_splits": len(crossing_sequences),
        "sequence_images_crossing_splits": sum(sequence_sizes[p] for p in crossing_sequences),
        "sequence_crossing_examples": [
            {"prefix": p, "images": sequence_sizes[p], "splits": sorted(sequence_map[p])}
            for p in sorted(crossing_sequences, key=lambda p: -sequence_sizes[p])[:10]
        ],
    }


def _aggregate_totals(split_audits: dict[str, SplitAudit], dataset: DatasetConfig) -> dict[str, Any]:
    """Agrege les compteurs de tous les splits."""
    combined_classes: Counter[str] = Counter()
    issue_totals: Counter[str] = Counter()
    totals: dict[str, Any] = {
        "images": 0,
        "label_files": 0,
        "annotations": 0,
        "detection_lines": 0,
        "polygon_lines": 0,
        "malformed_lines": 0,
        "empty_label_files": 0,
        "images_missing_label": 0,
        "labels_without_image": 0,
        "unreadable_images": 0,
    }
    for audit in split_audits.values():
        combined_classes.update(audit.class_counts)
        issue_totals.update(audit.issue_counts)
        totals["images"] += audit.n_images
        totals["label_files"] += audit.n_labels
        totals["annotations"] += audit.n_annotations
        totals["detection_lines"] += audit.n_detection_lines
        totals["polygon_lines"] += audit.n_polygon_lines
        totals["malformed_lines"] += audit.n_malformed_lines
        totals["empty_label_files"] += audit.n_empty_label_files
        totals["images_missing_label"] += len(audit.missing_labels)
        totals["labels_without_image"] += len(audit.orphan_labels)
        totals["unreadable_images"] += len(audit.unreadable_images)

    totals["classes"] = _imbalance_stats(combined_classes, dataset.names)
    totals["issues"] = dict(issue_totals)
    return totals


def _collect_blocking_problems(
    split_audits: dict[str, SplitAudit], duplicates: dict[str, Any]
) -> list[dict[str, Any]]:
    """Liste hierarchisee des problemes detectes (bloquants et informatifs)."""
    problems: list[dict[str, Any]] = []

    def add(severity: str, code: str, message: str, **extra: Any) -> None:
        problems.append({"severity": severity, "code": code, "message": message, **extra})

    for name, audit in split_audits.items():
        if audit.missing_labels:
            add(
                "error",
                "images_without_label",
                f"[{name}] {len(audit.missing_labels)} image(s) sans fichier de label.",
                examples=audit.missing_labels[:5],
            )
        if audit.orphan_labels:
            add(
                "error",
                "labels_without_image",
                f"[{name}] {len(audit.orphan_labels)} label(s) sans image correspondante.",
                examples=audit.orphan_labels[:5],
            )
        if audit.unreadable_images:
            add(
                "error",
                "unreadable_images",
                f"[{name}] {len(audit.unreadable_images)} image(s) illisible(s) ou corrompue(s).",
                examples=[item["file"] for item in audit.unreadable_images[:5]],
            )
        if audit.n_malformed_lines:
            add(
                "error",
                "malformed_lines",
                f"[{name}] {audit.n_malformed_lines} ligne(s) d'annotation au format inexploitable.",
            )
        if audit.n_polygon_lines:
            add(
                "warning",
                "polygon_lines",
                f"[{name}] {audit.n_polygon_lines} ligne(s) de segmentation (polygone) a convertir "
                f"en boite englobante pour une tache de detection.",
            )
        if audit.n_empty_label_files:
            add(
                "info",
                "empty_label_files",
                f"[{name}] {audit.n_empty_label_files} fichier(s) de label vide(s) "
                f"(images sans objet, valides pour YOLO en tant que negatifs).",
            )
        for code in (
            IssueCode.CLASS_ID_OUT_OF_RANGE,
            IssueCode.NON_NUMERIC,
            IssueCode.NON_POSITIVE_SIZE,
            IssueCode.CENTER_OUT_OF_RANGE,
            IssueCode.SIZE_OUT_OF_RANGE,
            IssueCode.DEGENERATE_AFTER_CLIP,
        ):
            count = audit.issue_counts.get(code.value, 0)
            if count:
                add("error", code.value, f"[{name}] {count} ligne(s) : {code.value}.")
        for code in (IssueCode.CLIPPED_TO_BOUNDS, IssueCode.CLAMPED_MINOR):
            count = audit.issue_counts.get(code.value, 0)
            if count:
                add("warning", code.value, f"[{name}] {count} boite(s) corrigee(s) : {code.value}.")
        tiny = audit.issue_counts.get("very_small_box", 0)
        if tiny:
            add("info", "very_small_box", f"[{name}] {tiny} boite(s) tres petite(s) (aire < 1e-5).")

    if duplicates["exact_cross_split_groups"]:
        add(
            "error",
            "exact_duplicates_across_splits",
            f"{duplicates['exact_cross_split_groups']} groupe(s) d'images identiques presentes "
            f"dans plusieurs splits — fuite de donnees entre entrainement et evaluation.",
        )
    elif duplicates["exact_duplicate_groups"]:
        add(
            "warning",
            "exact_duplicates",
            f"{duplicates['exact_duplicate_groups']} groupe(s) de doublons binaires exacts "
            f"(a l'interieur d'un meme split).",
        )
    if duplicates["near_duplicate_cross_split_clusters"]:
        add(
            "warning",
            "near_duplicates_across_splits",
            f"{duplicates['near_duplicate_cross_split_clusters']} groupe(s) d'images visuellement "
            f"quasi identiques ({duplicates['near_duplicate_cross_split_images']} images) repartis "
            f"entre plusieurs splits — les metriques de validation/test sont optimistes.",
        )
    if duplicates["shared_source_stem_groups"]:
        add(
            "warning",
            "shared_source_images",
            f"{duplicates['shared_source_stem_groups']} image(s) source Roboflow "
            f"({duplicates['shared_source_stem_files']} fichiers) presente(s) dans plusieurs splits : "
            f"meme prefixe avant '.rf.' — variantes augmentees d'une meme photo reparties "
            f"entre entrainement et evaluation.",
        )
    if duplicates.get("sequence_prefixes_crossing_splits"):
        add(
            "warning",
            "video_sequences_across_splits",
            f"{duplicates['sequence_prefixes_crossing_splits']} sequence(s) d'images numerotees "
            f"({duplicates['sequence_images_crossing_splits']} images de type 'frame_000324') "
            f"reparties entre plusieurs splits : les frames consecutives d'une meme video sont "
            f"quasi identiques.",
        )
    return problems


def _build_conclusion(
    totals: dict[str, Any], problems: list[dict[str, Any]], duplicates: dict[str, Any]
) -> dict[str, Any]:
    """Synthetise l'audit en un verdict exploitable."""
    errors = [p for p in problems if p["severity"] == "error"]
    warnings = [p for p in problems if p["severity"] == "warning"]
    return {
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "detection_ready": not errors and totals["polygon_lines"] == 0,
        "requires_conversion": totals["polygon_lines"] > 0,
        "summary": (
            f"{totals['images']} images, {totals['annotations']} annotations, "
            f"{totals['polygon_lines']} ligne(s) polygonale(s) a convertir, "
            f"{len(errors)} erreur(s), {len(warnings)} avertissement(s), "
            f"{duplicates['exact_duplicate_groups']} groupe(s) de doublons exacts."
        ),
    }


# --------------------------------------------------------------------------- #
# Rendu Markdown et graphiques
# --------------------------------------------------------------------------- #
def render_markdown(report: dict[str, Any], *, assets: dict[str, Path] | None = None) -> str:
    """Genere le rapport Markdown lisible a partir du rapport JSON."""
    dataset = report["dataset"]
    totals = report["totals"]
    conclusion = report["conclusion"]
    class_names: list[str] = dataset["class_names"]
    lines: list[str] = []

    lines.append("# Rapport d'audit du dataset EPI")
    lines.append("")
    lines.append(f"- **Genere le** : {report['generated_at']}")
    lines.append(f"- **data.yaml** : `{dataset['data_yaml']}`")
    lines.append(f"- **Classes ({dataset['num_classes']})** : {', '.join(class_names)}")
    lines.append(f"- **Splits trouves** : {', '.join(dataset['splits_found'])}")
    if dataset["missing_splits"]:
        lines.append(f"- **Splits manquants** : {', '.join(dataset['missing_splits'])}")
    lines.append("")

    verdict = "PRET POUR LA DETECTION" if conclusion["detection_ready"] else "CONVERSION REQUISE"
    lines.append(f"> **Verdict : {verdict}** — {conclusion['summary']}")
    lines.append("")

    lines.append("## 1. Vue d'ensemble par split")
    lines.append("")
    rows = []
    for split, data in report["splits"].items():
        counts = data["counts"]
        pairing = data["pairing"]
        rows.append(
            [
                split,
                counts["images"],
                counts["label_files"],
                counts["annotations"],
                counts["detection_lines"],
                counts["polygon_lines"],
                counts["malformed_lines"],
                "oui" if pairing["perfect_match"] else "NON",
                counts["total_size"],
            ]
        )
    rows.append(
        [
            "**TOTAL**",
            totals["images"],
            totals["label_files"],
            totals["annotations"],
            totals["detection_lines"],
            totals["polygon_lines"],
            totals["malformed_lines"],
            "oui" if totals["images_missing_label"] == 0 and totals["labels_without_image"] == 0 else "NON",
            "",
        ]
    )
    lines.append(
        markdown_table(
            [
                "Split", "Images", "Labels", "Annot.", "Detection",
                "Polygone", "Malformees", "Appariement", "Taille",
            ],
            rows,
        )
    )
    lines.append("")

    lines.append("## 2. Distribution des classes")
    lines.append("")
    header = ["Classe", "ID", *report["splits"].keys(), "Total", "Part (%)"]
    rows = []
    for class_id, name in enumerate(class_names):
        per_split = [report["splits"][s]["classes"]["per_class"].get(name, 0) for s in report["splits"]]
        total = totals["classes"]["per_class"].get(name, 0)
        share = totals["classes"]["share_percent"].get(name, 0.0)
        rows.append([name, class_id, *per_split, total, f"{share:.2f}"])
    lines.append(markdown_table(header, rows))
    lines.append("")
    imbalance = totals["classes"]
    lines.append(f"- Classe majoritaire : **{imbalance['majority_class']}**")
    lines.append(f"- Classe minoritaire : **{imbalance['minority_class']}**")
    lines.append(f"- Ratio max/min : **{imbalance['max_over_min_ratio']}**")
    if imbalance["absent_classes"]:
        lines.append(f"- Classes absentes : {', '.join(imbalance['absent_classes'])}")
    lines.append("")

    lines.append("## 3. Integrite des images")
    lines.append("")
    rows = []
    for split, data in report["splits"].items():
        images = data["images"]
        rows.append(
            [
                split,
                ", ".join(f"{ext}:{n}" for ext, n in sorted(images["extensions"].items())),
                len(images["unreadable"]),
                images["distinct_resolutions"],
                "x".join(map(str, images["min_resolution"])) if images["min_resolution"] else "-",
                "x".join(map(str, images["max_resolution"])) if images["max_resolution"] else "-",
                f"{images['aspect_ratio_min']:.2f} - {images['aspect_ratio_max']:.2f}",
            ]
        )
    lines.append(
        markdown_table(
            ["Split", "Extensions", "Illisibles", "Resolutions", "Min", "Max", "Ratio L/H"],
            rows,
        )
    )
    lines.append("")

    lines.append("## 4. Statistiques geometriques des boites")
    lines.append("")
    rows = []
    for split, data in report["splits"].items():
        stats = data["box_stats"]
        if not stats.get("count"):
            continue
        rows.append(
            [
                split,
                stats["count"],
                stats["area_min"],
                stats["area_p05"],
                stats["area_median"],
                stats["area_p95"],
                stats["area_max"],
                f"{stats['small_objects_share_percent']} %",
            ]
        )
    lines.append(
        markdown_table(
            ["Split", "Boites", "Aire min", "p05", "Mediane", "p95", "Aire max", "Petits objets (<1 %)"],
            rows,
        )
    )
    lines.append("")

    lines.append("## 5. Doublons et risques de fuite")
    lines.append("")
    duplicates = report["duplicates"]
    distance = duplicates["near_duplicate_distance"]
    rows = [
        ["Groupes de doublons binaires exacts", duplicates["exact_duplicate_groups"]],
        ["Fichiers en trop (doublons exacts)", duplicates["exact_duplicate_files"]],
        ["Groupes de doublons exacts inter-splits", duplicates["exact_cross_split_groups"]],
        ["Hash perceptuel disponible", "oui" if duplicates["perceptual_hash_available"] else "non"],
        [f"Clusters quasi identiques (Hamming <= {distance})", duplicates["near_duplicate_clusters"]],
        ["Images concernees", duplicates["near_duplicate_images"]],
        ["Plus grand cluster", duplicates["near_duplicate_largest_cluster"]],
        ["Clusters s'etendant sur plusieurs splits", duplicates["near_duplicate_cross_split_clusters"]],
        ["Images dans ces clusters inter-splits", duplicates["near_duplicate_cross_split_images"]],
        ["Images source Roboflow partagees entre splits", duplicates["shared_source_stem_groups"]],
        ["Fichiers concernes", duplicates["shared_source_stem_files"]],
        [
            "Sequences numerotees reparties entre splits",
            duplicates.get("sequence_prefixes_crossing_splits", 0),
        ],
        ["Images de ces sequences", duplicates.get("sequence_images_crossing_splits", 0)],
    ]
    lines.append(markdown_table(["Indicateur", "Valeur"], rows))
    lines.append("")
    lines.append(
        f"> Le nombre brut de *paires* quasi identiques ({duplicates['near_duplicate_pairs']}) n'est pas "
        f"un bon indicateur : un groupe de N images similaires produit N*(N-1)/2 paires. "
        f"Les clusters ci-dessus refletent la realite."
    )
    lines.append("")
    if duplicates.get("sequence_crossing_examples"):
        lines.append("Principales sequences reparties entre splits :")
        lines.append("")
        lines.append(
            markdown_table(
                ["Prefixe", "Images", "Splits"],
                [
                    [item["prefix"], item["images"], ", ".join(item["splits"])]
                    for item in duplicates["sequence_crossing_examples"]
                ],
            )
        )
        lines.append("")

    lines.append("## 6. Problemes detectes")
    lines.append("")
    if not report["problems"]:
        lines.append("Aucun probleme detecte.")
    else:
        severity_order = {"error": 0, "warning": 1, "info": 2}
        ordered = sorted(report["problems"], key=lambda p: severity_order.get(p["severity"], 3))
        rows = [
            [
                {"error": "ERREUR", "warning": "AVERT.", "info": "INFO"}.get(p["severity"], p["severity"]),
                p["code"],
                p["message"],
            ]
            for p in ordered
        ]
        lines.append(markdown_table(["Severite", "Code", "Description"], rows))
    lines.append("")

    lines.append("## 7. Detail des codes d'anomalie par split")
    lines.append("")
    all_codes = sorted({code for data in report["splits"].values() for code in data["issues"]})
    if all_codes:
        rows = [
            [code, *[report["splits"][s]["issues"].get(code, 0) for s in report["splits"]]]
            for code in all_codes
        ]
        lines.append(markdown_table(["Code", *report["splits"].keys()], rows))
    else:
        lines.append("Aucune anomalie de ligne relevee.")
    lines.append("")

    if assets:
        lines.append("## 8. Graphiques et exemples")
        lines.append("")
        for label, path in assets.items():
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"![{label}]({path.as_posix()})")
            lines.append("")

    lines.append("## Glossaire des codes")
    lines.append("")
    glossary = [
        ("polygon_lines", "Ligne de segmentation (class_id + paires x/y). A convertir en boite."),
        ("converted_from_polygon", "Boite obtenue par min/max sur les sommets d'un polygone."),
        ("clamped_minor", "Derive numerique <= tolerance ramenee dans [0, 1]."),
        ("clipped_to_bounds", "Objet partiellement hors cadre : boite rognee sur l'image."),
        ("class_id_out_of_range", "Identifiant de classe absent de data.yaml."),
        ("non_numeric", "Champ non convertible en nombre."),
        ("non_positive_size", "Largeur ou hauteur nulle/negative."),
        ("center_out_of_range", "Centre de la boite hors de l'image."),
        ("bad_field_count", "Nombre de champs incompatible avec detection ou polygone."),
        ("very_small_box", "Boite d'aire normalisee < 1e-5."),
    ]
    lines.append(markdown_table(["Code", "Signification"], glossary))
    lines.append("")
    return "\n".join(lines)


def generate_assets(
    report: dict[str, Any],
    dataset: DatasetConfig,
    output_dir: Path,
    *,
    n_samples: int = 6,
) -> dict[str, Path]:
    """Genere graphiques et exemples d'annotations visualisees.

    Args:
        report: Rapport d'audit.
        dataset: Configuration dataset.
        output_dir: Repertoire de destination des PNG.
        n_samples: Nombre d'images d'exemple a annoter.

    Returns:
        ``{libelle: chemin_relatif}`` des ressources reellement produites.
    """
    ensure_dir(output_dir)
    assets: dict[str, Path] = {}

    counts_by_split = {
        split: data["classes"]["per_class"] for split, data in report["splits"].items()
    }
    chart = plot_class_distribution(
        counts_by_split, dataset.names, output_dir / "class_distribution.png"
    )
    if chart:
        assets["Frequence des classes par split"] = Path(chart.name)

    areas_by_split: dict[str, list[float]] = {}
    for split in dataset.splits:
        labels_dir = dataset.labels_dir(split)
        areas: list[float] = []
        for label_path in sorted(labels_dir.glob("*.txt"))[:1500]:
            text = label_path.read_text(encoding="utf-8", errors="replace")
            for _, parsed in parse_label_text(text, num_classes=dataset.num_classes):
                if not parsed.is_error and parsed.box is not None:
                    areas.append(parsed.box.area)
        areas_by_split[split] = areas
    size_chart = plot_box_size_distribution(areas_by_split, output_dir / "box_size_distribution.png")
    if size_chart:
        assets["Distribution des tailles de boites"] = Path(size_chart.name)

    sample = _render_annotation_samples(dataset, output_dir, n_samples=n_samples)
    if sample:
        assets["Exemples d'annotations (verification visuelle)"] = Path(sample.name)
    return assets


def _render_annotation_samples(
    dataset: DatasetConfig, output_dir: Path, *, n_samples: int = 6
) -> Path | None:
    """Assemble une planche contact d'images annotees depuis les labels du dataset.

    Privilegie les images contenant des annotations polygonales afin de rendre
    visible la conversion attendue.
    """
    try:
        import cv2
        import numpy as np

        from .visualization import draw_yolo_labels
    except ImportError:  # pragma: no cover
        LOGGER.warning("OpenCV indisponible — planche d'exemples non generee.")
        return None

    candidates: list[tuple[Path, list[tuple[int, float, float, float, float]]]] = []
    polygon_first: list[tuple[Path, list[tuple[int, float, float, float, float]]]] = []

    for split in dataset.splits:
        images_dir = dataset.splits[split]
        labels_dir = dataset.labels_dir(split)
        for label_path in sorted(labels_dir.glob("*.txt"))[:800]:
            text = label_path.read_text(encoding="utf-8", errors="replace")
            parsed_lines = parse_label_text(text, num_classes=dataset.num_classes)
            boxes = [
                (p.class_id, p.box.cx, p.box.cy, p.box.w, p.box.h)
                for _, p in parsed_lines
                if not p.is_error and p.box is not None and p.class_id is not None
            ]
            if not boxes:
                continue
            image_path = next(
                (
                    images_dir / f"{label_path.stem}{ext}"
                    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                    if (images_dir / f"{label_path.stem}{ext}").is_file()
                ),
                None,
            )
            if image_path is None:
                continue
            has_polygon = any(p.kind is LineKind.POLYGON for _, p in parsed_lines)
            (polygon_first if has_polygon else candidates).append((image_path, boxes))
            if len(polygon_first) >= n_samples:
                break
        if len(polygon_first) >= n_samples:
            break

    selected = (polygon_first + candidates)[:n_samples]
    if not selected:
        return None

    tiles: list[Any] = []
    tile_size = 400
    for image_path, boxes in selected:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        annotated = draw_yolo_labels(image, boxes, dataset.names, font_scale=0.45)
        height, width = annotated.shape[:2]
        scale = tile_size / max(height, width)
        resized = cv2.resize(annotated, (max(1, int(width * scale)), max(1, int(height * scale))))
        canvas = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        y_off = (tile_size - resized.shape[0]) // 2
        x_off = (tile_size - resized.shape[1]) // 2
        canvas[y_off : y_off + resized.shape[0], x_off : x_off + resized.shape[1]] = resized
        tiles.append(canvas)

    if not tiles:
        return None

    columns = min(3, len(tiles))
    rows_count = (len(tiles) + columns - 1) // columns
    while len(tiles) < rows_count * columns:
        tiles.append(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
    grid = np.vstack(
        [np.hstack(tiles[r * columns : (r + 1) * columns]) for r in range(rows_count)]
    )
    target = output_dir / "annotation_samples.png"
    cv2.imwrite(str(target), grid)
    return target


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_audit(
    data_yaml: str | Path,
    output_json: str | Path,
    *,
    workers: int = 8,
    compute_phash: bool = True,
    near_duplicate_distance: int = 3,
    generate_plots: bool = True,
    markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute l'audit et ecrit tous les livrables sur disque.

    Args:
        data_yaml: Chemin du ``data.yaml`` a auditer.
        output_json: Chemin du rapport JSON.
        workers: Threads d'inspection.
        compute_phash: Active la detection de quasi-doublons.
        near_duplicate_distance: Distance de Hamming maximale.
        generate_plots: Genere graphiques et planche d'exemples.
        markdown_path: Chemin du rapport Markdown (deduit du JSON si ``None``).

    Returns:
        Le rapport d'audit.
    """
    report = audit_dataset(
        data_yaml,
        workers=workers,
        compute_phash=compute_phash,
        near_duplicate_distance=near_duplicate_distance,
    )
    json_path = resolve_path(output_json)
    write_json(json_path, report)
    LOGGER.info("Rapport JSON ecrit : %s", json_path)

    assets: dict[str, Path] = {}
    if generate_plots:
        dataset = load_dataset_config(data_yaml)
        assets = generate_assets(report, dataset, json_path.parent / f"{json_path.stem}_assets")
        assets = {
            label: Path(f"{json_path.stem}_assets") / path for label, path in assets.items()
        }

    md_path = resolve_path(markdown_path) if markdown_path else json_path.with_suffix(".md")
    write_text(md_path, render_markdown(report, assets=assets))
    LOGGER.info("Rapport Markdown ecrit : %s", md_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments de la commande d'audit."""
    parser = argparse.ArgumentParser(
        prog="python -m ppe_detection.dataset_audit",
        description="Audit complet d'un dataset de detection YOLO (EPI).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="data.yaml", help="Chemin du data.yaml a auditer.")
    parser.add_argument(
        "--output",
        default="artifacts/reports/dataset_audit.json",
        help="Chemin du rapport JSON de sortie.",
    )
    parser.add_argument("--markdown", default=None, help="Chemin du rapport Markdown (defaut : <output>.md).")
    parser.add_argument("--workers", type=int, default=8, help="Threads pour l'inspection des images.")
    parser.add_argument(
        "--skip-perceptual-hash",
        action="store_true",
        help="Desactive la detection de quasi-doublons visuels (plus rapide).",
    )
    parser.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=3,
        help="Distance de Hamming maximale entre deux hashes perceptuels.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Ne genere ni graphiques ni exemples.")
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Retourne un code de sortie non nul si des erreurs bloquantes sont detectees.",
    )
    parser.add_argument(
        "--fail-on-polygon",
        action="store_true",
        help="Echoue si des annotations polygonales subsistent (utilise apres conversion).",
    )
    parser.add_argument("--log-level", default="INFO", help="Niveau de log.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entree CLI. Retourne le code de sortie du processus."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=project_root() / "artifacts" / "logs" / "audit.log")

    try:
        report = run_audit(
            args.data,
            args.output,
            workers=args.workers,
            compute_phash=not args.skip_perceptual_hash,
            near_duplicate_distance=args.near_duplicate_distance,
            generate_plots=not args.no_plots,
            markdown_path=args.markdown,
        )
    except (ConfigError, FileNotFoundError) as exc:
        LOGGER.error("Audit impossible : %s", exc)
        return 2

    conclusion = report["conclusion"]
    LOGGER.info("Resultat : %s", conclusion["summary"])

    exit_code = 0
    if args.fail_on_error and conclusion["n_errors"] > 0:
        LOGGER.error("%d erreur(s) bloquante(s) detectee(s).", conclusion["n_errors"])
        exit_code = 1
    if args.fail_on_polygon and report["totals"]["polygon_lines"] > 0:
        LOGGER.error(
            "%d ligne(s) polygonale(s) subsistent alors que le dataset devait etre normalise.",
            report["totals"]["polygon_lines"],
        )
        exit_code = 1
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
