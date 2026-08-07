"""Tests de la fusion de datasets externes avec remappage de classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppe_detection.dataset_merge import (
    MergeError,
    merge_dataset,
    render_merge_report,
    resolve_split_dirs,
    write_target_yaml,
)
from ppe_detection.taxonomy import extended_schema
from ppe_detection.utils import write_yaml


def _make_source(
    root: Path, names: list[str], labels_by_image: dict[str, list[str]], split: str = "train"
) -> Path:
    """Cree un dataset YOLO minimal mais valide."""
    import cv2

    images = root / split / "images"
    labels = root / split / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    for stem, lines in labels_by_image.items():
        cv2.imwrite(str(images / f"{stem}.jpg"), np.full((64, 64, 3), 128, dtype=np.uint8))
        (labels / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    data_yaml = root / "data.yaml"
    write_yaml(
        data_yaml,
        {"train": f"{split}/images", "val": f"{split}/images", "nc": len(names), "names": names},
    )
    return data_yaml


def test_resolve_split_roboflow_layout(tmp_path: Path) -> None:
    """Disposition Roboflow : <split>/images."""
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir(parents=True)
    resolved = resolve_split_dirs(tmp_path, "train")
    assert resolved is not None
    assert resolved[0] == tmp_path / "train" / "images"


def test_resolve_split_fiftyone_layout(tmp_path: Path) -> None:
    """Disposition FiftyOne / YOLOv5 : images/<split>."""
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    resolved = resolve_split_dirs(tmp_path, "train")
    assert resolved is not None
    assert resolved[0] == tmp_path / "images" / "train"


def test_resolve_split_accepts_val_alias(tmp_path: Path) -> None:
    """'val' et 'validation' designent le meme split que 'valid'."""
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "labels" / "val").mkdir(parents=True)
    resolved = resolve_split_dirs(tmp_path, "valid")
    assert resolved is not None
    assert resolved[0].name == "val"


def test_resolve_split_returns_none_when_absent(tmp_path: Path) -> None:
    assert resolve_split_dirs(tmp_path, "test") is None


def test_merge_handles_fiftyone_layout(tmp_path: Path) -> None:
    """Fusion depuis un export FiftyOne, disposition images/<split>."""
    import cv2

    root = tmp_path / "oi"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "train" / "a.jpg"), np.full((64, 64, 3), 100, dtype=np.uint8))
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    write_yaml(
        root / "dataset.yaml",
        {"train": "./images/train/", "nc": 1, "names": {0: "Bicycle helmet"}},
    )

    stats = merge_dataset(
        root / "dataset.yaml",
        tmp_path / "out",
        {"Bicycle helmet": "Non-Safety Headwear"},
        source_name="oi",
    )
    assert stats.annotations_kept == 1
    assert stats.per_class["Non-Safety Headwear"] == 1


def test_merge_reads_dict_style_names(tmp_path: Path) -> None:
    """Les exports recents ecrivent names sous forme de dictionnaire indexe."""
    import cv2

    root = tmp_path / "d"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    cv2.imwrite(str(root / "train" / "images" / "a.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
    (root / "train" / "labels" / "a.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    write_yaml(
        root / "data.yaml",
        {"train": "train/images", "nc": 2, "names": {0: "zero", 1: "Hat"}},
    )
    stats = merge_dataset(
        root / "data.yaml", tmp_path / "o", {"Hat": "Non-Safety Headwear"}
    )
    assert stats.annotations_kept == 1


def test_merge_remaps_class_ids(tmp_path: Path) -> None:
    """Cas reel : la classe 0 'With Helmet' devient 'Non-Safety Headwear'."""
    source = _make_source(
        tmp_path / "bike",
        names=["With Helmet", "Without Helmet"],
        labels_by_image={"a": ["0 0.5 0.3 0.2 0.2"], "b": ["1 0.5 0.3 0.2 0.2"]},
    )
    target = tmp_path / "merged"
    stats = merge_dataset(
        source, target, {"With Helmet": "Non-Safety Headwear"}, source_name="bike"
    )

    assert stats.annotations_kept == 1
    assert stats.annotations_dropped == 1
    assert stats.per_class["Non-Safety Headwear"] == 1

    written = list((target / "train" / "labels").glob("*.txt"))
    assert len(written) == 2  # l'image 'b' est conservee comme negative
    kept = [p for p in written if p.read_text(encoding="utf-8").strip()]
    assert len(kept) == 1
    # 'Non-Safety Headwear' porte l'identifiant 7 dans le schema etendu.
    assert kept[0].read_text(encoding="utf-8").startswith("7 ")


def test_merge_keeps_empty_images_as_negatives(tmp_path: Path) -> None:
    """Une image dont toutes les annotations sont ecartees reste utile.

    Elle devient une image negative : le modele y apprend a ne rien signaler,
    ce qui reduit les faux positifs.
    """
    source = _make_source(
        tmp_path / "src",
        names=["helmet", "bicycle"],
        labels_by_image={"y": ["1 0.5 0.5 0.3 0.3"]},  # seul un velo, non mappe
    )
    stats = merge_dataset(
        source, tmp_path / "out", {"helmet": "Non-Safety Headwear"}, source_name="s"
    )
    assert stats.empty_labels == 1
    assert stats.images_copied == 1
    assert stats.annotations_kept == 0


def test_merge_drop_empty_option(tmp_path: Path) -> None:
    source = _make_source(
        tmp_path / "src",
        names=["helmet", "bicycle"],
        labels_by_image={"y": ["1 0.5 0.5 0.3 0.3"]},
    )
    stats = merge_dataset(
        source,
        tmp_path / "out",
        {"helmet": "Non-Safety Headwear"},
        source_name="s",
        keep_empty=False,
    )
    assert stats.images_copied == 0
    assert stats.images_skipped == 1


def test_merge_rejects_unknown_source_class(tmp_path: Path) -> None:
    """Erreur actionnable : la liste des classes disponibles est affichee."""
    source = _make_source(
        tmp_path / "src", names=["helmet"], labels_by_image={"a": ["0 0.5 0.5 0.2 0.2"]}
    )
    with pytest.raises(MergeError) as excinfo:
        merge_dataset(source, tmp_path / "out", {"Casque": "Non-Safety Headwear"})
    assert "helmet" in str(excinfo.value)


def test_merge_rejects_empty_mapping(tmp_path: Path) -> None:
    source = _make_source(
        tmp_path / "src", names=["helmet"], labels_by_image={"a": ["0 0.5 0.5 0.2 0.2"]}
    )
    with pytest.raises(MergeError, match="aucune classe"):
        merge_dataset(source, tmp_path / "out", {"helmet": ""})


def test_merge_dry_run_writes_nothing(tmp_path: Path) -> None:
    source = _make_source(
        tmp_path / "src", names=["helmet"], labels_by_image={"a": ["0 0.5 0.5 0.2 0.2"]}
    )
    target = tmp_path / "out"
    stats = merge_dataset(
        source, target, {"helmet": "Non-Safety Headwear"}, dry_run=True
    )
    assert stats.annotations_kept == 1
    assert not target.exists()


def test_merge_is_idempotent_on_filenames(tmp_path: Path) -> None:
    """Reimporter la meme source ne duplique pas les fichiers."""
    source = _make_source(
        tmp_path / "src", names=["helmet"], labels_by_image={"a": ["0 0.5 0.5 0.2 0.2"]}
    )
    target = tmp_path / "out"
    merge_dataset(source, target, {"helmet": "Non-Safety Headwear"}, source_name="s")
    first = sorted(p.name for p in (target / "train" / "images").iterdir())
    merge_dataset(source, target, {"helmet": "Non-Safety Headwear"}, source_name="s")
    second = sorted(p.name for p in (target / "train" / "images").iterdir())
    assert first == second


def test_merge_skips_polygon_lines(tmp_path: Path) -> None:
    """Les polygones ne sont pas geres ici : ils doivent passer par le cleaner."""
    source = _make_source(
        tmp_path / "src",
        names=["helmet"],
        labels_by_image={"a": ["0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2"]},
    )
    stats = merge_dataset(source, tmp_path / "out", {"helmet": "Non-Safety Headwear"})
    assert stats.annotations_kept == 0
    assert stats.annotations_dropped == 1


def test_write_target_yaml_uses_extended_schema(tmp_path: Path) -> None:
    path = write_target_yaml(tmp_path / "ds", extended_schema())
    from ppe_detection.utils import read_yaml

    payload = read_yaml(path)
    assert payload["nc"] == 10
    assert payload["names"][7] == "Non-Safety Headwear"


def test_report_flags_classes_without_instances(tmp_path: Path) -> None:
    """Un schema a 10 classes dont 3 sont vides doit etre signale."""
    source = _make_source(
        tmp_path / "src", names=["helmet"], labels_by_image={"a": ["0 0.5 0.5 0.2 0.2"]}
    )
    stats = merge_dataset(source, tmp_path / "out", {"helmet": "Non-Safety Headwear"})
    report = render_merge_report([stats], extended_schema())
    assert "Classes sans aucune instance" in report
    assert "Safety Helmet" in report
