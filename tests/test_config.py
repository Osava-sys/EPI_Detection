"""Tests de la configuration, de l'audit et du nettoyage sur dataset synthetique."""

from __future__ import annotations

from pathlib import Path

import pytest

from ppe_detection.config import (
    ComplianceConfig,
    ConfigError,
    InferenceConfig,
    load_compliance_config,
    load_dataset_config,
    load_inference_config,
    load_train_config,
)
from ppe_detection.dataset_audit import (
    audit_dataset,
    cluster_pairs,
    find_near_duplicates,
    roboflow_source_stem,
    sequence_prefix,
)
from ppe_detection.dataset_cleaner import CleaningError, clean_dataset
from ppe_detection.utils import markdown_table, safe_filename, write_yaml


# --------------------------------------------------------------------------- #
# DatasetConfig
# --------------------------------------------------------------------------- #
def test_load_dataset_config_resolves_splits(synthetic_dataset: Path) -> None:
    dataset = load_dataset_config(synthetic_dataset)
    assert dataset.num_classes == 7
    assert set(dataset.splits) == {"train", "valid", "test"}
    assert dataset.splits["train"].is_dir()
    assert dataset.labels_dir("train").name == "labels"
    assert dataset.class_name(1) == "Person"
    assert dataset.class_name(999).startswith("unknown")


def test_roboflow_relative_paths_are_resolved(roboflow_style_dataset: Path) -> None:
    """Les chemins `../train/images` de Roboflow doivent rester exploitables."""
    dataset = load_dataset_config(roboflow_style_dataset)
    assert set(dataset.splits) == {"train", "valid", "test"}
    for path in dataset.splits.values():
        assert path.is_dir()


def test_names_as_mapping_is_supported(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    (tmp_path / "train" / "images").mkdir(parents=True)
    write_yaml(
        data_yaml,
        {"path": str(tmp_path), "train": "train/images", "names": {0: "A", 1: "B"}},
    )
    dataset = load_dataset_config(data_yaml)
    assert dataset.names == ["A", "B"]


def test_missing_names_raises(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    write_yaml(data_yaml, {"train": "train/images", "nc": 2})
    with pytest.raises(ConfigError, match="names"):
        load_dataset_config(data_yaml)


def test_no_usable_split_raises(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    write_yaml(data_yaml, {"train": "nowhere/images", "names": ["A"]})
    with pytest.raises(ConfigError, match="Aucun split"):
        load_dataset_config(data_yaml)


# --------------------------------------------------------------------------- #
# Configurations d'entrainement / inference / conformite
# --------------------------------------------------------------------------- #
def test_train_config_defaults_and_overrides() -> None:
    config = load_train_config(None, epochs=7, imgsz=512)
    assert config.epochs == 7
    assert config.imgsz == 512
    kwargs = config.to_ultralytics_kwargs()
    assert "model" not in kwargs
    assert Path(kwargs["data"]).is_absolute()


def test_train_config_ignores_none_overrides() -> None:
    config = load_train_config(None, epochs=None)
    assert config.epochs == 100


def test_project_train_yaml_is_valid() -> None:
    """La configuration livree doit se charger sans erreur."""
    from ppe_detection.config import default_config_path

    path = default_config_path("train.yaml")
    if not path.is_file():
        pytest.skip("configs/train.yaml absent")
    config = load_train_config(path)
    assert config.imgsz > 0
    assert config.epochs > 0
    assert config.model.endswith(".pt")


def test_inference_class_threshold_can_only_tighten() -> None:
    config = InferenceConfig(conf=0.30, class_conf={"Face Mask": 0.10, "Person": 0.80})
    # Un seuil par classe plus bas que le seuil global ne peut pas l'assouplir.
    assert config.threshold_for("Face Mask") == 0.30
    assert config.threshold_for("Person") == 0.80
    assert config.threshold_for("Inconnue") == 0.30


def test_project_inference_yaml_is_valid() -> None:
    from ppe_detection.config import default_config_path

    path = default_config_path("inference.yaml")
    if not path.is_file():
        pytest.skip("configs/inference.yaml absent")
    inference = load_inference_config(path)
    compliance = load_compliance_config(path)
    assert 0.0 < inference.conf < 1.0
    assert compliance.person_class == "Person"
    assert "Safety Helmet" in compliance.required_ppe
    assert compliance.region_by_class["Safety Shoes"] == "feet"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"containment_threshold": 0.0},
        {"containment_threshold": 1.5},
        {"helmet_region": 0.0},
        {"torso_region": (0.8, 0.2)},
    ],
)
def test_compliance_config_validates_bounds(kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        ComplianceConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def test_audit_detects_seeded_problems(synthetic_dataset: Path) -> None:
    report = audit_dataset(synthetic_dataset, workers=2, compute_phash=False)
    totals = report["totals"]

    assert totals["images"] == 8
    assert totals["polygon_lines"] == 1
    assert report["splits"]["train"]["issues"].get("class_id_out_of_range") == 1
    assert report["splits"]["train"]["issues"].get("non_numeric") == 1
    assert report["splits"]["train"]["issues"].get("clipped_to_bounds") == 1
    assert report["splits"]["valid"]["counts"]["empty_label_files"] == 1
    assert report["conclusion"]["requires_conversion"] is True
    assert report["conclusion"]["detection_ready"] is False


def test_audit_reports_perfect_pairing(synthetic_dataset: Path) -> None:
    report = audit_dataset(synthetic_dataset, workers=2, compute_phash=False)
    for split in report["splits"].values():
        assert split["pairing"]["perfect_match"] is True


def test_audit_flags_orphan_label(synthetic_dataset: Path) -> None:
    labels_dir = synthetic_dataset.parent / "train" / "labels"
    (labels_dir / "ghost.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    report = audit_dataset(synthetic_dataset, workers=2, compute_phash=False)
    assert report["totals"]["labels_without_image"] == 1
    assert any(p["code"] == "labels_without_image" for p in report["problems"])


# --------------------------------------------------------------------------- #
# Utilitaires d'audit
# --------------------------------------------------------------------------- #
def test_roboflow_source_stem_extracts_original_name() -> None:
    assert roboflow_source_stem("photo_jpg.rf.abc123.jpg") == "photo_jpg"
    assert roboflow_source_stem("plain.jpg") == "plain"


def test_sequence_prefix_recognises_video_frames() -> None:
    assert sequence_prefix("frame_000324_jpg.rf.abc.jpg") == "frame_"
    assert sequence_prefix("random.jpg") is None


def test_find_near_duplicates_detects_close_hashes() -> None:
    # Deux hashes ne differant que d'un bit
    entries = [("a", "0000000000000000"), ("b", "0000000000000001"), ("c", "ffffffffffffffff")]
    pairs = find_near_duplicates(entries, max_distance=3)
    found = {(a, b) for a, b, _ in pairs}
    assert ("a", "b") in found
    assert not any("c" in pair for pair in found)


def test_cluster_pairs_groups_connected_components() -> None:
    clusters = cluster_pairs([("a", "b", 1), ("b", "c", 2), ("x", "y", 1)])
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 3]


# --------------------------------------------------------------------------- #
# Nettoyage
# --------------------------------------------------------------------------- #
def test_clean_dataset_produces_only_five_field_labels(synthetic_dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "derived"
    report = clean_dataset(synthetic_dataset, output, mode="copy")

    assert report["statistics"]["lines_converted_from_polygon"] == 1
    # class_id hors limites + champ non numerique
    assert report["statistics"]["lines_dropped"] == 2

    for label_file in (output).rglob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                assert len(line.split()) == 5, f"{label_file}: {line!r}"

    assert (output / "data.yaml").is_file()


def test_clean_dataset_leaves_source_untouched(synthetic_dataset: Path, tmp_path: Path) -> None:
    before = {
        path: path.read_bytes()
        for path in synthetic_dataset.parent.rglob("*")
        if path.is_file()
    }
    clean_dataset(synthetic_dataset, tmp_path / "derived", mode="copy")
    after = {
        path: path.read_bytes()
        for path in synthetic_dataset.parent.rglob("*")
        if path.is_file()
    }
    assert before == after


def test_clean_dataset_dry_run_writes_nothing(synthetic_dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "derived"
    report = clean_dataset(synthetic_dataset, output, dry_run=True)
    assert report["statistics"]["lines_kept"] > 0
    assert not output.exists() or not any(output.iterdir())


def test_clean_dataset_requires_overwrite(synthetic_dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "derived"
    clean_dataset(synthetic_dataset, output, mode="copy")
    with pytest.raises(CleaningError, match="overwrite"):
        clean_dataset(synthetic_dataset, output, mode="copy")
    # Avec --overwrite, l'operation aboutit.
    report = clean_dataset(synthetic_dataset, output, mode="copy", overwrite=True)
    assert report["statistics"]["lines_kept"] > 0


def test_clean_dataset_strict_fails_on_dropped_lines(synthetic_dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(CleaningError, match="strict"):
        clean_dataset(synthetic_dataset, tmp_path / "strict", strict=True)


def test_clean_dataset_rejects_unknown_mode(synthetic_dataset: Path, tmp_path: Path) -> None:
    with pytest.raises(CleaningError, match="Mode inconnu"):
        clean_dataset(synthetic_dataset, tmp_path / "x", mode="hardlink")


def test_clean_dataset_refuses_to_overwrite_source(synthetic_dataset: Path) -> None:
    with pytest.raises(CleaningError, match="identique"):
        clean_dataset(synthetic_dataset, synthetic_dataset.parent, overwrite=True)


def test_cleaned_dataset_passes_audit(synthetic_dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "derived"
    clean_dataset(synthetic_dataset, output, mode="copy")
    report = audit_dataset(output / "data.yaml", workers=2, compute_phash=False)
    assert report["totals"]["polygon_lines"] == 0
    assert report["totals"]["malformed_lines"] == 0
    assert report["conclusion"]["detection_ready"] is True


def test_regroup_by_source_removes_cross_split_leak(tmp_path: Path) -> None:
    """Deux variantes d'une meme image source doivent finir dans le meme split."""
    import cv2
    import numpy as np

    root = tmp_path / "leaky"
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    for split, suffix in (("train", "aaa"), ("valid", "bbb")):
        name = f"shared_jpg.rf.{suffix}"
        cv2.imwrite(str(root / split / "images" / f"{name}.jpg"), image)
        (root / split / "labels" / f"{name}.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    data_yaml = root / "data.yaml"
    write_yaml(
        data_yaml,
        {
            "path": str(root),
            "train": "train/images",
            "val": "valid/images",
            "nc": 7,
            "names": ["Face Mask", "Person", "Safety Gloves", "Safety Harness",
                      "Safety Helmet", "Safety Shoes", "Safety Vest"],
        },
    )

    without = clean_dataset(data_yaml, tmp_path / "keep", regroup_by_source=False)
    assert without["regrouping"]["groups_spanning_splits"] == 1
    assert without["regrouping"]["files_reassigned"] == 0

    with_regroup = clean_dataset(data_yaml, tmp_path / "regrouped", regroup_by_source=True)
    assert with_regroup["regrouping"]["files_reassigned"] == 1
    counts = with_regroup["output"]["images_per_split"]
    assert sorted(counts.values()) == [2] or list(counts.values()) == [2]


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "hostile,forbidden",
    [
        ("../../etc/passwd", ".."),
        ("..\\..\\windows\\system32\\cmd.exe", ".."),
        ("C:\\secret\\file.jpg", ":"),
        ("normal.jpg:hidden.txt", ":"),
    ],
)
def test_safe_filename_neutralises_paths(hostile: str, forbidden: str) -> None:
    result = safe_filename(hostile)
    assert forbidden not in result
    assert "/" not in result
    assert "\\" not in result


def test_safe_filename_falls_back_when_empty() -> None:
    assert safe_filename("...") == "file"
    assert safe_filename("") == "file"


def test_safe_filename_keeps_reasonable_names() -> None:
    assert safe_filename("photo_01.jpg") == "photo_01.jpg"


def test_markdown_table_alignment() -> None:
    table = markdown_table(["A", "BB"], [[1, 2], [333, 4]])
    lines = table.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("| A")
    assert set(lines[1]) <= {"|", "-"}
