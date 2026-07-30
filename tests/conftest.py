"""Fixtures partagees : datasets synthetiques et poids reels optionnels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppe_detection.utils import project_root, write_yaml

CLASS_NAMES = [
    "Face Mask",
    "Person",
    "Safety Gloves",
    "Safety Harness",
    "Safety Helmet",
    "Safety Shoes",
    "Safety Vest",
]


@pytest.fixture
def class_names() -> list[str]:
    """Les 7 classes EPI du projet."""
    return list(CLASS_NAMES)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Construit un mini dataset YOLO synthetique couvrant tous les cas limites.

    Contenu par split :

    * ``train`` — une boite valide, un polygone convertible, une classe hors
      limites, un champ non numerique, une boite debordant du cadre ;
    * ``valid`` — une boite valide et un fichier de labels vide ;
    * ``test``  — une boite valide.

    Returns:
        Le chemin du ``data.yaml`` genere.
    """
    import cv2

    root = tmp_path / "dataset"
    contents: dict[str, dict[str, str]] = {
        "train": {
            "img_valid": "1 0.5 0.5 0.2 0.4\n",
            "img_polygon": "4 0.10 0.10 0.30 0.12 0.32 0.28 0.09 0.26\n",
            "img_badclass": "1 0.5 0.5 0.2 0.2\n99 0.5 0.5 0.1 0.1\n",
            "img_nonnumeric": "2 abc 0.5 0.1 0.1\n",
            "img_overflow": "5 0.95 0.5 0.20 0.30\n",
        },
        "valid": {
            "img_v1": "1 0.4 0.4 0.3 0.3\n",
            "img_empty": "",
        },
        "test": {
            "img_t1": "6 0.5 0.5 0.25 0.25\n",
        },
    }

    rng = np.random.default_rng(1234)
    for split, files in contents.items():
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for stem, label_text in files.items():
            image = rng.integers(0, 255, size=(96, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / f"{stem}.jpg"), image)
            (labels_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")

    data_yaml = root / "data.yaml"
    write_yaml(
        data_yaml,
        {
            "path": str(root),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(CLASS_NAMES),
            "names": CLASS_NAMES,
        },
    )
    return data_yaml


@pytest.fixture
def roboflow_style_dataset(tmp_path: Path) -> Path:
    """Dataset dont les chemins imitent l'export Roboflow (``../train/images``).

    Sert a verifier que la resolution des chemins tolere cette convention.
    """
    import cv2

    root = tmp_path / "roboflow"
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(root / split / "images" / "a_jpg.rf.deadbeef.jpg"), image)
        (root / split / "labels" / "a_jpg.rf.deadbeef.txt").write_text(
            "1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "\n"
        "nc: 7\n"
        f"names: {CLASS_NAMES}\n",
        encoding="utf-8",
    )
    return data_yaml


def _find_weights() -> Path | None:
    """Cherche des poids reels utilisables pour les tests d'integration."""
    models_dir = project_root() / "artifacts" / "models"
    if not models_dir.is_dir():
        return None
    for candidate in ("best.pt", "smoke_best.pt", "last.pt", "smoke_last.pt"):
        path = models_dir / candidate
        if path.is_file():
            return path
    return None


@pytest.fixture(scope="session")
def real_weights() -> Path:
    """Poids reels si disponibles, sinon le test appelant est ignore."""
    weights = _find_weights()
    if weights is None:
        pytest.skip(
            "Aucun poids disponible dans artifacts/models — "
            "executez d'abord `python -m ppe_detection.train --smoke`."
        )
    return weights


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Petite image JPEG synthetique."""
    import cv2

    path = tmp_path / "sample.jpg"
    rng = np.random.default_rng(7)
    cv2.imwrite(str(path), rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8))
    return path
