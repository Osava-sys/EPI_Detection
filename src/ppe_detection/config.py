"""Configuration centralisee du projet (dataset, entrainement, inference, conformite).

Toutes les configurations sont des dataclasses typees, chargeables depuis YAML et
surchargeables par arguments CLI. Aucun chemin absolu propre a une machine n'est
code en dur.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from .utils import get_logger, project_root, read_yaml, resolve_path

LOGGER = get_logger(__name__)

SPLIT_KEYS: dict[str, str] = {"train": "train", "valid": "val", "test": "test"}
"""Nom de repertoire de split -> cle correspondante dans ``data.yaml``."""

T = TypeVar("T")


class ConfigError(ValueError):
    """Erreur de configuration avec un message actionnable."""


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    """Description resolue d'un dataset YOLO.

    Attributes:
        yaml_path: Chemin du ``data.yaml`` source.
        root: Repertoire racine du dataset (parent du ``data.yaml``).
        names: Noms de classes indexes par identifiant.
        splits: Nom de split -> repertoire d'images resolu (uniquement les
            splits reellement presents sur le disque).
    """

    yaml_path: Path
    root: Path
    names: list[str]
    splits: dict[str, Path]

    @property
    def num_classes(self) -> int:
        """Nombre de classes declarees."""
        return len(self.names)

    def labels_dir(self, split: str) -> Path:
        """Repertoire des labels associe au repertoire d'images d'un split.

        Applique la convention Ultralytics : le segment ``images`` du chemin est
        remplace par ``labels``.
        """
        images_dir = self.splits[split]
        parts = list(images_dir.parts)
        for index in range(len(parts) - 1, -1, -1):
            if parts[index].lower() == "images":
                parts[index] = "labels"
                return Path(*parts)
        return images_dir.parent / "labels"

    def class_name(self, class_id: int) -> str:
        """Nom lisible d'une classe, avec repli explicite si l'id est inconnu."""
        if 0 <= class_id < len(self.names):
            return self.names[class_id]
        return f"unknown_{class_id}"


def _coerce_names(raw: Any) -> list[str]:
    """Normalise le champ ``names`` (liste ou mapping id -> nom) en liste ordonnee."""
    if isinstance(raw, Mapping):
        try:
            indexed = {int(key): str(value) for key, value in raw.items()}
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Cles de 'names' non entieres dans data.yaml : {raw!r}") from exc
        if not indexed:
            return []
        expected = set(range(max(indexed) + 1))
        missing = expected - set(indexed)
        if missing:
            raise ConfigError(
                f"'names' incomplet dans data.yaml : identifiants manquants {sorted(missing)}."
            )
        return [indexed[i] for i in range(max(indexed) + 1)]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item) for item in raw]
    raise ConfigError(
        f"Champ 'names' invalide dans data.yaml : attendu une liste ou un mapping, recu {type(raw).__name__}."
    )


def _candidate_dirs(raw_value: str, yaml_dir: Path, declared_root: Path | None) -> list[Path]:
    """Construit les resolutions plausibles d'un chemin de split.

    Les exports Roboflow ecrivent ``../train/images`` en supposant une racine
    differente de celle utilisee par Ultralytics. On teste donc plusieurs
    interpretations et on retient la premiere qui existe reellement.
    """
    raw_path = Path(raw_value)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        bases = [declared_root] if declared_root is not None else []
        bases.append(yaml_dir)
        stripped = Path(*[part for part in raw_path.parts if part not in {"..", "."}])
        for base in bases:
            candidates.append(base / raw_path)
            if stripped.parts:
                candidates.append(base / stripped)
    # Deduplique en preservant l'ordre de preference.
    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:  # pragma: no cover - chemins pathologiques Windows
            continue
        key = str(resolved).lower()
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def load_dataset_config(yaml_path: str | Path) -> DatasetConfig:
    """Charge et valide un ``data.yaml`` YOLO.

    Resout chaque split en testant plusieurs interpretations du chemin relatif,
    ce qui rend la fonction tolerante aux exports Roboflow dont les chemins
    (``../train/images``) ne se resolvent pas correctement depuis la racine du
    dataset.

    Args:
        yaml_path: Chemin du fichier ``data.yaml``.

    Returns:
        La configuration dataset resolue.

    Raises:
        ConfigError: Si le fichier est invalide ou si aucun split n'est trouvable.
    """
    path = resolve_path(yaml_path)
    raw = read_yaml(path)
    yaml_dir = path.parent

    if "names" not in raw:
        raise ConfigError(f"Champ 'names' absent de {path}.")
    names = _coerce_names(raw["names"])
    if not names:
        raise ConfigError(f"Aucune classe declaree dans {path}.")

    declared_nc = raw.get("nc")
    if declared_nc is not None and int(declared_nc) != len(names):
        LOGGER.warning(
            "Incoherence dans %s : nc=%s mais %d noms de classes. 'names' fait foi.",
            path.name,
            declared_nc,
            len(names),
        )

    declared_root: Path | None = None
    if raw.get("path"):
        declared_root = resolve_path(str(raw["path"]), base=yaml_dir)

    splits: dict[str, Path] = {}
    for split_dir_name, yaml_key in SPLIT_KEYS.items():
        raw_value = raw.get(yaml_key)
        if not raw_value:
            continue
        if isinstance(raw_value, (list, tuple)):
            raw_value = raw_value[0] if raw_value else None
        if not raw_value:
            continue
        for candidate in _candidate_dirs(str(raw_value), yaml_dir, declared_root):
            if candidate.is_dir():
                splits[split_dir_name] = candidate
                break
        else:
            LOGGER.warning(
                "Split '%s' declare dans %s (%r) mais aucun repertoire correspondant n'existe.",
                split_dir_name,
                path.name,
                raw_value,
            )

    if not splits:
        raise ConfigError(
            f"Aucun split exploitable dans {path}.\n"
            f"Verifiez que les repertoires train/valid/test existent a cote du data.yaml."
        )

    root = declared_root if declared_root is not None else yaml_dir
    return DatasetConfig(yaml_path=path, root=root, names=names, splits=splits)


# --------------------------------------------------------------------------- #
# Chargement generique de dataclasses depuis YAML
# --------------------------------------------------------------------------- #
def _from_mapping(cls: type[T], data: Mapping[str, Any], *, context: str) -> T:
    """Instancie une dataclass depuis un mapping en ignorant les cles inconnues."""
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        LOGGER.warning("Cles ignorees dans %s : %s", context, ", ".join(sorted(unknown)))
    return cls(**{key: value for key, value in data.items() if key in known})  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# Entrainement
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Hyperparametres d'entrainement (valeurs par defaut = configs/train.yaml)."""

    data: str = "artifacts/dataset_detection/data.yaml"
    model: str = "yolo26s.pt"
    imgsz: int = 640
    epochs: int = 100
    batch: float | int = -1  # -1 = auto-batch Ultralytics (~60 % VRAM)
    device: str = "auto"
    workers: int = 8
    seed: int = 42
    deterministic: bool = True
    patience: int = 25
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    amp: bool = True
    cache: str | bool = False
    cos_lr: bool = False
    close_mosaic: int = 10
    fraction: float = 1.0
    val: bool = True
    plots: bool = True
    save_period: int = -1
    project: str = "artifacts/runs"
    name: str = "ppe_yolo26s"
    exist_ok: bool = False
    resume: bool = False
    pretrained: bool = True
    # Augmentations
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    erasing: float = 0.0

    def to_ultralytics_kwargs(self) -> dict[str, Any]:
        """Convertit la config en kwargs pour ``YOLO.train()``.

        ``model`` est exclu (passe au constructeur) et ``data`` est resolu en
        chemin absolu pour rester valide quel que soit le repertoire courant.
        """
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        payload.pop("model")
        payload["data"] = str(resolve_path(self.data))
        payload["project"] = str(resolve_path(self.project))
        return payload


def load_train_config(config_path: str | Path | None = None, **overrides: Any) -> TrainConfig:
    """Charge ``configs/train.yaml`` puis applique les surcharges CLI non nulles."""
    data: dict[str, Any] = {}
    if config_path is not None:
        data = dict(read_yaml(resolve_path(config_path)))
    config = _from_mapping(TrainConfig, data, context=str(config_path or "defaults"))
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(config, key):
            LOGGER.warning("Surcharge CLI inconnue ignoree : %s", key)
            continue
        setattr(config, key, value)
    return config


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    """Parametres d'inference et de rendu."""

    weights: str = "artifacts/models/best.pt"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: str = "auto"
    max_det: int = 300
    half: bool = False
    agnostic_nms: bool = False
    # Tracker Ultralytics utilise en mode suivi : bytetrack.yaml ou botsort.yaml.
    # ByteTrack est plus rapide ; BoT-SORT est plus robuste aux occlusions car il
    # ajoute une re-identification par apparence.
    tracker: str = "bytetrack.yaml"
    class_conf: dict[str, float] = field(default_factory=dict)
    show_labels: bool = True
    show_conf: bool = True
    line_width: int | None = None
    font_scale: float = 0.5

    def threshold_for(self, class_name: str) -> float:
        """Seuil de confiance applicable a une classe donnee.

        Le seuil par classe surcharge le seuil global. Comme Ultralytics filtre
        deja a ``conf``, un seuil par classe ne peut que **durcir** le filtrage.
        """
        return max(self.class_conf.get(class_name, self.conf), self.conf)


def load_inference_config(config_path: str | Path | None = None, **overrides: Any) -> InferenceConfig:
    """Charge ``configs/inference.yaml`` puis applique les surcharges CLI non nulles."""
    data: dict[str, Any] = {}
    if config_path is not None:
        path = resolve_path(config_path)
        if path.is_file():
            data = dict(read_yaml(path))
        else:
            LOGGER.warning("Config d'inference introuvable (%s) — valeurs par defaut utilisees.", path)
    inference_section = data.get("inference", data)
    if not isinstance(inference_section, dict):
        inference_section = {}
    config = _from_mapping(InferenceConfig, inference_section, context=str(config_path or "defaults"))
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(config, key):
            LOGGER.warning("Surcharge CLI inconnue ignoree : %s", key)
            continue
        setattr(config, key, value)
    return config


# --------------------------------------------------------------------------- #
# Conformite EPI
# --------------------------------------------------------------------------- #
@dataclass
class ComplianceConfig:
    """Regles metier d'association EPI <-> personne.

    Attributes:
        enabled: Active la couche de conformite.
        person_class: Nom exact de la classe "personne" dans ``data.yaml``.
        required_ppe: EPI obligatoires pour qu'une personne soit conforme.
        containment_threshold: Fraction minimale de la boite EPI devant se
            trouver dans la region attendue de la personne.
        helmet_region: Hauteur relative de la zone "tete" (haut de la personne).
        shoes_region: Hauteur relative de la zone "pieds" (bas de la personne).
        torso_region: Bornes relatives (debut, fin) de la zone "torse".
        region_by_class: Zone attendue par classe d'EPI.
        min_person_conf: Confiance minimale pour considerer une personne.
        min_ppe_conf: Confiance minimale pour considerer un EPI.
        counter_evidence: EPI requis -> classes « sosies » dont la detection
            prouve que l'EPI n'est pas porte (casquette a la place d'un casque).
            Une contre-preuve est une constatation, pas une absence de preuve.
        min_region_height_px: Hauteur en pixels minimale d'une zone (tete, torse,
            pieds) pour que l'absence d'un EPI y soit affirmable. En dessous,
            le verdict est *indetermine* plutot que *non conforme* : a cette
            echelle le detecteur ne peut tout simplement pas resoudre l'objet.
        edge_margin_px: Distance au bord du cadre en deca de laquelle une boite
            est consideree comme tronquee. Une zone tronquee n'est pas observable.
        temporal_window: Nombre de verdicts conserves par personne suivie.
        temporal_min_ratio: Proportion de verdicts concordants exigee pour
            trancher (lissage temporel).
        temporal_min_observations: Nombre minimal de verdicts exploitables avant
            de conclure sur une personne suivie.
    """

    enabled: bool = False
    person_class: str = "Person"
    required_ppe: list[str] = field(default_factory=lambda: ["Safety Helmet", "Safety Vest"])
    containment_threshold: float = 0.50
    helmet_region: float = 0.35
    shoes_region: float = 0.30
    torso_region: tuple[float, float] = (0.20, 0.80)
    region_by_class: dict[str, str] = field(
        default_factory=lambda: {
            "Safety Helmet": "head",
            "Face Mask": "head",
            "Safety Vest": "torso",
            "Safety Harness": "torso",
            "Safety Gloves": "any",
            "Safety Shoes": "feet",
        }
    )
    min_person_conf: float = 0.25
    min_ppe_conf: float = 0.25
    # Classes dont le detecteur n'est pas assez fiable pour fonder une alerte.
    # Les inscrire dans required_ppe declenche un avertissement au chargement.
    unreliable_ppe: list[str] = field(default_factory=lambda: ["Safety Gloves"])
    # --- contre-preuve (taxonomie etendue a 10 classes) ---
    # {EPI requis: [classes dont la presence prouve qu'il n'est pas porte]}.
    # Detecter un couvre-chef non conforme est une preuve POSITIVE de violation,
    # bien plus fiable qu'une simple absence de detection.
    # Vide par defaut : sans modele entraine sur les classes negatives, aucune
    # contre-preuve ne peut apparaitre et le mecanisme reste inerte.
    counter_evidence: dict[str, list[str]] = field(default_factory=dict)
    # --- observabilite (niveau 1) ---
    min_region_height_px: float = 24.0
    edge_margin_px: float = 2.0
    # --- lissage temporel (niveau 2) ---
    temporal_window: int = 15
    temporal_min_ratio: float = 0.70
    temporal_min_observations: int = 5

    def __post_init__(self) -> None:
        """Valide les bornes et normalise ``torso_region``."""
        if not 0.0 < self.containment_threshold <= 1.0:
            raise ConfigError(
                f"containment_threshold doit etre dans ]0, 1], recu {self.containment_threshold}."
            )
        for name in ("helmet_region", "shoes_region"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ConfigError(f"{name} doit etre dans ]0, 1], recu {value}.")
        torso = tuple(float(v) for v in self.torso_region)
        if len(torso) != 2 or not 0.0 <= torso[0] < torso[1] <= 1.0:
            raise ConfigError(
                f"torso_region doit etre (debut, fin) avec 0 <= debut < fin <= 1, "
                f"recu {self.torso_region}."
            )
        self.torso_region = (torso[0], torso[1])
        if self.min_region_height_px < 0:
            raise ConfigError(
                f"min_region_height_px doit etre >= 0, recu {self.min_region_height_px}."
            )
        if self.edge_margin_px < 0:
            raise ConfigError(f"edge_margin_px doit etre >= 0, recu {self.edge_margin_px}.")
        if self.temporal_window < 1:
            raise ConfigError(f"temporal_window doit etre >= 1, recu {self.temporal_window}.")
        if not 0.0 < self.temporal_min_ratio <= 1.0:
            raise ConfigError(
                f"temporal_min_ratio doit etre dans ]0, 1], recu {self.temporal_min_ratio}."
            )
        if self.temporal_min_observations < 1:
            raise ConfigError(
                f"temporal_min_observations doit etre >= 1, recu {self.temporal_min_observations}."
            )
        # Avertissement, pas erreur : le choix reste a l'utilisateur, mais il doit
        # etre pris en connaissance de cause.
        risky = [name for name in self.required_ppe if name in self.unreliable_ppe]
        if risky:
            LOGGER.warning(
                "EPI requis peu fiables : %s. Le detecteur les rate frequemment, "
                "ce qui produira des alertes 'non conforme' largement erronees. "
                "Retirez-les de required_ppe ou ameliorez le modele avant de vous "
                "appuyer sur ces regles.",
                ", ".join(risky),
            )


def load_compliance_config(config_path: str | Path | None = None, **overrides: Any) -> ComplianceConfig:
    """Charge la section ``compliance`` d'un YAML puis applique les surcharges."""
    data: dict[str, Any] = {}
    if config_path is not None:
        path = resolve_path(config_path)
        if path.is_file():
            raw = read_yaml(path)
            section = raw.get("compliance", raw)
            if isinstance(section, dict):
                data = dict(section)
                association = data.pop("association", None)
                if isinstance(association, dict):
                    data.update(association)
        else:
            LOGGER.warning("Config de conformite introuvable (%s) — valeurs par defaut utilisees.", path)
    config = _from_mapping(ComplianceConfig, data, context=str(config_path or "defaults"))
    for key, value in overrides.items():
        if value is None:
            continue
        if hasattr(config, key):
            setattr(config, key, value)
    config.__post_init__()
    return config


def default_config_path(name: str) -> Path:
    """Chemin d'un fichier de ``configs/`` (ex. ``"train.yaml"``)."""
    return project_root() / "configs" / name
