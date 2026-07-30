"""Utilitaires transverses : logging, seed, device, entrees/sorties.

Aucune valeur absolue propre a une machine n'est codee en dur : la racine du
projet est deduite de l'emplacement du paquet et peut etre surchargee par la
variable d'environnement ``PPE_PROJECT_ROOT``.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

LOGGER_NAME = "ppe_detection"
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"})

_LOGGING_CONFIGURED = False


# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    """Retourne la racine du projet.

    Utilise ``PPE_PROJECT_ROOT`` si definie, sinon remonte depuis ce fichier
    (``<root>/src/ppe_detection/utils.py`` -> ``<root>``).
    """
    override = os.environ.get("PPE_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path, *, base: Path | None = None) -> Path:
    """Resout un chemin relatif par rapport a ``base`` (racine projet par defaut).

    Gere correctement les chemins Windows contenant des espaces car aucune
    manipulation textuelle n'est effectuee.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root = base if base is not None else project_root()
    return (root / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Cree le repertoire (et ses parents) s'il n'existe pas, puis le retourne."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_image(path: Path) -> bool:
    """True si l'extension correspond a un format image supporte."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video(path: Path) -> bool:
    """True si l'extension correspond a un format video supporte."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def iter_images(directory: Path) -> list[Path]:
    """Liste triee des images d'un repertoire (non recursive)."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and is_image(p))


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure le logger racine du paquet.

    Args:
        level: Niveau de log (``logging.INFO`` ou ``"DEBUG"``...).
        log_file: Fichier de log optionnel (en plus de la sortie console).
        force: Reconfigure meme si le logging a deja ete initialise.

    Returns:
        Le logger ``ppe_detection``.
    """
    global _LOGGING_CONFIGURED
    logger = logging.getLogger(LOGGER_NAME)
    if _LOGGING_CONFIGURED and not force:
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        ensure_dir(Path(log_file).parent)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGING_CONFIGURED = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Retourne un logger enfant du logger paquet."""
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name.rsplit('.', 1)[-1]}")


# --------------------------------------------------------------------------- #
# Reproductibilite
# --------------------------------------------------------------------------- #
def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Fixe les graines aleatoires Python/NumPy/PyTorch.

    Args:
        seed: Graine a appliquer.
        deterministic: Active les algorithmes deterministes cuDNN. Ralentit
            l'entrainement et peut lever une erreur si un operateur n'a pas
            d'implementation deterministe.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy est une dependance dure
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def resolve_device(requested: str = "auto") -> str:
    """Traduit une demande de device en valeur acceptee par Ultralytics.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"0"``, ``"0,1"``...

    Returns:
        ``"cpu"`` ou un index GPU sous forme de chaine (``"0"``).
        Retombe sur ``"cpu"`` avec un avertissement si CUDA est demande mais
        indisponible.
    """
    logger = get_logger(__name__)
    requested = (requested or "auto").strip().lower()

    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        cuda_available = False

    if requested in {"cpu"}:
        return "cpu"

    if requested in {"auto", ""}:
        return "0" if cuda_available else "cpu"

    if requested in {"cuda", "gpu"}:
        if cuda_available:
            return "0"
        logger.warning("CUDA demande mais indisponible — bascule sur CPU.")
        return "cpu"

    if requested == "mps":
        return "mps"

    # Index(es) GPU explicites
    if not cuda_available:
        logger.warning("Device '%s' demande mais CUDA est indisponible — bascule sur CPU.", requested)
        return "cpu"
    return requested


def describe_environment() -> dict[str, Any]:
    """Collecte les versions et capacites materielles pour la tracabilite."""
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gb"] = round(props.total_memory / 1024**3, 2)
    except ImportError:  # pragma: no cover
        info["torch"] = None
    for module_name in ("ultralytics", "cv2", "numpy", "onnxruntime"):
        try:
            module = __import__(module_name)
            info[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[module_name] = None
    return info


def pip_freeze() -> list[str]:
    """Retourne la liste des paquets installes (``pip freeze``).

    Retourne une liste vide si pip n'est pas invocable, sans lever d'exception.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Entrees/sorties
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    """Serialise les types non-JSON usuels (Path, dataclass, set, numpy)."""
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    for attr in ("tolist", "item"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return method()
            except (TypeError, ValueError):
                continue
    return str(obj)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Ecrit un objet en JSON UTF-8, en creant les repertoires manquants."""
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return target


def read_json(path: str | Path) -> Any:
    """Lit un fichier JSON UTF-8."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: str | Path, content: str) -> Path:
    """Ecrit du texte UTF-8, en creant les repertoires manquants."""
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(content, encoding="utf-8")
    return target


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Lit un fichier YAML et garantit un mapping en retour.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le contenu n'est pas un mapping.
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"Fichier YAML introuvable : {target}\n"
            f"Verifiez le chemin fourni (--data / --config)."
        )
    with target.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Le fichier YAML {target} doit contenir un mapping, pas {type(data).__name__}.")
    return data


def write_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    """Ecrit un mapping en YAML UTF-8."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return target


# --------------------------------------------------------------------------- #
# Formatage
# --------------------------------------------------------------------------- #
def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[Any]]) -> str:
    """Construit un tableau Markdown aligne.

    Args:
        headers: Libelles de colonnes.
        rows: Lignes de valeurs (converties en ``str``).

    Returns:
        Le tableau Markdown, ou une chaine vide s'il n'y a aucune colonne.
    """
    header_list = [str(h) for h in headers]
    if not header_list:
        return ""
    row_list = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in header_list]
    for row in row_list:
        for index, cell in enumerate(row[: len(widths)]):
            widths[index] = max(widths[index], len(cell))

    def render(cells: list[str]) -> str:
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(cells[: len(widths)])]
        return "| " + " | ".join(padded) + " |"

    lines = [render(header_list), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines.extend(render(row) for row in row_list)
    return "\n".join(lines)


def human_bytes(size: float) -> str:
    """Formate une taille en octets de maniere lisible."""
    for unit in ("o", "Ko", "Mo", "Go", "To"):
        if abs(size) < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} Po"


def safe_filename(name: str, *, fallback: str = "file", max_length: int = 96) -> str:
    """Neutralise un nom de fichier fourni par un utilisateur.

    Supprime toute composante de chemin et ne conserve que des caracteres surs,
    afin d'empecher une ecriture arbitraire via un nom d'upload malveillant
    (``../../etc/passwd``, noms UNC, flux ADS Windows...).

    Args:
        name: Nom potentiellement hostile.
        fallback: Nom utilise si rien d'exploitable ne subsiste.
        max_length: Longueur maximale du resultat.

    Returns:
        Un nom de fichier sans separateur ni caractere special.
    """
    candidate = str(name or "").replace("\\", "/").split("/")[-1]
    candidate = candidate.split(":")[-1]  # neutralise "C:" et les flux ADS
    cleaned = "".join(char if (char.isalnum() or char in "._-") else "_" for char in candidate)
    cleaned = cleaned.strip("._")
    if not cleaned:
        return fallback
    stem, dot, suffix = cleaned.rpartition(".")
    if dot and len(suffix) <= 8 and stem:
        return f"{stem[: max_length - len(suffix) - 1]}.{suffix}"
    return cleaned[:max_length]
