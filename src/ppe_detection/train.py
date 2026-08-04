"""Entrainement du detecteur EPI via l'API Python d'Ultralytics.

Usage :

    python -m ppe_detection.train --config configs/train.yaml
    python -m ppe_detection.train --config configs/train.yaml --smoke
    python -m ppe_detection.train --resume artifacts/runs/ppe_yolo26s/weights/last.pt

Le script gere explicitement l'absence de GPU, la saturation memoire, la reprise
depuis un checkpoint et l'interruption clavier, et archive systematiquement la
configuration resolue, les versions de dependances et la graine utilisee.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ConfigError, TrainConfig, default_config_path, load_train_config
from .utils import (
    describe_environment,
    ensure_dir,
    get_logger,
    pip_freeze,
    project_root,
    resolve_device,
    resolve_path,
    set_seed,
    setup_logging,
    write_json,
    write_yaml,
)

LOGGER = get_logger(__name__)

SMOKE_OVERRIDES: dict[str, Any] = {
    "epochs": 2,
    "fraction": 0.04,
    "batch": 8,
    "patience": 0,
    "name": "smoke_test",
    "exist_ok": True,
    "close_mosaic": 0,
    "workers": 4,
    "plots": True,
    "val": True,
}
"""Surcharges appliquees en mode ``--smoke`` : valide le pipeline en minutes."""


class TrainingError(RuntimeError):
    """Erreur bloquante durant l'entrainement."""


def _check_dataset(config: TrainConfig) -> Path:
    """Verifie que le dataset d'entrainement existe et est exploitable.

    Raises:
        TrainingError: Si le ``data.yaml`` est introuvable, avec un message
            indiquant la commande de generation a executer.
    """
    data_path = resolve_path(config.data)
    if not data_path.is_file():
        raise TrainingError(
            f"Dataset introuvable : {data_path}\n"
            f"Generez d'abord le dataset de detection normalise :\n"
            f"  python -m ppe_detection.dataset_cleaner --source data.yaml "
            f"--output artifacts/dataset_detection"
        )
    return data_path


def _resolve_batch(config: TrainConfig, device: str) -> float | int:
    """Adapte la taille de lot au device reellement utilise.

    Ultralytics accepte trois formes : un entier explicite, ``-1`` pour
    l'auto-batch, ou une fraction de VRAM entre 0 et 1. Un entier fourni en
    ligne de commande arrive sous forme de ``float`` (``12.0``) ; le
    ``DataLoader`` de PyTorch exige un entier strict et leve sinon
    ``ValueError: batch_size should be a positive integer value``. On
    reconvertit donc les valeurs >= 1 en entier, tout en preservant la
    semantique fractionnaire.

    L'auto-batch mesure l'occupation VRAM et n'a aucun sens sur CPU : on
    retombe alors sur une valeur modeste.
    """
    batch = config.batch
    if device == "cpu" and (isinstance(batch, (int, float)) and batch < 0):
        LOGGER.warning("Auto-batch indisponible sur CPU — batch=4 utilise.")
        return 4
    if isinstance(batch, float) and batch >= 1 and batch.is_integer():
        return int(batch)
    return batch


def _archive_run_metadata(run_dir: Path, config: TrainConfig, device: str, smoke: bool) -> None:
    """Archive config resolue, environnement et dependances dans le repertoire du run."""
    ensure_dir(run_dir)
    write_yaml(run_dir / "resolved_train_config.yaml", asdict(config))
    write_json(
        run_dir / "run_metadata.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "smoke_test": smoke,
            "device_resolved": device,
            "seed": config.seed,
            "deterministic": config.deterministic,
            "environment": describe_environment(),
            "pip_freeze": pip_freeze(),
        },
    )


def _publish_weights(run_dir: Path, destination: Path, *, prefix: str = "") -> dict[str, str]:
    """Copie ``best.pt`` / ``last.pt`` vers un emplacement stable.

    Args:
        run_dir: Repertoire du run Ultralytics.
        destination: Repertoire de publication (``artifacts/models``).
        prefix: Prefixe optionnel des fichiers publies (ex. ``"smoke_"``).

    Returns:
        ``{"best": chemin, "last": chemin}`` pour les poids reellement trouves.
    """
    ensure_dir(destination)
    published: dict[str, str] = {}
    for kind in ("best", "last"):
        source = run_dir / "weights" / f"{kind}.pt"
        if source.is_file():
            target = destination / f"{prefix}{kind}.pt"
            shutil.copy2(source, target)
            published[kind] = str(target)
            LOGGER.info("Poids publies : %s", target)
    return published


def _metrics_to_dict(metrics: Any, class_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Extrait les metriques exploitables d'un objet de resultats Ultralytics."""
    payload: dict[str, Any] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        payload["summary"] = {str(k): float(v) for k, v in results_dict.items() if _is_number(v)}
    box = getattr(metrics, "box", None)
    if box is not None:
        for attr in ("map", "map50", "map75", "mp", "mr"):
            value = getattr(box, attr, None)
            if value is not None and _is_number(value):
                payload[attr] = float(value)
        per_class = getattr(box, "maps", None)
        if per_class is not None and class_names:
            with contextlib.suppress(TypeError, IndexError):  # pragma: no cover
                payload["map50_95_per_class"] = {
                    class_names[i]: float(v) for i, v in enumerate(per_class) if i < len(class_names)
                }
    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        payload["speed_ms"] = {str(k): float(v) for k, v in speed.items() if _is_number(v)}
    return payload


def _is_number(value: Any) -> bool:
    """True si la valeur est convertible en float sans exception."""
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def train(
    config: TrainConfig,
    *,
    smoke: bool = False,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Lance un entrainement Ultralytics a partir d'une configuration resolue.

    Args:
        config: Hyperparametres d'entrainement.
        smoke: Applique les surcharges de test rapide.
        resume_from: Checkpoint ``last.pt`` a reprendre.

    Returns:
        Un dictionnaire recapitulant le run (chemins, metriques, device).

    Raises:
        TrainingError: Dataset absent, saturation memoire GPU, ou echec Ultralytics.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise TrainingError(
            "Ultralytics n'est pas installe. Executez : pip install ultralytics"
        ) from exc

    if smoke:
        for key, value in SMOKE_OVERRIDES.items():
            setattr(config, key, value)
        LOGGER.info("Mode SMOKE TEST : %s", SMOKE_OVERRIDES)

    data_path = _check_dataset(config)
    device = resolve_device(config.device)
    set_seed(config.seed, deterministic=config.deterministic)

    environment = describe_environment()
    LOGGER.info(
        "Environnement : torch=%s, ultralytics=%s, CUDA=%s (%s)",
        environment.get("torch"),
        environment.get("ultralytics"),
        environment.get("cuda_available"),
        environment.get("gpu_name", "CPU"),
    )
    if device == "cpu":
        LOGGER.warning(
            "Entrainement sur CPU : cela sera tres lent pour %d epoques. "
            "Verifiez l'installation CUDA si un GPU est disponible.",
            config.epochs,
        )

    weights_source = str(resume_from) if resume_from else config.model
    if resume_from:
        checkpoint = resolve_path(resume_from)
        if not checkpoint.is_file():
            raise TrainingError(
                f"Checkpoint de reprise introuvable : {checkpoint}\n"
                f"Verifiez le chemin, par exemple artifacts/runs/<nom>/weights/last.pt"
            )
        weights_source = str(checkpoint)
        config.resume = True
        LOGGER.info("Reprise de l'entrainement depuis %s", checkpoint)

    kwargs = config.to_ultralytics_kwargs()
    kwargs["data"] = str(data_path)
    kwargs["device"] = device
    kwargs["batch"] = _resolve_batch(config, device)

    LOGGER.info(
        "Demarrage : modele=%s, imgsz=%d, epochs=%d, batch=%s, device=%s",
        weights_source,
        config.imgsz,
        config.epochs,
        kwargs["batch"],
        device,
    )

    model = YOLO(weights_source)
    started = datetime.now(timezone.utc)

    try:
        results = model.train(**kwargs)
    except KeyboardInterrupt:
        LOGGER.warning(
            "Entrainement interrompu par l'utilisateur. Le dernier checkpoint reste "
            "disponible dans %s/<nom>/weights/last.pt — reprenez avec --resume.",
            resolve_path(config.project),
        )
        raise
    except (RuntimeError, MemoryError) as exc:
        message = str(exc).lower()
        if "out of memory" in message or "cuda error" in message:
            raise TrainingError(
                f"Memoire GPU insuffisante : {exc}\n\n"
                f"Pistes de resolution, par ordre d'efficacite :\n"
                f"  1. Reduire le batch      : --batch 16 (puis 8)\n"
                f"  2. Reduire la resolution : --imgsz 512\n"
                f"  3. Choisir un modele plus petit : --model yolo26n.pt\n"
                f"  4. Desactiver le cache   : cache: false dans configs/train.yaml\n"
                f"  5. Fermer les autres applications utilisant le GPU (nvidia-smi)"
            ) from exc
        raise TrainingError(f"Echec de l'entrainement Ultralytics : {exc}") from exc

    finished = datetime.now(timezone.utc)
    trainer_save_dir = getattr(getattr(model, "trainer", None), "save_dir", "")
    run_dir = Path(trainer_save_dir or resolve_path(config.project) / config.name)
    LOGGER.info("Entrainement termine en %s. Run : %s", finished - started, run_dir)

    _archive_run_metadata(run_dir, config, device, smoke)
    prefix = "smoke_" if smoke else ""
    published = _publish_weights(run_dir, project_root() / "artifacts" / "models", prefix=prefix)

    class_names = list(getattr(model, "names", {}).values()) if getattr(model, "names", None) else None
    summary = {
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 1),
        "smoke_test": smoke,
        "run_dir": str(run_dir),
        "device": device,
        "seed": config.seed,
        "model": weights_source,
        "data": str(data_path),
        "published_weights": published,
        "metrics": _metrics_to_dict(results, class_names),
    }
    write_json(run_dir / "training_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments de la commande d'entrainement."""
    parser = argparse.ArgumentParser(
        prog="python -m ppe_detection.train",
        description="Entraine un detecteur d'EPI (Ultralytics YOLO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Fichier YAML de configuration (defaut : configs/train.yaml).",
    )
    parser.add_argument("--data", default=None, help="Chemin du data.yaml du dataset normalise.")
    parser.add_argument("--model", default=None, help="Poids ou architecture de depart (ex. yolo26s.pt).")
    parser.add_argument("--imgsz", type=int, default=None, help="Taille d'image.")
    parser.add_argument("--epochs", type=int, default=None, help="Nombre d'epoques.")
    parser.add_argument("--batch", type=float, default=None, help="Taille de lot (-1 = automatique).")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda | 0 | 0,1")
    parser.add_argument("--workers", type=int, default=None, help="Workers du DataLoader.")
    parser.add_argument("--seed", type=int, default=None, help="Graine aleatoire.")
    parser.add_argument("--patience", type=int, default=None, help="Patience de l'early stopping.")
    parser.add_argument("--optimizer", default=None, help="auto | SGD | Adam | AdamW | NAdam | RMSProp")
    parser.add_argument("--lr0", type=float, default=None, help="Taux d'apprentissage initial.")
    parser.add_argument("--weight-decay", type=float, dest="weight_decay", default=None, help="Weight decay.")
    parser.add_argument("--fraction", type=float, default=None, help="Fraction du train set utilisee.")
    parser.add_argument("--cache", default=None, help="false | disk | ram")
    parser.add_argument("--name", default=None, help="Nom de l'experience.")
    parser.add_argument("--project", default=None, help="Repertoire des runs.")
    parser.add_argument(
        "--exist-ok",
        dest="exist_ok",
        action="store_true",
        default=None,
        help="Reutilise un repertoire de run existant.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        default=None,
        help="Desactive la precision mixte.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Test rapide du pipeline (2 epoques, 4 %% des donnees).",
    )
    parser.add_argument("--resume", default=None, help="Chemin d'un last.pt a reprendre.")
    parser.add_argument("--log-level", default="INFO", help="Niveau de log.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entree CLI. Retourne le code de sortie du processus."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=project_root() / "artifacts" / "logs" / "train.log")

    config_path = args.config or default_config_path("train.yaml")
    if not Path(config_path).is_file():
        LOGGER.warning("Configuration %s introuvable — valeurs par defaut du code utilisees.", config_path)
        config_path = None

    overrides = {
        key: value
        for key, value in vars(args).items()
        if key not in {"config", "smoke", "resume", "log_level"} and value is not None
    }

    try:
        config = load_train_config(config_path, **overrides)
        summary = train(config, smoke=args.smoke, resume_from=args.resume)
    except KeyboardInterrupt:
        return 130
    except (TrainingError, ConfigError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        return 2

    metrics = summary.get("metrics", {})
    LOGGER.info(
        "Resume : mAP50=%.4f mAP50-95=%.4f precision=%.4f rappel=%.4f (duree %.1f s)",
        metrics.get("map50", float("nan")),
        metrics.get("map", float("nan")),
        metrics.get("mp", float("nan")),
        metrics.get("mr", float("nan")),
        summary["duration_seconds"],
    )
    if not summary["published_weights"]:
        LOGGER.error("Aucun poids n'a ete produit — verifiez le journal Ultralytics ci-dessus.")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
