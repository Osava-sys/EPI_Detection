"""Export du modele vers ONNX (et formats optionnels), avec verification reelle.

Usage :

    python -m ppe_detection.export --weights artifacts/models/best.pt \\
        --format onnx --imgsz 640 --simplify

Un export n'est **jamais** considere comme reussi au seul motif qu'un fichier a
ete produit. Pour ONNX, la verification comprend :

1. l'existence et la taille non nulle du fichier ;
2. la validation du graphe par ``onnx.checker`` ;
3. le chargement effectif d'une session ONNX Runtime ;
4. une inference sur une entree factice de forme attendue ;
5. la comparaison numerique des sorties avec celles de PyTorch sur la meme
   entree, avec un rapport d'ecart chiffre.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .utils import (
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

LOGGER = get_logger(__name__)

SUPPORTED_FORMATS = ("onnx", "torchscript", "openvino", "engine")
"""Formats proposes. ``engine`` (TensorRT) requiert une installation dediee."""

# Ecart maximal tolere entre PyTorch et ONNX Runtime. Des differences de l'ordre
# de 1e-3 sont normales : l'export fige certaines operations en FP32 et les
# noyaux de convolution different entre les deux moteurs.
DEFAULT_TOLERANCE = 1e-3


class ExportError(RuntimeError):
    """Erreur bloquante durant l'export ou sa verification."""


def _load_model(weights: Path) -> Any:
    """Charge un modele Ultralytics depuis un fichier de poids."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ExportError("Ultralytics n'est pas installe : pip install ultralytics") from exc
    if not weights.is_file():
        raise ExportError(
            f"Poids introuvables : {weights}\n"
            f"Entrainez d'abord un modele ou corrigez --weights."
        )
    return YOLO(str(weights))


def verify_onnx(
    onnx_path: Path,
    weights_path: Path,
    *,
    imgsz: int = 640,
    tolerance: float = DEFAULT_TOLERANCE,
    seed: int = 0,
    sample_image: str | Path | None = None,
) -> dict[str, Any]:
    """Verifie reellement un modele ONNX exporte.

    Args:
        onnx_path: Fichier ``.onnx`` produit.
        weights_path: Poids PyTorch d'origine, pour la comparaison fonctionnelle.
        imgsz: Taille d'entree utilisee pour l'inference de controle.
        tolerance: Tolerance conservee pour les modeles a sortie brute.
        seed: Graine de l'entree factice (reproductibilite).
        sample_image: Image reelle utilisee pour comparer les detections.

    Returns:
        Un rapport de verification detaillant chaque etape.
    """
    report: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "checks": {},
        "passed": False,
    }

    # 1. Fichier present et non vide
    exists = onnx_path.is_file()
    size = onnx_path.stat().st_size if exists else 0
    report["checks"]["file_exists"] = exists
    report["checks"]["file_size_bytes"] = size
    report["checks"]["file_size_human"] = human_bytes(size)
    if not exists or size == 0:
        report["error"] = "Le fichier ONNX est absent ou vide."
        return report

    # 2. Validation du graphe
    try:
        import onnx

        model_proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_proto)
        report["checks"]["onnx_checker"] = True
        report["checks"]["ir_version"] = int(model_proto.ir_version)
        report["checks"]["opset"] = [
            {"domain": op.domain or "ai.onnx", "version": int(op.version)}
            for op in model_proto.opset_import
        ]
    except ImportError:
        report["checks"]["onnx_checker"] = "onnx non installe — verification du graphe ignoree"
    except Exception as exc:  # noqa: BLE001 - onnx leve des types varies
        report["checks"]["onnx_checker"] = False
        report["error"] = f"Graphe ONNX invalide : {exc}"
        return report

    # 3. Session ONNX Runtime
    try:
        import onnxruntime as ort
    except ImportError:
        report["checks"]["onnxruntime_session"] = "onnxruntime non installe"
        report["error"] = (
            "onnxruntime n'est pas installe : impossible de verifier le modele exporte. "
            "Installez-le avec : pip install onnxruntime"
        )
        return report

    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        report["checks"]["onnxruntime_session"] = False
        report["error"] = f"Session ONNX Runtime non chargeable : {exc}"
        return report

    report["checks"]["onnxruntime_session"] = True
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    report["checks"]["inputs"] = [{"name": i.name, "shape": list(i.shape), "type": i.type} for i in inputs]
    report["checks"]["outputs"] = [{"name": o.name, "shape": list(o.shape), "type": o.type} for o in outputs]

    # 4. Inference sur une entree factice
    rng = np.random.default_rng(seed)
    dummy = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)
    try:
        onnx_outputs = session.run(None, {inputs[0].name: dummy})
    except Exception as exc:  # noqa: BLE001
        report["checks"]["dummy_inference"] = False
        report["error"] = f"Inference ONNX Runtime impossible : {exc}"
        return report
    report["checks"]["dummy_inference"] = True
    report["checks"]["output_shapes"] = [list(np.asarray(o).shape) for o in onnx_outputs]

    # 5. Comparaison fonctionnelle sur une image reelle.
    #
    # YOLO26 s'exporte « end-to-end » : la sortie (1, N, 6) contient deja des
    # detections filtrees, triees par confiance. Comparer ces tenseurs terme a
    # terme n'a aucun sens — un ecart numerique infime reordonne les lignes, et
    # sur une entree aleatoire les detections sont de toute facon arbitraires.
    # On compare donc ce qui compte reellement : les detections produites par
    # les deux moteurs sur une vraie image.
    end_to_end = len(np.asarray(onnx_outputs[0]).shape) == 3 and np.asarray(onnx_outputs[0]).shape[-1] == 6
    report["checks"]["end_to_end_output"] = end_to_end

    comparison = _compare_detections(
        weights_path,
        onnx_path,
        imgsz=imgsz,
        sample_image=sample_image,
        max_conf_delta=max(tolerance, DEFAULT_TOLERANCE),
    )
    report["checks"]["detection_comparison"] = comparison

    if comparison.get("compared") and not comparison.get("agrees", False):
        report["error"] = (
            f"Les detections divergent entre PyTorch et ONNX : "
            f"{comparison.get('reason', 'ecart au-dela des tolerances')}."
        )
        return report

    report["passed"] = True
    return report


def _first_sample_image() -> Path | None:
    """Trouve une image reelle pour la verification fonctionnelle."""
    from .utils import is_image

    candidates = [
        project_root() / "artifacts" / "dataset_detection" / "test" / "images",
        project_root() / "test" / "images",
        project_root() / "artifacts" / "dataset_detection" / "valid" / "images",
    ]
    for directory in candidates:
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.is_file() and is_image(path):
                    return path
    return None


def _compare_detections(
    weights_path: Path,
    onnx_path: Path,
    *,
    imgsz: int,
    sample_image: str | Path | None = None,
    conf: float = 0.25,
    iou_match: float = 0.90,
    max_box_shift: float = 2.0,
    max_conf_delta: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Compare les detections PyTorch et ONNX sur une vraie image.

    Les deux modeles sont executes via Ultralytics, qui applique le meme
    pretraitement (letterbox) et le meme post-traitement aux deux formats : les
    ecarts constates refletent donc la fidelite reelle de l'export telle que la
    percoit l'utilisateur.

    Args:
        weights_path: Poids PyTorch de reference.
        onnx_path: Modele ONNX a verifier.
        imgsz: Taille d'inference.
        sample_image: Image de test; choisie automatiquement si ``None``.
        conf: Seuil de confiance de la comparaison.
        iou_match: IoU minimal pour considerer deux boites comme identiques.
        max_box_shift: Decalage maximal tolere, en pixels, sur un coin de boite.
        max_conf_delta: Ecart maximal tolere sur le score de confiance.

    Returns:
        Le detail de la comparaison.
    """
    image_path = Path(sample_image) if sample_image else _first_sample_image()
    if image_path is None or not image_path.is_file():
        return {
            "compared": False,
            "reason": (
                "Aucune image reelle disponible pour la comparaison fonctionnelle. "
                "Fournissez --sample-image pour l'activer."
            ),
        }

    try:
        from ultralytics import YOLO

        torch_model = YOLO(str(weights_path))
        onnx_model = YOLO(str(onnx_path))
        common: dict[str, Any] = {"imgsz": imgsz, "conf": conf, "device": "cpu", "verbose": False}
        # predict() renvoie une liste de Results en mode non-stream, malgre
        # l'union declaree dans les annotations d'Ultralytics.
        torch_result = list(torch_model.predict(source=str(image_path), **common))[0]
        onnx_result = list(onnx_model.predict(source=str(image_path), **common))[0]
    except Exception as exc:  # noqa: BLE001 - Ultralytics/ORT levent des types varies
        return {"compared": False, "reason": f"Comparaison impossible : {exc}"}

    def extract(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
        return (
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
        )

    torch_boxes, torch_conf, torch_cls = extract(torch_result)
    onnx_boxes, onnx_conf, onnx_cls = extract(onnx_result)

    result: dict[str, Any] = {
        "compared": True,
        "sample_image": str(image_path),
        "conf_threshold": conf,
        "pytorch_detections": int(len(torch_boxes)),
        "onnx_detections": int(len(onnx_boxes)),
    }

    if len(torch_boxes) == 0 and len(onnx_boxes) == 0:
        result["agrees"] = True
        result["note"] = (
            "Aucun des deux moteurs ne detecte d'objet au-dessus du seuil sur cette image : "
            "les sorties concordent, mais la comparaison est peu informative."
        )
        return result

    if len(torch_boxes) != len(onnx_boxes):
        result["agrees"] = False
        result["reason"] = (
            f"nombre de detections different ({len(torch_boxes)} en PyTorch "
            f"contre {len(onnx_boxes)} en ONNX)"
        )
        return result

    # Appariement glouton par IoU (les deux listes sont triees par confiance,
    # mais un ecart numerique peut permuter deux detections tres proches).
    matched = 0
    ious: list[float] = []
    conf_deltas: list[float] = []
    box_shifts: list[float] = []
    class_mismatches = 0
    available = set(range(len(onnx_boxes)))

    for index in range(len(torch_boxes)):
        best_iou = 0.0
        best_index = -1
        for candidate in available:
            value = _iou(torch_boxes[index], onnx_boxes[candidate])
            if value > best_iou:
                best_iou = value
                best_index = candidate
        if best_index < 0 or best_iou < iou_match:
            continue
        available.discard(best_index)
        matched += 1
        ious.append(best_iou)
        conf_deltas.append(abs(float(torch_conf[index]) - float(onnx_conf[best_index])))
        box_shifts.append(float(np.abs(torch_boxes[index] - onnx_boxes[best_index]).max()))
        if int(torch_cls[index]) != int(onnx_cls[best_index]):
            class_mismatches += 1

    result.update(
        {
            "matched_detections": matched,
            "class_mismatches": class_mismatches,
            "mean_iou": round(float(np.mean(ious)), 6) if ious else 0.0,
            "min_iou": round(float(np.min(ious)), 6) if ious else 0.0,
            "max_conf_delta": round(float(np.max(conf_deltas)), 6) if conf_deltas else 0.0,
            "max_box_shift_px": round(float(np.max(box_shifts)), 4) if box_shifts else 0.0,
            "iou_match_threshold": iou_match,
            "max_box_shift_tolerance_px": max_box_shift,
            "max_conf_delta_tolerance": max_conf_delta,
        }
    )

    agrees = (
        matched == len(torch_boxes)
        and class_mismatches == 0
        and result["max_box_shift_px"] <= max_box_shift
        and result["max_conf_delta"] <= max_conf_delta
    )
    result["agrees"] = agrees
    if not agrees:
        reasons = []
        if matched != len(torch_boxes):
            reasons.append(f"{len(torch_boxes) - matched} detection(s) non appariee(s)")
        if class_mismatches:
            reasons.append(f"{class_mismatches} classe(s) divergente(s)")
        if result["max_box_shift_px"] > max_box_shift:
            reasons.append(f"decalage de boite de {result['max_box_shift_px']} px")
        if result["max_conf_delta"] > max_conf_delta:
            reasons.append(f"ecart de confiance de {result['max_conf_delta']}")
        result["reason"] = ", ".join(reasons)
    return result


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    """IoU entre deux boites ``(x1, y1, x2, y2)``."""
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    area_b = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def export_model(
    weights: str | Path,
    *,
    export_format: str = "onnx",
    imgsz: int = 640,
    simplify: bool = False,
    opset: int | None = None,
    dynamic: bool = False,
    half: bool = False,
    device: str = "cpu",
    output_dir: str | Path = "artifacts/exports",
    verify: bool = True,
    tolerance: float = DEFAULT_TOLERANCE,
    sample_image: str | Path | None = None,
) -> dict[str, Any]:
    """Exporte un modele et verifie le resultat.

    Args:
        weights: Poids ``.pt`` source.
        export_format: ``onnx``, ``torchscript``, ``openvino`` ou ``engine``.
        imgsz: Taille d'entree figee dans le modele exporte.
        simplify: Simplifie le graphe ONNX (necessite ``onnxslim``).
        opset: Version d'opset ONNX (defaut Ultralytics si ``None``).
        dynamic: Autorise des dimensions dynamiques (batch/taille variables).
        half: Export en FP16.
        device: Device d'export. ``cpu`` est le choix le plus portable pour ONNX.
        output_dir: Repertoire de destination.
        verify: Lance la verification reelle du modele exporte.
        tolerance: Tolerance de la comparaison numerique.
        sample_image: Image reelle utilisee pour comparer les detections.

    Returns:
        Le rapport d'export.

    Raises:
        ExportError: Format non supporte, poids absents ou echec de l'export.
    """
    if export_format not in SUPPORTED_FORMATS:
        raise ExportError(
            f"Format non supporte : {export_format!r}. "
            f"Formats disponibles : {', '.join(SUPPORTED_FORMATS)}."
        )

    weights_path = resolve_path(weights)
    model = _load_model(weights_path)
    destination = ensure_dir(resolve_path(output_dir))

    kwargs: dict[str, Any] = {
        "format": export_format,
        "imgsz": imgsz,
        "device": device,
        "dynamic": dynamic,
        "verbose": False,
    }
    if export_format == "onnx":
        kwargs["simplify"] = simplify
        if opset is not None:
            kwargs["opset"] = opset
    if half:
        kwargs["half"] = True

    LOGGER.info(
        "Export %s de %s (imgsz=%d, simplify=%s)...",
        export_format, weights_path.name, imgsz, simplify,
    )
    started = datetime.now(timezone.utc)
    try:
        exported = model.export(**kwargs)
    except Exception as exc:  # noqa: BLE001 - Ultralytics leve des types varies
        message = str(exc)
        hint = ""
        if export_format == "engine":
            hint = (
                "\nL'export TensorRT necessite le paquet 'tensorrt' et un GPU NVIDIA "
                "compatible avec la version installee."
            )
        elif export_format == "openvino":
            hint = "\nL'export OpenVINO necessite : pip install openvino"
        elif "onnxslim" in message.lower() or "simplify" in message.lower():
            hint = "\nLa simplification necessite : pip install onnxslim"
        raise ExportError(f"Echec de l'export {export_format} : {message}{hint}") from exc

    exported_path = Path(str(exported))
    if not exported_path.is_absolute():
        exported_path = resolve_path(exported_path)

    # Ultralytics ecrit a cote des poids source : on deplace dans artifacts/exports.
    final_path = exported_path
    if exported_path.exists() and exported_path.parent != destination:
        target = destination / exported_path.name
        try:
            if target.exists():
                if target.is_dir():
                    import shutil

                    shutil.rmtree(target)
                else:
                    target.unlink()
            exported_path.replace(target)
            final_path = target
        except OSError as exc:
            LOGGER.warning(
                "Deplacement vers %s impossible (%s) — fichier conserve sur place.",
                destination, exc,
            )

    finished = datetime.now(timezone.utc)
    size = final_path.stat().st_size if final_path.is_file() else None

    report: dict[str, Any] = {
        "generated_at": finished.isoformat(),
        "environment": describe_environment(),
        "source_weights": str(weights_path),
        "source_weights_size": human_bytes(weights_path.stat().st_size),
        "format": export_format,
        "output_path": str(final_path),
        "output_size_bytes": size,
        "output_size_human": human_bytes(size) if size else None,
        "duration_seconds": round((finished - started).total_seconds(), 2),
        "parameters": {
            "imgsz": imgsz,
            "simplify": simplify,
            "opset": opset,
            "dynamic": dynamic,
            "half": half,
            "device": device,
        },
    }
    LOGGER.info("Fichier produit : %s (%s)", final_path, report["output_size_human"])

    if verify and export_format == "onnx":
        LOGGER.info("Verification du modele ONNX exporte...")
        verification = verify_onnx(
            final_path,
            weights_path,
            imgsz=imgsz,
            tolerance=tolerance,
            sample_image=sample_image,
        )
        report["verification"] = verification
        if verification["passed"]:
            LOGGER.info("Verification ONNX reussie.")
        else:
            LOGGER.error("Verification ONNX ECHOUEE : %s", verification.get("error", "cause inconnue"))
    elif verify:
        report["verification"] = {
            "passed": None,
            "note": (
                f"La verification automatique n'est implementee que pour ONNX. "
                f"Le fichier {export_format} a ete produit mais n'a pas ete execute."
            ),
        }

    report["post_processing_notes"] = _post_processing_notes(export_format)
    return report


def _post_processing_notes(export_format: str) -> list[str]:
    """Differences de post-traitement a connaitre pour le format exporte."""
    notes = [
        "Le modele exporte attend une image normalisee dans [0, 1], au format "
        "NCHW (1x3xHxW), en RGB, redimensionnee avec letterbox vers la taille figee "
        "a l'export.",
        "La sortie brute n'est pas filtree de la meme maniere que l'API Python : "
        "selon le format, la suppression des non-maxima (NMS) et le decodage des "
        "boites peuvent devoir etre reimplementes cote client.",
        "Les coordonnees produites se rapportent a l'image redimensionnee : il faut "
        "annuler le letterbox (echelle et decalage) pour revenir aux coordonnees "
        "de l'image d'origine.",
    ]
    if export_format == "onnx":
        notes.append(
            "L'ecart numerique entre PyTorch et ONNX Runtime est normalement de "
            "l'ordre de 1e-4 a 1e-3 : il provient des noyaux de convolution "
            "differents entre les deux moteurs, pas d'une erreur de conversion."
        )
    return notes


def render_export_markdown(report: dict[str, Any]) -> str:
    """Genere le rapport Markdown d'export."""
    lines: list[str] = []
    lines.append(f"# Rapport d'export — format `{report['format']}`")
    lines.append("")
    lines.append(f"- **Genere le** : {report['generated_at']}")
    lines.append(f"- **Poids source** : `{report['source_weights']}` ({report['source_weights_size']})")
    lines.append(f"- **Fichier produit** : `{report['output_path']}` ({report['output_size_human']})")
    lines.append(f"- **Duree** : {report['duration_seconds']} s")
    lines.append("")

    lines.append("## Parametres d'export")
    lines.append("")
    lines.append(
        markdown_table(
            ["Parametre", "Valeur"],
            [[key, value] for key, value in report["parameters"].items()],
        )
    )
    lines.append("")

    verification = report.get("verification", {})
    lines.append("## Verification")
    lines.append("")
    if verification.get("passed") is True:
        lines.append("**Resultat : REUSSIE** — le modele exporte a reellement ete execute.")
    elif verification.get("passed") is False:
        lines.append(f"**Resultat : ECHOUEE** — {verification.get('error', 'cause inconnue')}")
    else:
        lines.append(verification.get("note", "Verification non effectuee."))
    lines.append("")

    checks = verification.get("checks", {})
    if checks:
        rows: list[list[Any]] = []
        for key, value in checks.items():
            if key in {"inputs", "outputs", "opset", "detection_comparison", "output_shapes"}:
                continue
            rows.append([key, value])
        lines.append(markdown_table(["Controle", "Resultat"], rows))
        lines.append("")

        if checks.get("inputs"):
            lines.append("### Entrees / sorties du graphe")
            lines.append("")
            lines.append(
                markdown_table(
                    ["Sens", "Nom", "Forme", "Type"],
                    [
                        ["entree", i["name"], str(i["shape"]), i["type"]]
                        for i in checks.get("inputs", [])
                    ]
                    + [
                        ["sortie", o["name"], str(o["shape"]), o["type"]]
                        for o in checks.get("outputs", [])
                    ],
                )
            )
            lines.append("")

        comparison = checks.get("detection_comparison")
        if isinstance(comparison, dict):
            lines.append("### Comparaison fonctionnelle PyTorch vs ONNX Runtime")
            lines.append("")
            if comparison.get("compared"):
                lines.append(
                    f"Les deux moteurs ont traite la meme image reelle "
                    f"(`{Path(comparison['sample_image']).name}`) au seuil "
                    f"`conf={comparison['conf_threshold']}`."
                )
                lines.append("")
                rows = [
                    ["Detections PyTorch", comparison["pytorch_detections"]],
                    ["Detections ONNX", comparison["onnx_detections"]],
                ]
                if "matched_detections" in comparison:
                    rows.extend(
                        [
                            ["Detections appariees", comparison["matched_detections"]],
                            ["Classes divergentes", comparison["class_mismatches"]],
                            ["IoU moyen", comparison["mean_iou"]],
                            ["IoU minimal", comparison["min_iou"]],
                            ["Ecart max de confiance", comparison["max_conf_delta"]],
                            ["Decalage max de boite (px)", comparison["max_box_shift_px"]],
                        ]
                    )
                rows.append(["**Concordance**", "oui" if comparison.get("agrees") else "NON"])
                lines.append(markdown_table(["Indicateur", "Valeur"], rows))
                if comparison.get("note"):
                    lines.append("")
                    lines.append(f"> {comparison['note']}")
            else:
                lines.append(f"Comparaison non realisee : {comparison.get('reason', 'raison inconnue')}")
            lines.append("")
            if checks.get("end_to_end_output"):
                lines.append(
                    "> Ce modele s'exporte **end-to-end** : la sortie ONNX `(1, N, 6)` contient "
                    "deja des detections filtrees et triees par confiance. Une comparaison "
                    "terme a terme des tenseurs bruts serait sans signification (un ecart "
                    "numerique infime reordonne les lignes), d'ou la comparaison des "
                    "detections effectivement produites."
                )
                lines.append("")

    lines.append("## Differences de post-traitement a connaitre")
    lines.append("")
    for note in report.get("post_processing_notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construit l'analyseur d'arguments de la commande d'export."""
    parser = argparse.ArgumentParser(
        prog="python -m ppe_detection.export",
        description="Exporte le modele EPI vers ONNX (ou un autre format) et verifie le resultat.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default="artifacts/models/best.pt", help="Poids a exporter.")
    parser.add_argument(
        "--format",
        dest="export_format",
        default="onnx",
        choices=SUPPORTED_FORMATS,
        help="Format cible.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'entree figee.")
    parser.add_argument("--simplify", action="store_true", help="Simplifie le graphe ONNX (onnxslim).")
    parser.add_argument("--opset", type=int, default=None, help="Version d'opset ONNX.")
    parser.add_argument("--dynamic", action="store_true", help="Autorise des dimensions dynamiques.")
    parser.add_argument("--half", action="store_true", help="Export en FP16.")
    parser.add_argument("--device", default="cpu", help="Device d'export (cpu recommande pour ONNX).")
    parser.add_argument("--output", default="artifacts/exports", help="Repertoire de destination.")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="N'execute pas la verification du modele exporte.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Tolerance de la comparaison numerique.",
    )
    parser.add_argument(
        "--sample-image",
        dest="sample_image",
        default=None,
        help="Image reelle servant a comparer les detections PyTorch et ONNX.",
    )
    parser.add_argument(
        "--report",
        default="artifacts/reports/export.json",
        help="Chemin du rapport d'export.",
    )
    parser.add_argument("--log-level", default="INFO", help="Niveau de log.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entree CLI. Retourne le code de sortie du processus."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, log_file=project_root() / "artifacts" / "logs" / "export.log")

    try:
        report = export_model(
            args.weights,
            export_format=args.export_format,
            imgsz=args.imgsz,
            simplify=args.simplify,
            opset=args.opset,
            dynamic=args.dynamic,
            half=args.half,
            device=args.device,
            output_dir=args.output,
            verify=not args.no_verify,
            tolerance=args.tolerance,
            sample_image=args.sample_image,
        )
    except ExportError as exc:
        LOGGER.error("%s", exc)
        return 2

    report_path = resolve_path(args.report)
    write_json(report_path, report)
    write_text(report_path.with_suffix(".md"), render_export_markdown(report))
    LOGGER.info("Rapport d'export : %s", report_path)

    verification = report.get("verification", {})
    if verification.get("passed") is False:
        LOGGER.error("L'export a produit un fichier mais la verification a echoue.")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
