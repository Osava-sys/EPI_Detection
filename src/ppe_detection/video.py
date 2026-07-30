"""Inference sur flux video : fichier, webcam ou flux reseau.

Le modele est charge une seule fois par l'appelant (:class:`~ppe_detection.predict.PPEDetector`)
puis reutilise pour toutes les frames. La camera et le ``VideoWriter`` sont
liberes dans un bloc ``finally`` afin que le peripherique reste utilisable meme
en cas d'interruption ou d'exception.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from .compliance import ComplianceTracker, summarise_compliance
from .utils import ensure_dir, get_logger, write_json

if TYPE_CHECKING:  # pragma: no cover
    from .predict import PPEDetector

LOGGER = get_logger(__name__)

QUIT_KEYS = {ord("q"), 27}  # 'q' ou Echap
"""Touches interrompant l'affichage temps reel."""

DEFAULT_FPS = 25.0
"""FPS de repli lorsque la source n'expose pas cette propriete (webcams)."""

# Ordre de preference des codecs de sortie.
#
# 'mp4v' produit un flux MPEG-4 Part 2 (FOURCC 'FMP4') qu'aucun navigateur ne
# sait decoder nativement : la video annotee est alors illisible dans Streamlit
# ou dans une balise HTML <video>. 'avc1' produit du H.264, lu partout.
# OpenCV peut afficher une erreur libopenh264 puis basculer sur un autre
# encodeur H.264 : le fichier produit reste valide, l'erreur est benigne.
VIDEO_CODECS: tuple[str, ...] = ("avc1", "mp4v")


BROWSER_PLAYABLE_CODECS: frozenset[str] = frozenset({"h264", "avc1", "vp09", "av01"})
"""Codecs qu'une balise HTML ``<video>`` sait decoder nativement."""


def probe_video_codec(path: Path) -> str:
    """Retourne le FOURCC reellement stocke dans un fichier video.

    Indispensable car OpenCV **substitue silencieusement** un codec lorsque
    celui demande est indisponible : le writer s'ouvre sans erreur et le
    fichier produit n'a pas le format attendu. Seule la relecture dit la verite.

    Args:
        path: Fichier a inspecter.

    Returns:
        Le FOURCC en minuscules (``"h264"``, ``"fmp4"``...), ou ``""`` si le
        fichier est illisible.
    """
    if not path.is_file():
        return ""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return ""
    raw = int(capture.get(cv2.CAP_PROP_FOURCC))
    capture.release()
    return "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ").lower()


def create_video_writer(
    path: Path,
    fps: float,
    size: tuple[int, int],
    *,
    codecs: Sequence[str] = VIDEO_CODECS,
) -> tuple[cv2.VideoWriter | None, str]:
    """Cree un ``VideoWriter`` en privilegiant un codec lisible par navigateur.

    Attention : ``isOpened()`` ne garantit pas que le codec demande a ete
    retenu — OpenCV peut en substituer un autre sans le signaler. Le codec
    reellement ecrit ne se connait qu'apres fermeture du fichier, via
    :func:`probe_video_codec`.

    Args:
        path: Fichier de sortie.
        fps: Images par seconde.
        size: ``(largeur, hauteur)``.
        codecs: Codecs a essayer, du plus souhaitable au repli.

    Returns:
        ``(writer, codec_demande)``; ``(None, "")`` si aucun writer ne s'ouvre.
    """
    ensure_dir(path.parent)
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
            writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        except (cv2.error, TypeError, ValueError) as exc:  # pragma: no cover
            LOGGER.debug("Codec %s indisponible : %s", codec, exc)
            continue
        if writer.isOpened():
            if codec != codecs[0]:
                LOGGER.warning(
                    "Codec %s indisponible — repli sur %s. La video produite risque de ne "
                    "pas etre lisible directement dans un navigateur.",
                    codecs[0],
                    codec,
                )
            return writer, codec
        writer.release()
    return None, ""


class VideoSourceError(RuntimeError):
    """La source video n'a pas pu etre ouverte ou lue."""


def open_capture(source: str) -> tuple[cv2.VideoCapture, str]:
    """Ouvre une source video et renvoie la capture avec son libelle.

    Args:
        source: Chemin de fichier, index de webcam (``"0"``) ou URL de flux.

    Returns:
        ``(capture, libelle)``.

    Raises:
        VideoSourceError: Si la source ne peut pas etre ouverte, avec un message
            distinguant webcam, flux reseau et fichier.
    """
    text = str(source).strip()
    if text.isdigit():
        index = int(text)
        # CAP_DSHOW evite un demarrage tres lent des webcams sous Windows.
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
        label = f"webcam:{index}"
        if not capture.isOpened():
            raise VideoSourceError(
                f"Impossible d'ouvrir la webcam d'index {index}.\n"
                f"Verifiez qu'aucune autre application n'utilise la camera, "
                f"que le peripherique existe, et que l'acces a la camera est "
                f"autorise dans les parametres de confidentialite de Windows."
            )
        return capture, label

    capture = cv2.VideoCapture(text)
    label = text
    if not capture.isOpened():
        capture.release()
        if text.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "tcp://", "udp://")):
            raise VideoSourceError(
                f"Impossible d'ouvrir le flux : {text}\n"
                f"Verifiez l'URL, les identifiants, l'accessibilite reseau, "
                f"et que le codec du flux est supporte par votre build OpenCV."
            )
        raise VideoSourceError(
            f"Impossible d'ouvrir la video : {text}\n"
            f"Verifiez que le fichier existe et que son codec est supporte par OpenCV."
        )
    return capture, label


def run_video_inference(
    detector: PPEDetector,
    source: str,
    *,
    output_dir: Path,
    save_video: bool = True,
    show: bool = False,
    frame_skip: int = 0,
    max_frames: int = 0,
    save_json: bool = False,
    output_name: str | None = None,
    track: bool = False,
) -> dict[str, Any]:
    """Traite une source video frame par frame.

    L'ordre des frames est preserve. Lorsque ``frame_skip`` est actif, les
    frames non inferees sont tout de meme ecrites dans la video de sortie en
    reutilisant les dernieres detections, ce qui conserve la duree et la
    fluidite du fichier produit.

    Avec ``track=True``, les personnes recoivent un identifiant persistant et
    les verdicts de conformite sont lisses dans le temps : une alerte n'est
    levee qu'apres plusieurs frames concordantes, et une seule fois par
    personne. C'est ce qui rend le systeme exploitable en surveillance continue.

    Args:
        detector: Detecteur deja initialise (modele charge une seule fois).
        source: Fichier video, index de webcam ou URL de flux.
        output_dir: Repertoire de sortie.
        save_video: Ecrit une video annotee.
        show: Affiche une fenetre temps reel (arret par 'q' ou Echap).
        frame_skip: Nombre de frames sautees entre deux inferences.
        max_frames: Limite de frames traitees (0 = illimite).
        save_json: Ecrit le detail des detections par frame.
        output_name: Nom du fichier video de sortie.
        track: Active le suivi d'objets et le lissage temporel des verdicts.

    Returns:
        Un resume de l'execution (frames, FPS, detections, chemins produits).

    Raises:
        VideoSourceError: Si la source est inaccessible.
    """
    capture, label = open_capture(source)
    writer: cv2.VideoWriter | None = None
    window_name = "PPE Detection — 'q' ou Echap pour quitter"
    window_created = False

    frames_read = 0
    frames_processed = 0
    total_detections = 0
    per_frame: list[dict[str, Any]] = []
    last_detections: list[dict[str, Any]] = []
    last_compliance: list[dict[str, Any]] = []
    started = time.perf_counter()
    interrupted = False
    new_alerts: list[dict[str, Any]] = []
    output_video_path: Path | None = None
    output_codec = ""

    compliance_config = getattr(detector, "compliance_config", None)
    tracker: ComplianceTracker | None = None
    if track and compliance_config is not None and compliance_config.enabled:
        tracker = ComplianceTracker(compliance_config)

    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        fps = float(source_fps) if source_fps and source_fps > 0 else DEFAULT_FPS
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        LOGGER.info(
            "Source ouverte (%s) : %dx%d @ %.2f FPS%s",
            label,
            width,
            height,
            fps,
            f", {total} frames" if total > 0 else "",
        )

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames_read += 1

            should_infer = frame_skip <= 0 or (frames_read - 1) % (frame_skip + 1) == 0
            if should_infer:
                prediction = detector.predict_array(
                    frame, source_name=f"{label}#{frames_read}", track=track
                )
                if tracker is not None:
                    prediction.compliance = tracker.update(prediction.compliance)
                    for person in prediction.compliance:
                        if person.get("is_new_alert"):
                            alert = {
                                "frame_index": frames_read,
                                "track_id": person.get("track_id"),
                                "missing_ppe": person.get("missing_ppe", []),
                                "agreement": person.get("agreement"),
                            }
                            new_alerts.append(alert)
                            LOGGER.warning(
                                "ALERTE frame %d — personne #%s non conforme : %s manquant(s)",
                                frames_read,
                                person.get("track_id"),
                                ", ".join(person.get("missing_ppe") or []) or "?",
                            )
                last_detections = prediction.detections
                last_compliance = prediction.compliance
                frames_processed += 1
                total_detections += len(prediction.detections)
                if save_json:
                    entry = prediction.to_dict()
                    entry["frame_index"] = frames_read
                    per_frame.append(entry)

            annotated = frame
            if save_video or show:
                from .predict import ImagePrediction

                snapshot = ImagePrediction(
                    source=f"{label}#{frames_read}",
                    width=frame.shape[1],
                    height=frame.shape[0],
                    detections=last_detections,
                    compliance=last_compliance,
                )
                annotated = detector.annotate(frame, snapshot)

                elapsed = max(time.perf_counter() - started, 1e-6)
                cv2.putText(
                    annotated,
                    f"FPS {frames_read / elapsed:5.1f} | frame {frames_read} | {len(last_detections)} obj",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )

            if save_video:
                if writer is None:
                    default_name = f"{Path(str(source)).stem or 'stream'}_annotated.mp4"
                    target = output_dir / (output_name or default_name)
                    writer, codec_used = create_video_writer(
                        target, fps, (annotated.shape[1], annotated.shape[0])
                    )
                    if writer is None:
                        LOGGER.error(
                            "Aucun codec video disponible (%s) — l'enregistrement est desactive. "
                            "Les detections restent exportables en JSON.",
                            ", ".join(VIDEO_CODECS),
                        )
                        save_video = False
                    else:
                        output_video_path = target
                        output_codec = codec_used
                        LOGGER.info(
                            "Ecriture de la video annotee : %s (codec %s)", target, codec_used
                        )
                if writer is not None:
                    writer.write(annotated)

            if show:
                cv2.imshow(window_name, annotated)
                window_created = True
                if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
                    LOGGER.info("Arret demande par l'utilisateur.")
                    interrupted = True
                    break

            if max_frames and frames_read >= max_frames:
                LOGGER.info("Limite de %d frames atteinte.", max_frames)
                break
            if frames_read % 100 == 0:
                LOGGER.info("… %d frames lues (%d inferees)", frames_read, frames_processed)

    except KeyboardInterrupt:
        LOGGER.warning("Interruption clavier — fermeture propre de la source.")
        interrupted = True
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if window_created:
            cv2.destroyWindow(window_name)
            # Sous Windows, quelques iterations sont necessaires pour que la
            # fenetre se ferme reellement.
            for _ in range(4):
                cv2.waitKey(1)

    duration = max(time.perf_counter() - started, 1e-6)
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "label": label,
        "frames_read": frames_read,
        "frames_processed": frames_processed,
        "frames_skipped": frames_read - frames_processed,
        "total_detections": total_detections,
        "duration_seconds": round(duration, 2),
        "average_fps": round(frames_read / duration, 2),
        "inference_fps": round(frames_processed / duration, 2),
        "interrupted": interrupted,
        "tracking_enabled": track,
        "output_dir": str(output_dir),
        "output_video": str(output_video_path) if output_video_path else None,
        "requested_codec": output_codec,
    }

    # Le codec effectivement ecrit ne se lit qu'apres fermeture du fichier :
    # c'est lui, et non celui demande, qui determine la lisibilite navigateur.
    if output_video_path is not None:
        actual_codec = probe_video_codec(output_video_path)
        summary["output_codec"] = actual_codec
        summary["browser_playable"] = actual_codec in BROWSER_PLAYABLE_CODECS
        if actual_codec and not summary["browser_playable"]:
            LOGGER.warning(
                "La video annotee est encodee en '%s' : les navigateurs ne la liront pas "
                "nativement. Telechargez-la pour l'ouvrir dans un lecteur local.",
                actual_codec,
            )
    else:
        summary["output_codec"] = ""
        summary["browser_playable"] = False

    if tracker is not None:
        # Bilan par personne suivie : c'est la vue qui a un sens operationnel,
        # puisqu'elle compte des personnes et non des detections repetees.
        summary["tracked_compliance"] = tracker.summary()
        summary["alerts"] = new_alerts
        summary["n_alerts"] = len(new_alerts)

    if per_frame:
        all_persons = [p for entry in per_frame for p in entry.get("compliance", [])]
        if all_persons:
            summary["per_detection_compliance"] = summarise_compliance(all_persons)
        write_json(output_dir / "video_predictions.json", {"summary": summary, "frames": per_frame})
    write_json(output_dir / "video_summary.json", summary)
    return summary
