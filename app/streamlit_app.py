"""Interface locale Streamlit pour la detection d'EPI.

Lancement :

    streamlit run app/streamlit_app.py

Quatre modes :

* **Image** — televersement d'une photo ;
* **Video** — televersement d'un fichier, traitement puis lecture annotee ;
* **Webcam (direct)** — inference en continu sur la camera locale ;
* **Resultats** — relecture des videos annotees deja produites.

Cette interface est strictement dediee a l'inference : elle ne declenche jamais
d'entrainement. Si aucun poids entraine n'est disponible, un message explicite
indique la commande a executer.

Note sur la webcam : la capture est faite **cote serveur**, par le processus
Streamlit. C'est adapte a un usage local (le navigateur et la camera sont sur la
meme machine). Pour un deploiement distant, il faudrait passer par WebRTC.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# Permet un lancement direct (`streamlit run app/streamlit_app.py`) meme si le
# paquet n'a pas ete installe en mode editable.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ppe_detection.compliance import ComplianceTracker, summarise_compliance  # noqa: E402
from ppe_detection.config import (  # noqa: E402
    ComplianceConfig,
    InferenceConfig,
    load_compliance_config,
    load_inference_config,
)
from ppe_detection.predict import PPEDetector, PredictionError  # noqa: E402
from ppe_detection.taxonomy import display_name  # noqa: E402
from ppe_detection.utils import ensure_dir, is_video, project_root  # noqa: E402

ROOT = project_root()
CONFIG_PATH = ROOT / "configs" / "inference.yaml"
WEIGHTS_DIRS = [ROOT / "artifacts" / "models", ROOT / "artifacts" / "runs"]
RESULTS_DIR = ROOT / "artifacts" / "predictions"
STREAMLIT_OUT = RESULTS_DIR / "streamlit"

st.set_page_config(page_title="Detection EPI", page_icon="🦺", layout="wide")


# --------------------------------------------------------------------------- #
# Ressources
# --------------------------------------------------------------------------- #
def discover_weights() -> list[Path]:
    """Liste les poids ``.pt`` disponibles, les plus recents en premier."""
    found: list[Path] = []
    for directory in WEIGHTS_DIRS:
        if directory.is_dir():
            found.extend(p for p in directory.rglob("*.pt") if p.is_file())
    unique = {path.resolve(): path for path in found}
    return sorted(unique.values(), key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_resource(show_spinner=False)
def load_detector(
    weights: str,
    conf: float,
    iou: float,
    device: str,
    compliance_enabled: bool,
    required: tuple[str, ...],
    fingerprint: tuple[int, int] = (0, 0),  # noqa: ARG001 - cle de cache uniquement
) -> PPEDetector:
    """Charge (et met en cache) un detecteur pour un jeu de parametres donne.

    Args:
        fingerprint: ``(mtime_ns, taille)`` du fichier de poids. Indispensable :
            un reentrainement reecrit ``best.pt`` sans changer son chemin, et le
            cache indexe sur le seul chemin continuerait de servir l'ancien
            modele. Ce couple force le rechargement des que le fichier change.
    """
    inference_config: InferenceConfig = load_inference_config(
        CONFIG_PATH, weights=weights, conf=conf, iou=iou, device=device
    )
    compliance_config: ComplianceConfig = load_compliance_config(CONFIG_PATH)
    compliance_config.enabled = compliance_enabled
    if required:
        compliance_config.required_ppe = list(required)
    return PPEDetector(inference_config, compliance=compliance_config)


def render_no_weights_message() -> None:
    """Affiche la marche a suivre lorsqu'aucun poids n'est disponible."""
    st.error("Aucun modele entraine n'a ete trouve.")
    st.markdown(
        """
Cette interface ne fait qu'exploiter un modele existant : elle n'entraine rien.

**Pour produire des poids, executez d'abord, depuis la racine du projet :**

```powershell
python -m ppe_detection.dataset_cleaner --source data.yaml --output artifacts/dataset_detection
python -m ppe_detection.train --config configs/train.yaml --smoke
python -m ppe_detection.train --config configs/train.yaml
```

Les poids apparaitront dans `artifacts/models/`.
        """
    )


def compliance_metrics(persons: list[dict]) -> None:
    """Affiche les compteurs de conformite a trois etats."""
    summary = summarise_compliance(persons)
    columns = st.columns(4)
    columns[0].metric("Personnes", summary["persons_detected"])
    columns[1].metric("Conformes", summary["compliant"])
    columns[2].metric("Non conformes", summary["non_compliant"])
    columns[3].metric("Indetermines", summary["indeterminate"])
    if summary["indeterminate"]:
        st.caption(
            "« Indetermine » = zone tronquee par le bord du cadre ou trop petite pour "
            "que l'EPI soit resoluble. Aucune conclusion n'en est tiree."
        )


def compliance_table(persons: list[dict]) -> None:
    """Tableau detaille des verdicts par personne."""
    if not persons:
        return
    rows = []
    for person in persons:
        status = person.get("smoothed_status") or person.get("status", "")
        rows.append(
            {
                "ID": person.get("track_id", person.get("person_index")),
                "Statut": {
                    "compliant": "CONFORME",
                    "non_compliant": "NON CONFORME",
                    "indeterminate": "INDETERMINE",
                }.get(str(status), str(status)),
                "EPI détectés": ", ".join(
                    display_name(str(x)) for x in (person.get("detected_ppe") or [])
                ) or "-",
                "EPI manquants": ", ".join(
                    display_name(str(x)) for x in (person.get("missing_ppe") or [])
                ) or "-",
                "Indéterminés": ", ".join(
                    display_name(str(x)) for x in (person.get("indeterminate_ppe") or [])
                ) or "-",
                "Confiance": person.get("verdict_confidence"),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)
    reasons = [r for person in persons for r in (person.get("reasons") or [])]
    if reasons:
        with st.expander(f"Pourquoi certains EPI sont indetermines ({len(reasons)})"):
            for reason in dict.fromkeys(reasons):
                st.write(f"- {reason}")


# --------------------------------------------------------------------------- #
# Mode image
# --------------------------------------------------------------------------- #
def mode_image(detector: PPEDetector) -> None:
    """Analyse d'une image televersee."""
    uploaded = st.file_uploader(
        "Image a analyser", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"]
    )
    if uploaded is None:
        st.info("Deposez une image pour lancer l'analyse.")
        return

    import cv2
    import numpy as np

    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Image illisible ou corrompue.")
        return

    with st.spinner("Analyse en cours..."):
        prediction = detector.predict_array(image, source_name=uploaded.name)
        annotated = detector.annotate(image, prediction)

    left, right = st.columns(2)
    left.subheader("Image d'origine")
    left.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), width="stretch")
    right.subheader("Detections")
    right.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")

    st.subheader("Detections")
    if prediction.detections:
        st.dataframe(
            [
                {
                    "Classe": d.get("class_name_fr") or display_name(d["class_name"]),
                    "Confiance": round(d["confidence"], 3),
                    "x1": d["bbox_xyxy"][0],
                    "y1": d["bbox_xyxy"][1],
                    "x2": d["bbox_xyxy"][2],
                    "y2": d["bbox_xyxy"][3],
                }
                for d in prediction.detections
            ],
            width="stretch",
            hide_index=True,
        )
        counts: dict[str, int] = {}
        for detection in prediction.detections:
            label = detection.get("class_name_fr") or display_name(detection["class_name"])
            counts[label] = counts.get(label, 0) + 1
        st.bar_chart(counts)
    else:
        st.warning("Aucun objet detecte au seuil choisi. Essayez de l'abaisser.")

    if prediction.compliance:
        st.subheader("Conformite (heuristique)")
        compliance_metrics(prediction.compliance)
        compliance_table(prediction.compliance)

    st.download_button(
        "Telecharger les resultats (JSON)",
        data=json.dumps(prediction.to_dict(), indent=2, ensure_ascii=False),
        file_name=f"{Path(uploaded.name).stem}_detections.json",
        mime="application/json",
    )
    success, buffer = cv2.imencode(".jpg", annotated)
    if success:
        st.download_button(
            "Telecharger l'image annotee",
            data=buffer.tobytes(),
            file_name=f"{Path(uploaded.name).stem}_annotated.jpg",
            mime="image/jpeg",
        )


# --------------------------------------------------------------------------- #
# Mode video
# --------------------------------------------------------------------------- #
def mode_video(detector: PPEDetector, track: bool) -> None:
    """Analyse d'une video televersee, avec lecture du resultat annote."""
    from ppe_detection.video import VideoSourceError, run_video_inference

    uploaded = st.file_uploader("Video a analyser", type=["mp4", "avi", "mov", "mkv", "m4v"])
    if uploaded is None:
        st.info("Deposez une video pour lancer l'analyse.")
        return

    columns = st.columns(2)
    max_frames = columns[0].number_input(
        "Frames analysees au maximum", min_value=10, max_value=10000, value=500, step=10
    )
    frame_skip = columns[1].number_input(
        "Frames sautees entre deux inferences",
        min_value=0,
        max_value=30,
        value=0,
        help="Augmentez pour accelerer le traitement des videos longues.",
    )

    if not st.button("Lancer l'analyse", type="primary"):
        return

    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(STREAMLIT_OUT / f"video_{stamp}")

    with tempfile.TemporaryDirectory() as temporary:
        # Nom fixe : le nom d'origine n'est jamais reutilise pour ecrire sur disque.
        video_path = Path(temporary) / f"input{suffix}"
        video_path.write_bytes(uploaded.getvalue())

        try:
            with st.spinner("Traitement de la video..."):
                summary = run_video_inference(
                    detector,
                    str(video_path),
                    output_dir=output_dir,
                    save_video=True,
                    show=False,
                    frame_skip=int(frame_skip),
                    max_frames=int(max_frames),
                    save_json=True,
                    output_name="annotated.mp4",
                    track=track,
                )
        except VideoSourceError as exc:
            st.error(f"Video illisible : {exc}")
            return

    _render_video_summary(summary, output_dir, Path(uploaded.name).stem)


def _render_video_summary(summary: dict, output_dir: Path, stem: str) -> None:
    """Affiche les indicateurs, la video annotee et les telechargements."""
    columns = st.columns(4)
    columns[0].metric("Frames lues", summary["frames_read"])
    columns[1].metric("Frames analysees", summary["frames_processed"])
    columns[2].metric("Detections", summary["total_detections"])
    columns[3].metric("FPS moyen", summary["average_fps"])

    tracked = summary.get("tracked_compliance")
    if tracked:
        st.subheader("Conformite par personne suivie")
        cols = st.columns(4)
        cols[0].metric("Personnes suivies", tracked["tracked_persons"])
        cols[1].metric("Conformes", tracked["compliant"])
        cols[2].metric("Non conformes", tracked["non_compliant"])
        cols[3].metric("Indetermines", tracked["indeterminate"])
        st.caption(
            f"{summary.get('n_alerts', 0)} evenement(s) d'alerte — une alerte est levee "
            f"une seule fois par personne, apres accord d'au moins "
            f"{tracked['min_ratio']:.0%} des {tracked['window']} derniers verdicts."
        )
        alerts = summary.get("alerts") or []
        if alerts:
            st.dataframe(
                [
                    {
                        "Frame": a["frame_index"],
                        "Personne": a["track_id"],
                        "EPI manquants": ", ".join(a.get("missing_ppe") or []) or "-",
                    }
                    for a in alerts
                ],
                width="stretch",
                hide_index=True,
            )

    annotated = Path(summary["output_video"]) if summary.get("output_video") else None
    if annotated and annotated.is_file():
        st.subheader("Video annotee")
        if not summary.get("browser_playable", True):
            st.warning(
                f"Codec reellement ecrit : « {summary.get('output_codec')} ». Les navigateurs ne "
                f"le lisent pas nativement — telechargez la video pour l'ouvrir dans un lecteur "
                f"local (VLC, lecteur Windows)."
            )
        else:
            st.caption(f"Codec : {summary.get('output_codec')} — lisible directement ici.")
        video_bytes = annotated.read_bytes()
        st.video(video_bytes)
        st.download_button(
            "Telecharger la video annotee",
            data=video_bytes,
            file_name=f"{stem}_annotated.mp4",
            mime="video/mp4",
        )
    else:
        st.warning("La video annotee n'a pas pu etre produite. Les detections restent en JSON.")

    details = output_dir / "video_predictions.json"
    if details.is_file():
        st.download_button(
            "Telecharger les detections (JSON)",
            data=details.read_text(encoding="utf-8"),
            file_name=f"{stem}_detections.json",
            mime="application/json",
        )
    st.caption(f"Resultats conserves dans `{output_dir.relative_to(ROOT)}`")


# --------------------------------------------------------------------------- #
# Mode webcam
# --------------------------------------------------------------------------- #
def mode_webcam(detector: PPEDetector, track: bool) -> None:
    """Inference en continu sur la camera locale."""
    import cv2

    from ppe_detection.video import create_video_writer, open_capture

    st.caption(
        "La camera est ouverte par le processus Streamlit (cote serveur). "
        "Adapte a un usage local ; pour un acces distant il faudrait WebRTC."
    )

    controls = st.columns(4)
    camera_index = controls[0].number_input("Index camera", min_value=0, max_value=10, value=0)
    max_seconds = controls[1].number_input(
        "Duree maximale (s)", min_value=5, max_value=3600, value=120,
        help="Garde-fou : la boucle s'arrete d'elle-meme au bout de ce delai.",
    )
    display_width = controls[2].select_slider(
        "Largeur d'affichage", options=[480, 640, 800, 960, 1280], value=800
    )
    record = controls[3].checkbox("Enregistrer", value=False)

    start_col, stop_col = st.columns(2)
    if start_col.button("Demarrer la camera", type="primary", width="stretch"):
        st.session_state["webcam_running"] = True
    if stop_col.button("Arreter", width="stretch"):
        st.session_state["webcam_running"] = False

    if not st.session_state.get("webcam_running"):
        snapshot = st.camera_input("Ou prendre une photo ponctuelle")
        if snapshot is not None:
            _analyse_snapshot(detector, snapshot)
        else:
            st.info("Cliquez sur « Demarrer la camera » pour l'inference en continu.")
        return

    frame_slot = st.empty()
    metric_slot = st.empty()
    alert_slot = st.empty()

    try:
        capture, label = open_capture(str(int(camera_index)))
    except Exception as exc:  # noqa: BLE001 - VideoSourceError et erreurs OpenCV
        st.session_state["webcam_running"] = False
        st.error(str(exc))
        return

    tracker: ComplianceTracker | None = None
    compliance_config = detector.compliance_config
    if track and compliance_config is not None and compliance_config.enabled:
        tracker = ComplianceTracker(compliance_config)

    writer = None
    output_dir = ensure_dir(STREAMLIT_OUT / f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    started = time.perf_counter()
    frames = 0
    alerts: list[str] = []

    try:
        while st.session_state.get("webcam_running"):
            ok, frame = capture.read()
            if not ok or frame is None:
                st.warning("Flux camera interrompu.")
                break
            frames += 1

            prediction = detector.predict_array(frame, source_name=f"{label}#{frames}", track=track)
            if tracker is not None:
                prediction.compliance = tracker.update(prediction.compliance)
                for person in prediction.compliance:
                    if person.get("is_new_alert"):
                        missing = ", ".join(person.get("missing_ppe") or []) or "?"
                        alerts.append(
                            f"Personne #{person.get('track_id')} — {missing} manquant(s)"
                        )

            annotated = detector.annotate(frame, prediction)

            if record:
                if writer is None:
                    writer, _codec = create_video_writer(
                        output_dir / "webcam_annotated.mp4",
                        15.0,
                        (annotated.shape[1], annotated.shape[0]),
                    )
                if writer is not None:
                    writer.write(annotated)

            frame_slot.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width=int(display_width)
            )

            elapsed = max(time.perf_counter() - started, 1e-6)
            with metric_slot.container():
                cols = st.columns(4)
                cols[0].metric("Frames", frames)
                cols[1].metric("FPS", f"{frames / elapsed:.1f}")
                cols[2].metric("Objets", len(prediction.detections))
                if tracker is not None:
                    cols[3].metric("Personnes suivies", tracker.summary()["tracked_persons"])
                else:
                    cols[3].metric("Personnes", len(prediction.compliance))

            if alerts:
                alert_slot.error("  \n".join(f"⚠ {a}" for a in alerts[-5:]))

            if elapsed > float(max_seconds):
                st.info(f"Duree maximale de {max_seconds} s atteinte — arret automatique.")
                st.session_state["webcam_running"] = False
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            st.success(f"Enregistrement : `{(output_dir / 'webcam_annotated.mp4').relative_to(ROOT)}`")

    if tracker is not None:
        st.subheader("Bilan de la session")
        st.json(tracker.summary())


def _analyse_snapshot(detector: PPEDetector, snapshot) -> None:  # noqa: ANN001 - type Streamlit
    """Analyse une photo prise via ``st.camera_input``."""
    import cv2
    import numpy as np

    data = np.frombuffer(snapshot.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Photo illisible.")
        return
    prediction = detector.predict_array(image, source_name="webcam_snapshot")
    annotated = detector.annotate(image, prediction)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
    if prediction.compliance:
        compliance_metrics(prediction.compliance)
        compliance_table(prediction.compliance)
    st.download_button(
        "Telecharger les resultats (JSON)",
        data=json.dumps(prediction.to_dict(), indent=2, ensure_ascii=False),
        file_name="webcam_snapshot.json",
        mime="application/json",
    )


# --------------------------------------------------------------------------- #
# Mode resultats
# --------------------------------------------------------------------------- #
def mode_results() -> None:
    """Relecture des videos annotees deja produites."""
    st.subheader("Videos annotees produites")
    if not RESULTS_DIR.is_dir():
        st.info("Aucun resultat pour l'instant.")
        return

    videos = sorted(
        (p for p in RESULTS_DIR.rglob("*.mp4") if p.is_file() and is_video(p)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not videos:
        st.info(
            "Aucune video annotee trouvee dans `artifacts/predictions/`. "
            "Lancez une analyse video ou une session webcam enregistree."
        )
        return

    labels = [
        f"{p.relative_to(RESULTS_DIR)}  —  {p.stat().st_size / 1e6:.1f} Mo  —  "
        f"{datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}"
        for p in videos
    ]
    index = st.selectbox(
        "Video", options=range(len(videos)), format_func=lambda i: labels[i]
    )
    selected = videos[index]

    import cv2

    from ppe_detection.video import BROWSER_PLAYABLE_CODECS, probe_video_codec

    codec = probe_video_codec(selected)
    capture = cv2.VideoCapture(str(selected))
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    capture.release()

    if codec and codec not in BROWSER_PLAYABLE_CODECS:
        st.warning(
            f"Cette video est encodee en « {codec} », un format que les navigateurs ne lisent "
            f"pas nativement — elle restera noire ci-dessous. Elle a probablement ete produite "
            f"avant la correction du codec : relancez l'analyse pour obtenir du H.264, ou "
            f"telechargez le fichier pour l'ouvrir dans VLC."
        )
    st.caption(f"Codec : {codec or 'inconnu'} — {n_frames} frames")

    st.video(selected.read_bytes())
    st.download_button(
        "Telecharger", data=selected.read_bytes(), file_name=selected.name, mime="video/mp4"
    )

    summary_file = selected.parent / "video_summary.json"
    if summary_file.is_file():
        with st.expander("Resume de l'execution"):
            st.json(json.loads(summary_file.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
def main() -> None:
    """Point d'entree de l'interface."""
    st.title("🦺 Detection d'equipements de protection individuelle")
    st.caption(
        "Interface locale d'inference. L'entrainement se lance en ligne de commande, "
        "jamais depuis cette page."
    )

    weights_list = discover_weights()
    if not weights_list:
        render_no_weights_message()
        return

    with st.sidebar:
        st.header("Modele")
        labels = [
            str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in weights_list
        ]
        choice = st.selectbox(
            "Poids", options=range(len(weights_list)), format_func=lambda i: labels[i]
        )
        weights_path = weights_list[choice]
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(weights_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        st.caption(f"{size_mb:.1f} Mo — modifie le {modified}")

        device = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)

        st.header("Seuils")
        conf = st.slider("Confiance", 0.01, 0.95, 0.25, 0.01)
        iou = st.slider("IoU (NMS)", 0.10, 0.95, 0.45, 0.05)

        st.header("Conformite EPI")
        compliance_enabled = st.checkbox("Activer la verification de conformite", value=False)
        required: tuple[str, ...] = ()
        track = False
        if compliance_enabled:
            base_config = load_compliance_config(CONFIG_PATH)
            options = [
                "Safety Helmet", "Safety Vest", "Safety Gloves",
                "Safety Shoes", "Safety Harness", "Face Mask",
            ]
            # Les options portent l'identifiant interne (les regles metier s'y
            # referent) mais s'affichent en francais.
            required = tuple(
                st.multiselect(
                    "EPI obligatoires",
                    options=options,
                    default=base_config.required_ppe,
                    format_func=display_name,
                )
            )
            track = st.checkbox(
                "Suivi + lissage temporel (video/webcam)",
                value=True,
                help=(
                    "Attribue un identifiant persistant a chaque personne et ne leve une "
                    "alerte qu'apres plusieurs verdicts concordants."
                ),
            )
            st.info(
                "L'association EPI/personne est une **heuristique geometrique**. "
                "Un statut « non conforme » est une alerte a verifier, pas un constat.",
                icon="⚠️",
            )

    try:
        stat = weights_path.stat()
        detector = load_detector(
            str(weights_path),
            conf,
            iou,
            device,
            compliance_enabled,
            required,
            fingerprint=(stat.st_mtime_ns, stat.st_size),
        )
    except PredictionError as exc:
        st.error(f"Chargement du modele impossible : {exc}")
        return

    info = detector.model_info()
    columns = st.columns(4)
    columns[0].metric("Classes", info["num_classes"])
    columns[1].metric("Parametres", f"{(info['parameters'] or 0) / 1e6:.2f} M")
    columns[2].metric("Device", info["device"])
    columns[3].metric("Taille d'inference", info["imgsz"])
    with st.expander("Classes reconnues"):
        st.write(", ".join(display_name(n) for n in info["class_names"]))

    st.divider()
    image_tab, video_tab, webcam_tab, results_tab = st.tabs(
        ["Image", "Video", "Webcam (direct)", "Resultats"]
    )
    with image_tab:
        mode_image(detector)
    with video_tab:
        mode_video(detector, track)
    with webcam_tab:
        mode_webcam(detector, track)
    with results_tab:
        mode_results()


if __name__ == "__main__":
    main()
