"""Tests de la logique de conformite, du rendu et de l'inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppe_detection.compliance import (
    STATUS_COMPLIANT,
    STATUS_INDETERMINATE,
    STATUS_NON_COMPLIANT,
    ComplianceTracker,
    box_area,
    containment_ratio,
    evaluate_compliance,
    intersection_area,
    person_region,
    region_observability,
    summarise_compliance,
)
from ppe_detection.config import ComplianceConfig, InferenceConfig
from ppe_detection.predict import (
    ImagePrediction,
    PredictionError,
    classify_source,
    is_stream_source,
    is_webcam_source,
    save_predictions_csv,
    save_predictions_json,
    save_yolo_txt,
)
from ppe_detection.visualization import color_for_class, draw_detections, draw_yolo_labels


def detection(name: str, bbox: list[float], confidence: float = 0.9, class_id: int = 0) -> dict:
    """Fabrique une detection synthetique."""
    return {
        "class_id": class_id,
        "class_name": name,
        "confidence": confidence,
        "bbox_xyxy": bbox,
    }


# --------------------------------------------------------------------------- #
# Geometrie
# --------------------------------------------------------------------------- #
def test_box_and_intersection_area() -> None:
    assert box_area([0, 0, 10, 10]) == pytest.approx(100.0)
    assert box_area([10, 10, 0, 0]) == 0.0
    assert intersection_area([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25.0)
    assert intersection_area([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_containment_ratio_is_asymmetric() -> None:
    small = [4, 4, 6, 6]
    large = [0, 0, 10, 10]
    assert containment_ratio(small, large) == pytest.approx(1.0)
    assert containment_ratio(large, small) == pytest.approx(0.04)
    assert containment_ratio([0, 0, 0, 0], large) == 0.0


def test_person_regions_partition_correctly() -> None:
    config = ComplianceConfig(helmet_region=0.35, shoes_region=0.30, torso_region=(0.2, 0.8))
    person = [0.0, 0.0, 100.0, 200.0]
    head = person_region(person, "head", config)
    feet = person_region(person, "feet", config)
    torso = person_region(person, "torso", config)
    whole = person_region(person, "any", config)

    assert head == (0.0, 0.0, 100.0, 70.0)
    assert feet == (0.0, 140.0, 100.0, 200.0)
    assert torso == (0.0, 40.0, 100.0, 160.0)
    assert whole == (0.0, 0.0, 100.0, 200.0)


# --------------------------------------------------------------------------- #
# Conformite
# --------------------------------------------------------------------------- #
def test_person_with_all_required_ppe_is_compliant() -> None:
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet", "Safety Vest"])
    detections = [
        detection("Person", [0, 0, 100, 200], 0.95, 1),
        detection("Safety Helmet", [30, 5, 70, 45], 0.90, 4),   # zone tete
        detection("Safety Vest", [25, 60, 75, 140], 0.85, 6),   # zone torse
    ]
    result = evaluate_compliance(detections, config)
    assert len(result) == 1
    person = result[0]
    assert person["compliant"] is True
    assert person["missing_ppe"] == []
    assert set(person["detected_ppe"]) == {"Safety Helmet", "Safety Vest"}
    assert person["verdict_confidence"] == pytest.approx(0.85)


def test_missing_ppe_is_reported() -> None:
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet", "Safety Vest"])
    detections = [
        detection("Person", [0, 0, 100, 200], 0.95, 1),
        detection("Safety Helmet", [30, 5, 70, 45], 0.90, 4),
    ]
    person = evaluate_compliance(detections, config)[0]
    assert person["compliant"] is False
    assert person["missing_ppe"] == ["Safety Vest"]
    # Pour une non-conformite, la confiance porte sur la detection de la personne.
    assert person["verdict_confidence"] == pytest.approx(0.95)


def test_helmet_in_wrong_region_is_not_attributed() -> None:
    """Un casque au niveau des pieds ne doit pas compter comme porte."""
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"])
    detections = [
        detection("Person", [0, 0, 100, 200], 0.95, 1),
        detection("Safety Helmet", [30, 170, 70, 195], 0.90, 4),  # tout en bas
    ]
    person = evaluate_compliance(detections, config)[0]
    assert person["detected_ppe"] == []
    assert person["missing_ppe"] == ["Safety Helmet"]


def test_ppe_assigned_to_single_best_person() -> None:
    """Un EPI ne doit etre attribue qu'a une seule personne."""
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"])
    detections = [
        detection("Person", [0, 0, 100, 200], 0.95, 1),
        detection("Person", [90, 0, 190, 200], 0.95, 1),
        detection("Safety Helmet", [20, 5, 60, 45], 0.90, 4),
    ]
    result = evaluate_compliance(detections, config)
    assigned = [person for person in result if person["detected_ppe"]]
    assert len(assigned) == 1
    assert assigned[0]["compliant"] is True
    assert sum(1 for p in result if not p["compliant"]) == 1


def test_no_person_yields_no_verdict() -> None:
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"])
    detections = [detection("Safety Helmet", [30, 5, 70, 45], 0.9, 4)]
    assert evaluate_compliance(detections, config) == []


def test_low_confidence_detections_are_ignored() -> None:
    config = ComplianceConfig(
        enabled=True, required_ppe=["Safety Helmet"], min_person_conf=0.5, min_ppe_conf=0.5
    )
    detections = [
        detection("Person", [0, 0, 100, 200], 0.95, 1),
        detection("Safety Helmet", [30, 5, 70, 45], 0.20, 4),  # sous le seuil
    ]
    person = evaluate_compliance(detections, config)[0]
    assert person["missing_ppe"] == ["Safety Helmet"]


def test_containment_threshold_is_enforced() -> None:
    """Un casque a moitie hors de la zone tete ne passe pas le seuil de 0.5."""
    config = ComplianceConfig(
        enabled=True, required_ppe=["Safety Helmet"], containment_threshold=0.9
    )
    detections = [
        detection("Person", [0, 0, 100, 200], 0.95, 1),
        detection("Safety Helmet", [-30, 5, 20, 45], 0.9, 4),  # majoritairement a gauche
    ]
    person = evaluate_compliance(detections, config)[0]
    assert person["detected_ppe"] == []


def test_compliance_response_documents_heuristic() -> None:
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"])
    detections = [detection("Person", [0, 0, 100, 200], 0.9, 1)]
    person = evaluate_compliance(detections, config)[0]
    assert "heuristic" in person
    assert "verifier" in person["heuristic"].lower()


def test_summarise_compliance_counts() -> None:
    persons = [
        {"status": STATUS_COMPLIANT, "missing_ppe": [], "indeterminate_ppe": []},
        {"status": STATUS_NON_COMPLIANT, "missing_ppe": ["Safety Vest"], "indeterminate_ppe": []},
        {
            "status": STATUS_NON_COMPLIANT,
            "missing_ppe": ["Safety Vest", "Safety Helmet"],
            "indeterminate_ppe": [],
        },
    ]
    summary = summarise_compliance(persons)
    assert summary["persons_detected"] == 3
    assert summary["compliant"] == 1
    assert summary["non_compliant"] == 2
    assert summary["compliance_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["missing_ppe_counts"]["Safety Vest"] == 2


def test_summarise_compliance_excludes_indeterminate_from_rate() -> None:
    """Une personne non observable ne doit pas faire chuter le taux de conformite."""
    persons = [
        {"status": STATUS_COMPLIANT, "missing_ppe": [], "indeterminate_ppe": []},
        {"status": STATUS_INDETERMINATE, "missing_ppe": [], "indeterminate_ppe": ["Safety Helmet"]},
        {"status": STATUS_INDETERMINATE, "missing_ppe": [], "indeterminate_ppe": ["Safety Helmet"]},
    ]
    summary = summarise_compliance(persons)
    assert summary["persons_detected"] == 3
    assert summary["indeterminate"] == 2
    assert summary["decidable"] == 1
    # 1 conforme sur 1 jugeable, et non 1 sur 3.
    assert summary["compliance_rate"] == pytest.approx(1.0)
    assert summary["indeterminate_ppe_counts"]["Safety Helmet"] == 2


def test_summarise_compliance_handles_empty() -> None:
    summary = summarise_compliance([])
    assert summary["persons_detected"] == 0
    assert summary["compliance_rate"] is None


# --------------------------------------------------------------------------- #
# Niveau 1 — observabilite et etat indetermine
# --------------------------------------------------------------------------- #
def test_region_observable_for_large_centered_person() -> None:
    config = ComplianceConfig()
    result = region_observability([100, 100, 200, 400], "head", config, (640, 480))
    assert result.observable is True


def test_region_not_observable_when_too_small() -> None:
    """Une zone de quelques pixels ne permet aucune conclusion."""
    config = ComplianceConfig(min_region_height_px=24)
    # Personne de 30 px de haut -> zone tete = 10.5 px
    result = region_observability([100, 100, 120, 130], "head", config, (640, 480))
    assert result.observable is False
    assert "trop petite" in result.reason


def test_region_not_observable_when_truncated_at_top() -> None:
    """Tete coupee par le bord haut du cadre : le casque peut etre hors champ."""
    config = ComplianceConfig()
    result = region_observability([100, 0, 300, 400], "head", config, (640, 480))
    assert result.observable is False
    assert "tronquee" in result.reason
    assert "haut" in result.reason


def test_feet_region_truncated_at_bottom() -> None:
    config = ComplianceConfig()
    result = region_observability([100, 50, 300, 480], "feet", config, (640, 480))
    assert result.observable is False
    assert "bas" in result.reason


def test_truncation_ignored_without_image_size() -> None:
    """Sans dimensions d'image, le test de troncature ne peut pas s'appliquer."""
    config = ComplianceConfig()
    result = region_observability([100, 0, 300, 400], "head", config, None)
    assert result.observable is True


def test_truncated_person_is_indeterminate_not_non_compliant() -> None:
    """Le coeur du niveau 1 : ne pas accuser une personne qu'on ne peut pas observer."""
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"])
    detections = [detection("Person", [100, 0, 300, 400], 0.95, 1)]  # tete hors cadre
    person = evaluate_compliance(detections, config, image_size=(640, 480))[0]

    assert person["status"] == STATUS_INDETERMINATE
    assert person["compliant"] is False
    assert person["missing_ppe"] == []
    assert person["indeterminate_ppe"] == ["Safety Helmet"]
    assert person["reasons"] and "tronquee" in person["reasons"][0]


def test_observable_person_without_helmet_is_non_compliant() -> None:
    """A l'inverse, une personne bien visible sans casque reste non conforme."""
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"])
    detections = [detection("Person", [100, 50, 300, 400], 0.95, 1)]
    person = evaluate_compliance(detections, config, image_size=(640, 480))[0]

    assert person["status"] == STATUS_NON_COMPLIANT
    assert person["missing_ppe"] == ["Safety Helmet"]
    assert person["indeterminate_ppe"] == []


def test_tiny_person_is_indeterminate() -> None:
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"], min_region_height_px=24)
    # Personne de 40 px de haut au milieu du cadre : zone tete = 14 px.
    detections = [detection("Person", [300, 200, 320, 240], 0.9, 1)]
    person = evaluate_compliance(detections, config, image_size=(640, 480))[0]
    assert person["status"] == STATUS_INDETERMINATE
    assert "trop petite" in person["reasons"][0]


def test_missing_and_indeterminate_together_yields_non_compliant() -> None:
    """Si un EPI manque de facon certaine, le verdict est non conforme."""
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet", "Safety Vest"])
    # Tete tronquee (casque indetermine) mais torse observable (gilet absent).
    detections = [detection("Person", [100, 0, 300, 400], 0.95, 1)]
    person = evaluate_compliance(detections, config, image_size=(640, 480))[0]

    assert person["status"] == STATUS_NON_COMPLIANT
    assert person["missing_ppe"] == ["Safety Vest"]
    assert person["indeterminate_ppe"] == ["Safety Helmet"]


def test_compliant_person_still_detected_with_new_logic() -> None:
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet", "Safety Vest"])
    detections = [
        detection("Person", [100, 50, 200, 250], 0.95, 1),
        detection("Safety Helmet", [130, 55, 170, 95], 0.90, 4),
        detection("Safety Vest", [125, 110, 175, 190], 0.85, 6),
    ]
    person = evaluate_compliance(detections, config, image_size=(640, 480))[0]
    assert person["status"] == STATUS_COMPLIANT
    assert person["compliant"] is True
    assert person["missing_ppe"] == []
    assert person["indeterminate_ppe"] == []


def test_compliance_payload_documents_indeterminate_semantics() -> None:
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"])
    detections = [detection("Person", [100, 50, 300, 400], 0.9, 1)]
    person = evaluate_compliance(detections, config, image_size=(640, 480))[0]
    assert "indeterminate" in person["heuristic"].lower()


def test_track_id_is_propagated_to_verdict() -> None:
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"])
    person_det = detection("Person", [100, 50, 300, 400], 0.9, 1)
    person_det["track_id"] = 7
    person = evaluate_compliance([person_det], config, image_size=(640, 480))[0]
    assert person["track_id"] == 7


# --------------------------------------------------------------------------- #
# Niveau 2 — lissage temporel
# --------------------------------------------------------------------------- #
def _tracked(status: str, track_id: int = 1) -> dict:
    """Verdict instantane minimal pour alimenter le lisseur."""
    return {
        "track_id": track_id,
        "status": status,
        "missing_ppe": ["Safety Vest"] if status == STATUS_NON_COMPLIANT else [],
        "indeterminate_ppe": [],
    }


def test_tracker_waits_for_enough_observations() -> None:
    """Aucune conclusion avant d'avoir assez d'observations."""
    config = ComplianceConfig(temporal_min_observations=5, temporal_window=15)
    tracker = ComplianceTracker(config)
    for _ in range(4):
        result = tracker.update([_tracked(STATUS_NON_COMPLIANT)])[0]
        assert result["smoothed_status"] == STATUS_INDETERMINATE
        assert result["is_new_alert"] is False
    result = tracker.update([_tracked(STATUS_NON_COMPLIANT)])[0]
    assert result["smoothed_status"] == STATUS_NON_COMPLIANT
    assert result["is_new_alert"] is True


def test_tracker_raises_alert_only_once() -> None:
    config = ComplianceConfig(temporal_min_observations=3)
    tracker = ComplianceTracker(config)
    alerts = [tracker.update([_tracked(STATUS_NON_COMPLIANT)])[0]["is_new_alert"] for _ in range(10)]
    assert sum(alerts) == 1


def test_tracker_absorbs_isolated_false_negative() -> None:
    """Une frame aberrante ne doit pas basculer le verdict."""
    config = ComplianceConfig(temporal_min_observations=3, temporal_min_ratio=0.7)
    tracker = ComplianceTracker(config)
    for _ in range(9):
        tracker.update([_tracked(STATUS_COMPLIANT)])
    result = tracker.update([_tracked(STATUS_NON_COMPLIANT)])[0]
    # 9 conformes contre 1 non conforme : le verdict reste conforme.
    assert result["smoothed_status"] == STATUS_COMPLIANT
    assert result["is_new_alert"] is False


def test_tracker_reports_unstable_as_indeterminate() -> None:
    """Alternance sans majorite nette : aucune conclusion."""
    config = ComplianceConfig(temporal_min_observations=4, temporal_min_ratio=0.8)
    tracker = ComplianceTracker(config)
    statuses = [STATUS_COMPLIANT, STATUS_NON_COMPLIANT] * 4
    for status in statuses:
        result = tracker.update([_tracked(status)])[0]
    assert result["smoothed_status"] == STATUS_INDETERMINATE


def test_tracker_ignores_indeterminate_in_ratio() -> None:
    """Les verdicts indetermines ne comptent pas dans la majorite."""
    config = ComplianceConfig(temporal_min_observations=3, temporal_min_ratio=0.7)
    tracker = ComplianceTracker(config)
    for _ in range(5):
        tracker.update([_tracked(STATUS_INDETERMINATE)])
    result = tracker.update([_tracked(STATUS_INDETERMINATE)])[0]
    assert result["observations"] == 0
    assert result["smoothed_status"] == STATUS_INDETERMINATE
    for _ in range(3):
        result = tracker.update([_tracked(STATUS_NON_COMPLIANT)])[0]
    assert result["observations"] == 3
    assert result["smoothed_status"] == STATUS_NON_COMPLIANT


def test_tracker_separates_persons() -> None:
    config = ComplianceConfig(temporal_min_observations=3)
    tracker = ComplianceTracker(config)
    for _ in range(5):
        tracker.update([_tracked(STATUS_COMPLIANT, 1), _tracked(STATUS_NON_COMPLIANT, 2)])
    summary = tracker.summary()
    assert summary["tracked_persons"] == 2
    assert summary["compliant"] == 1
    assert summary["non_compliant"] == 1
    assert summary["persons_currently_alerted"] == 1


def test_tracker_passthrough_without_track_id() -> None:
    """Sans suivi, le verdict instantane est renvoye tel quel."""
    tracker = ComplianceTracker(ComplianceConfig())
    result = tracker.update([{"status": STATUS_NON_COMPLIANT, "missing_ppe": ["Safety Vest"]}])[0]
    assert result["smoothed_status"] == STATUS_NON_COMPLIANT
    assert result["is_new_alert"] is False


def test_tracker_rearms_after_person_becomes_compliant() -> None:
    config = ComplianceConfig(temporal_min_observations=3, temporal_min_ratio=0.7)
    tracker = ComplianceTracker(config)
    for _ in range(5):
        tracker.update([_tracked(STATUS_NON_COMPLIANT)])
    for _ in range(15):
        tracker.update([_tracked(STATUS_COMPLIANT)])
    assert tracker.summary()["persons_currently_alerted"] == 0
    for _ in range(15):
        result = tracker.update([_tracked(STATUS_NON_COMPLIANT)])[0]
    assert result["smoothed_status"] == STATUS_NON_COMPLIANT


def test_tracker_reset_clears_state() -> None:
    tracker = ComplianceTracker(ComplianceConfig())
    tracker.update([_tracked(STATUS_NON_COMPLIANT)])
    tracker.reset()
    assert tracker.summary()["tracked_persons"] == 0


# --------------------------------------------------------------------------- #
# Ecriture video : codec lisible par navigateur
# --------------------------------------------------------------------------- #
def test_create_video_writer_prefers_browser_playable_codec(tmp_path: Path) -> None:
    """Le fichier produit doit etre du H.264, lisible dans une balise <video>.

    'mp4v' produit du MPEG-4 Part 2 (FOURCC 'FMP4') qu'aucun navigateur ne
    decode : la video annotee serait illisible dans Streamlit.
    """
    import cv2

    from ppe_detection.video import create_video_writer

    target = tmp_path / "out.mp4"
    writer, codec = create_video_writer(target, 10.0, (320, 240))
    assert writer is not None
    for value in range(12):
        writer.write(np.full((240, 320, 3), value * 20, dtype=np.uint8))
    writer.release()

    assert codec == "avc1"
    assert target.is_file() and target.stat().st_size > 0

    capture = cv2.VideoCapture(str(target))
    assert capture.isOpened()
    fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC))
    stored = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        frames += 1
    capture.release()

    assert stored.lower() == "h264"
    assert frames == 12


def test_probe_video_codec_reports_actual_format(tmp_path: Path) -> None:
    """OpenCV substitue silencieusement un codec : seule la relecture fait foi."""
    from ppe_detection.video import create_video_writer, probe_video_codec

    h264_path = tmp_path / "h264.mp4"
    writer, _ = create_video_writer(h264_path, 10.0, (320, 240), codecs=("avc1",))
    assert writer is not None
    for value in range(8):
        writer.write(np.full((240, 320, 3), value * 25, dtype=np.uint8))
    writer.release()

    mp4v_path = tmp_path / "mp4v.mp4"
    writer, _ = create_video_writer(mp4v_path, 10.0, (320, 240), codecs=("mp4v",))
    assert writer is not None
    for value in range(8):
        writer.write(np.full((240, 320, 3), value * 25, dtype=np.uint8))
    writer.release()

    assert probe_video_codec(h264_path) == "h264"
    # 'mp4v' produit un flux MPEG-4 Part 2, illisible par un navigateur.
    assert probe_video_codec(mp4v_path) == "fmp4"


def test_probe_video_codec_on_missing_file(tmp_path: Path) -> None:
    from ppe_detection.video import probe_video_codec

    assert probe_video_codec(tmp_path / "absent.mp4") == ""


def test_browser_playable_set_contains_h264() -> None:
    from ppe_detection.video import BROWSER_PLAYABLE_CODECS

    assert "h264" in BROWSER_PLAYABLE_CODECS
    assert "fmp4" not in BROWSER_PLAYABLE_CODECS


# --------------------------------------------------------------------------- #
# Classification des sources
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["0", "1", "webcam", "CAMERA"])
def test_webcam_sources(value: str) -> None:
    assert is_webcam_source(value)


@pytest.mark.parametrize("value", ["rtsp://host/stream", "http://x/y.mjpg", "RTMP://a/b"])
def test_stream_sources(value: str) -> None:
    assert is_stream_source(value)


def test_classify_source_for_files_and_dirs(sample_image: Path, tmp_path: Path) -> None:
    assert classify_source(str(sample_image)) == "image"
    assert classify_source(str(tmp_path)) == "directory"
    assert classify_source("0") == "webcam"
    assert classify_source("rtsp://host/stream") == "stream"


def test_classify_source_missing_path_raises() -> None:
    with pytest.raises(PredictionError, match="introuvable"):
        classify_source("chemin/qui/nexiste/pas.jpg")


def test_classify_source_unsupported_extension(tmp_path: Path) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(PredictionError, match="non supporte"):
        classify_source(str(bad))


# --------------------------------------------------------------------------- #
# Sorties
# --------------------------------------------------------------------------- #
def _prediction() -> ImagePrediction:
    return ImagePrediction(
        source="img.jpg",
        width=200,
        height=100,
        detections=[detection("Safety Helmet", [50.0, 20.0, 90.0, 60.0], 0.87, 4)],
        timing_ms={"inference": 12.5},
    )


def test_save_predictions_json(tmp_path: Path) -> None:
    import json

    path = save_predictions_json([_prediction()], tmp_path / "out.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_sources"] == 1
    assert payload["results"][0]["detections"][0]["class_name"] == "Safety Helmet"
    assert payload["results"][0]["image"] == {"width": 200, "height": 100}


def test_save_predictions_csv(tmp_path: Path) -> None:
    path = save_predictions_csv([_prediction()], tmp_path / "out.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "Safety Helmet" in lines[1]


def test_save_yolo_txt_normalises_coordinates(tmp_path: Path) -> None:
    path = save_yolo_txt(_prediction(), tmp_path / "img.txt")
    parts = path.read_text(encoding="utf-8").split()
    assert parts[0] == "4"
    # cx = (50+90)/2 / 200 = 0.35 ; cy = (20+60)/2 / 100 = 0.40
    assert float(parts[1]) == pytest.approx(0.35)
    assert float(parts[2]) == pytest.approx(0.40)
    assert float(parts[3]) == pytest.approx(0.20)  # w = 40/200
    assert float(parts[4]) == pytest.approx(0.40)  # h = 40/100


def test_image_prediction_to_dict_includes_compliance() -> None:
    prediction = _prediction()
    prediction.compliance = [{"compliant": True, "missing_ppe": []}]
    payload = prediction.to_dict()
    assert "compliance" in payload
    assert payload["compliance_summary"]["persons_detected"] == 1


# --------------------------------------------------------------------------- #
# Rendu
# --------------------------------------------------------------------------- #
def test_color_for_class_is_deterministic() -> None:
    assert color_for_class("Safety Helmet") == color_for_class("Safety Helmet")
    assert color_for_class("Classe Inconnue") == color_for_class("Classe Inconnue")
    assert color_for_class("Safety Helmet") != color_for_class("Person")


def test_draw_detections_does_not_mutate_source() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    original = image.copy()
    annotated = draw_detections(image, [detection("Person", [10, 10, 90, 90], 0.9, 1)])
    assert np.array_equal(image, original)
    assert not np.array_equal(annotated, original)


def test_draw_yolo_labels_renders(class_names: list[str]) -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated = draw_yolo_labels(image, [(1, 0.5, 0.5, 0.4, 0.4)], class_names)
    assert annotated.shape == image.shape
    assert annotated.any()


def test_draw_detections_handles_out_of_bounds_boxes() -> None:
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    annotated = draw_detections(image, [detection("Person", [-20, -20, 500, 500], 0.9, 1)])
    assert annotated.shape == image.shape


# --------------------------------------------------------------------------- #
# Inference reelle (necessite des poids)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_detector_runs_on_real_weights(real_weights: Path, sample_image: Path) -> None:
    from ppe_detection.predict import PPEDetector

    detector = PPEDetector(InferenceConfig(weights=str(real_weights), conf=0.25, device="cpu"))
    prediction = detector.predict_image(sample_image)

    assert prediction.width == 320
    assert prediction.height == 240
    assert isinstance(prediction.detections, list)
    for item in prediction.detections:
        assert set(item) >= {"class_id", "class_name", "confidence", "bbox_xyxy"}
        assert 0.0 <= item["confidence"] <= 1.0
        assert len(item["bbox_xyxy"]) == 4

    info = detector.model_info()
    assert info["num_classes"] == 7
    assert info["parameters"] and info["parameters"] > 0


@pytest.mark.slow
def test_detector_missing_weights_raises() -> None:
    from ppe_detection.predict import PPEDetector

    with pytest.raises(PredictionError, match="introuvables"):
        PPEDetector(InferenceConfig(weights="artifacts/models/does_not_exist.pt"))
