"""Tests de la calibration des seuils par classe.

Les predictions sont synthetiques : la logique de balayage et de selection se
verifie sans modele ni dataset.
"""

from __future__ import annotations

import pytest

from ppe_detection.calibrate import ClassCalibration, calibrate_class, render_markdown
from ppe_detection.evaluate import EvaluationError


def predictions_with_noise() -> list[tuple[int, float, bool]]:
    """5 vrais positifs bien notes et 5 faux positifs mal notes.

    Le seuil optimal doit se situer entre les deux groupes.
    """
    true_positives = [(0, conf, True) for conf in (0.90, 0.85, 0.80, 0.75, 0.70)]
    false_positives = [(0, conf, False) for conf in (0.40, 0.35, 0.30, 0.20, 0.10)]
    return true_positives + false_positives


def test_threshold_separates_true_from_false_positives() -> None:
    result = calibrate_class(0, "Test", predictions_with_noise(), n_ground_truth=5, baseline=0.25)
    # Un seuil au-dessus du meilleur faux positif (0.40) et sous le pire vrai
    # positif (0.70) donne un F1 parfait.
    assert 0.40 < result.best_threshold <= 0.70
    assert result.best_f1 == pytest.approx(1.0)
    assert result.precision_at_best == pytest.approx(1.0)
    assert result.recall_at_best == pytest.approx(1.0)


def test_gain_is_measured_against_baseline() -> None:
    result = calibrate_class(0, "Test", predictions_with_noise(), n_ground_truth=5, baseline=0.25)
    # A 0.25, quatre faux positifs passent : precision 5/9, F1 < 1.
    assert result.f1_at_baseline < 1.0
    assert result.gain > 0.0


def test_min_recall_constraint_is_respected() -> None:
    """Une contrainte metier de rappel doit ecarter les seuils trop severes."""
    predictions = [
        (0, 0.95, True),
        (0, 0.90, True),
        (0, 0.30, True),
        (0, 0.28, True),
        (0, 0.92, False),
    ]
    strict = calibrate_class(
        0, "Test", predictions, n_ground_truth=4, baseline=0.25, min_recall=0.9
    )
    assert strict.recall_at_best >= 0.9

    relaxed = calibrate_class(0, "Test", predictions, n_ground_truth=4, baseline=0.25)
    # Sans contrainte, un seuil plus severe peut etre prefere.
    assert relaxed.best_threshold >= strict.best_threshold


def test_impossible_min_recall_falls_back_without_crashing() -> None:
    """Si aucun seuil n'atteint le rappel exige, on selectionne quand meme."""
    predictions = [(0, 0.9, True)]
    result = calibrate_class(
        0, "Test", predictions, n_ground_truth=100, baseline=0.25, min_recall=0.99
    )
    assert result.best_f1 >= 0.0
    assert 0.0 <= result.best_threshold <= 1.0


def test_class_without_ground_truth_is_handled() -> None:
    result = calibrate_class(3, "Absente", [], n_ground_truth=0, baseline=0.25)
    assert result.best_f1 == 0.0
    assert result.n_ground_truth == 0


def test_other_classes_do_not_pollute_the_curve() -> None:
    """Seules les predictions de la classe calibree doivent compter."""
    predictions = [(0, 0.9, True), (1, 0.9, False), (1, 0.8, False)]
    result = calibrate_class(0, "Classe0", predictions, n_ground_truth=1, baseline=0.25)
    assert result.best_f1 == pytest.approx(1.0)


def test_curve_covers_the_threshold_grid() -> None:
    result = calibrate_class(0, "Test", predictions_with_noise(), n_ground_truth=5)
    thresholds = [entry["threshold"] for entry in result.curve]
    assert len(thresholds) > 10
    assert thresholds == sorted(thresholds)
    for entry in result.curve:
        assert 0.0 <= entry["precision"] <= 1.0
        assert 0.0 <= entry["recall"] <= 1.0


def test_to_dict_is_serialisable() -> None:
    result = calibrate_class(0, "Test", predictions_with_noise(), n_ground_truth=5)
    payload = result.to_dict()
    assert payload["class_name"] == "Test"
    assert "best_threshold" in payload and "f1_gain" in payload


def test_markdown_report_contains_yaml_snippet() -> None:
    calibration = ClassCalibration(
        class_id=0,
        class_name="Safety Helmet",
        n_ground_truth=10,
        best_threshold=0.30,
        best_f1=0.8,
        precision_at_best=0.8,
        recall_at_best=0.8,
        f1_at_baseline=0.75,
        baseline_threshold=0.25,
    )
    report = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "weights": "best.pt",
        "split": "valid",
        "n_images": 100,
        "baseline_threshold": 0.25,
        "iou_threshold": 0.5,
        "classes": [calibration.to_dict()],
        "summary": {
            "macro_f1_baseline": 0.75,
            "macro_f1_calibrated": 0.80,
            "macro_f1_gain": 0.05,
        },
    }
    markdown = render_markdown(report)
    assert "Safety Helmet: 0.3" in markdown
    assert "class_conf:" in markdown
    # Le rapport doit rappeler le protocole : seuils choisis sur la validation.
    assert "validation" in markdown.lower()


def test_calibrating_on_test_split_is_refused() -> None:
    """Choisir les seuils sur le test invaliderait l'evaluation finale."""
    from ppe_detection.calibrate import calibrate

    with pytest.raises(EvaluationError, match="test"):
        calibrate("artifacts/models/best.pt", "data.yaml", split="test")
