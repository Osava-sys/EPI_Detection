"""Tests du schema etendu et de la logique de contre-preuve."""

from __future__ import annotations

import pytest

from ppe_detection.compliance import (
    STATUS_COMPLIANT,
    STATUS_INDETERMINATE,
    STATUS_NON_COMPLIANT,
    evaluate_compliance,
)
from ppe_detection.config import ComplianceConfig
from ppe_detection.taxonomy import (
    BASE_CLASSES,
    COUNTER_EVIDENCE,
    EXTENDED_CLASSES,
    FEASIBILITY,
    NEGATIVE_CLASSES,
    ClassSchema,
    base_schema,
    build_class_mapping,
    describe_feasibility,
    extended_schema,
)


def detection(name: str, box: list[float], conf: float = 0.9) -> dict:
    """Detection minimale au format produit par predict."""
    return {"class_name": name, "confidence": conf, "bbox_xyxy": box}


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_extended_schema_preserves_base_ids() -> None:
    """Retro-compatibilite : les 7 classes d'origine gardent leurs identifiants."""
    base, extended = base_schema(), extended_schema()
    for name in BASE_CLASSES:
        assert base.id_of(name) == extended.id_of(name)


def test_extended_schema_appends_negatives_after_base() -> None:
    """Les classes negatives se numerotent APRES les classes d'origine.

    C'est ce qui rend un dataset etendu retro-compatible : les identifiants 0-6
    designent les memes classes qu'avant.
    """
    schema = extended_schema()
    assert schema.size == len(BASE_CLASSES) + len(NEGATIVE_CLASSES)
    assert schema.names[: len(BASE_CLASSES)] == list(BASE_CLASSES)
    assert schema.names[len(BASE_CLASSES) :] == list(NEGATIVE_CLASSES)
    # Non-Safety Headwear reste l'identifiant 7 : les poids deja entraines
    # restent utilisables.
    assert schema.id_of("Non-Safety Headwear") == 7


def test_schema_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="dupliques"):
        ClassSchema(names=["Person", "Person"])


def test_unknown_class_error_lists_available() -> None:
    """Le message d'erreur doit permettre de corriger sans lire le code."""
    with pytest.raises(KeyError) as excinfo:
        base_schema().id_of("Hard Hat")
    assert "Safety Helmet" in str(excinfo.value)


def test_name_of_rejects_out_of_range() -> None:
    with pytest.raises(KeyError, match="hors bornes"):
        base_schema().name_of(99)


def test_extended_is_idempotent() -> None:
    once = base_schema().extended()
    twice = once.extended()
    assert once.names == twice.names


def test_is_negative() -> None:
    schema = extended_schema()
    assert schema.is_negative("Non-Safety Headwear")
    assert not schema.is_negative("Safety Helmet")


def test_to_data_yaml() -> None:
    payload = extended_schema().to_data_yaml()
    assert payload["nc"] == len(payload["names"])
    assert payload["names"][7] == "Non-Safety Headwear"


def test_annotation_conventions_are_declared() -> None:
    """Fusionner des sources aux conventions differentes cree des contradictions.

    ``Non-Safety Headwear`` encadre l'objet porte, ``Uncovered Head`` encadre la
    tete : sur une casquette, les deux boites seraient quasi identiques avec des
    etiquettes opposees. La convention doit donc etre explicite.
    """
    from ppe_detection.taxonomy import ANNOTATION_CONVENTIONS

    assert ANNOTATION_CONVENTIONS["Non-Safety Headwear"] == "object"
    assert ANNOTATION_CONVENTIONS["Uncovered Head"] == "region"


def test_uncovered_head_counters_the_helmet_requirement() -> None:
    """Une tete nue prouve l'absence de casque, comme un couvre-chef non conforme."""
    assert "Uncovered Head" in COUNTER_EVIDENCE["Safety Helmet"]
    assert "Non-Safety Headwear" in COUNTER_EVIDENCE["Safety Helmet"]


# --------------------------------------------------------------------------- #
# Remappage de classes
# --------------------------------------------------------------------------- #
def test_build_class_mapping_translates_ids() -> None:
    """Cas reel : importer un dataset de casques de velo."""
    source = ["With Helmet", "Without Helmet"]
    target = extended_schema()
    mapping = build_class_mapping(source, target, {"With Helmet": "Non-Safety Headwear"})
    assert mapping == {0: target.id_of("Non-Safety Headwear")}
    # 'Without Helmet' n'a pas d'equivalent : exclu.
    assert 1 not in mapping


def test_build_class_mapping_skips_empty_targets() -> None:
    mapping = build_class_mapping(
        ["helmet", "bicycle"], extended_schema(), {"helmet": "Non-Safety Headwear", "bicycle": ""}
    )
    assert len(mapping) == 1


def test_build_class_mapping_rejects_unknown_target() -> None:
    with pytest.raises(KeyError):
        build_class_mapping(["x"], extended_schema(), {"x": "Classe Inexistante"})


# --------------------------------------------------------------------------- #
# Faisabilite documentee
# --------------------------------------------------------------------------- #
def test_every_negative_class_has_feasibility() -> None:
    for name in NEGATIVE_CLASSES:
        assert name in FEASIBILITY


def test_footwear_is_flagged_unfeasible() -> None:
    """La distinction chaussure de securite / ordinaire n'est pas visuelle."""
    info = FEASIBILITY["Non-Safety Footwear"]
    assert info.recommended is False
    assert info.level == "low"


def test_describe_feasibility_mentions_recommendation() -> None:
    text = describe_feasibility()
    assert "DECONSEILLEE" in text
    assert "Non-Safety Headwear" in text


# --------------------------------------------------------------------------- #
# Contre-preuve
# --------------------------------------------------------------------------- #
def _config_with_counter_evidence() -> ComplianceConfig:
    return ComplianceConfig(
        enabled=True,
        required_ppe=["Safety Helmet"],
        counter_evidence={"Safety Helmet": ["Non-Safety Headwear"]},
        region_by_class={
            "Safety Helmet": "head",
            "Non-Safety Headwear": "head",
        },
    )


def test_counter_evidence_produces_observed_violation() -> None:
    """Une casquette detectee prouve l'absence de casque."""
    config = _config_with_counter_evidence()
    detections = [
        detection("Person", [100, 50, 300, 450], 0.95),
        detection("Non-Safety Headwear", [160, 60, 240, 120], 0.88),
    ]
    person = evaluate_compliance(detections, config, image_size=(640, 640))[0]

    assert person["status"] == STATUS_NON_COMPLIANT
    assert person["missing_ppe"] == ["Safety Helmet"]
    assert person["evidence"] == "observed"
    assert person["violations"][0]["worn_instead"] == "Non-Safety Headwear"
    assert "a la place" in person["reasons"][0]


def test_absence_without_counter_evidence_is_flagged_as_absence() -> None:
    """Sans sosie detecte, le verdict reste fonde sur une absence."""
    config = _config_with_counter_evidence()
    detections = [detection("Person", [100, 50, 300, 450], 0.95)]
    person = evaluate_compliance(detections, config, image_size=(640, 640))[0]

    assert person["status"] == STATUS_NON_COMPLIANT
    assert person["evidence"] == "absence"
    assert not person.get("violations")


def test_counter_evidence_overrides_indeterminate() -> None:
    """Voir l'objet prouve que la zone est observable, malgre la troncature."""
    config = _config_with_counter_evidence()
    # Personne tronquee par le haut : sans contre-preuve, ce serait indetermine.
    detections = [
        detection("Person", [100, 0, 300, 450], 0.95),
        detection("Non-Safety Headwear", [160, 5, 240, 70], 0.83),
    ]
    person = evaluate_compliance(detections, config, image_size=(640, 640))[0]

    assert person["status"] == STATUS_NON_COMPLIANT
    assert person["indeterminate_ppe"] == []
    assert person["evidence"] == "observed"


def test_truncated_without_counter_evidence_stays_indeterminate() -> None:
    """Le comportement du niveau 1 reste inchange en l'absence de sosie."""
    config = _config_with_counter_evidence()
    detections = [detection("Person", [100, 0, 300, 450], 0.95)]
    person = evaluate_compliance(detections, config, image_size=(640, 640))[0]
    assert person["status"] == STATUS_INDETERMINATE


def test_real_helmet_still_wins_over_counter_evidence() -> None:
    """Un vrai casque detecte rend la personne conforme."""
    config = _config_with_counter_evidence()
    detections = [
        detection("Person", [100, 50, 300, 450], 0.95),
        detection("Safety Helmet", [160, 60, 240, 120], 0.91),
    ]
    person = evaluate_compliance(detections, config, image_size=(640, 640))[0]
    assert person["status"] == STATUS_COMPLIANT
    assert not person.get("violations")


def test_counter_evidence_confidence_uses_weakest_link() -> None:
    config = _config_with_counter_evidence()
    detections = [
        detection("Person", [100, 50, 300, 450], 0.95),
        detection("Non-Safety Headwear", [160, 60, 240, 120], 0.62),
    ]
    person = evaluate_compliance(detections, config, image_size=(640, 640))[0]
    assert person["verdict_confidence"] == pytest.approx(0.62, abs=1e-3)


def test_counter_evidence_ignored_when_not_configured() -> None:
    """Sans configuration, le mecanisme reste inerte (retro-compatibilite)."""
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"])
    assert config.counter_evidence == {}
    detections = [
        detection("Person", [100, 50, 300, 450], 0.95),
        detection("Non-Safety Headwear", [160, 60, 240, 120], 0.88),
    ]
    person = evaluate_compliance(detections, config, image_size=(640, 640))[0]
    assert person["evidence"] == "absence"


def test_counter_evidence_table_covers_configurable_ppe() -> None:
    """Les correspondances documentees pointent vers des classes existantes."""
    schema = extended_schema()
    for required, substitutes in COUNTER_EVIDENCE.items():
        assert required in EXTENDED_CLASSES
        for name in substitutes:
            assert schema.id_of(name) >= 7
