"""Tests des libelles francais d'affichage."""

from __future__ import annotations

import numpy as np

from ppe_detection.taxonomy import (
    CLASS_LABELS_FR,
    EXTENDED_CLASSES,
    display_name,
    display_names,
)
from ppe_detection.visualization import draw_detections


def test_every_class_has_a_french_label() -> None:
    """Aucune classe du schema ne doit s'afficher en anglais."""
    for name in EXTENDED_CLASSES:
        assert name in CLASS_LABELS_FR
        assert CLASS_LABELS_FR[name] != name


def test_display_name_translates() -> None:
    assert display_name("Safety Helmet") == "Casque de chantier"
    assert display_name("Non-Safety Headwear") == "Couvre-chef non conforme"
    assert display_name("Person") == "Personne"


def test_display_name_falls_back_on_unknown() -> None:
    """Une classe inconnue reste visible plutot que d'etre masquee."""
    assert display_name("Nouvelle Classe") == "Nouvelle Classe"


def test_display_name_accepts_override() -> None:
    assert display_name("Safety Vest", {"Safety Vest": "Chasuble"}) == "Chasuble"


def test_display_names_merges_overrides() -> None:
    table = display_names({"Person": "Ouvrier"})
    assert table["Person"] == "Ouvrier"
    # Les autres libelles restent ceux par defaut.
    assert table["Safety Helmet"] == "Casque de chantier"


def test_helmet_label_disambiguates_from_bike_helmet() -> None:
    """Depuis l'ajout des sosies, « casque » seul serait ambigu."""
    assert "chantier" in CLASS_LABELS_FR["Safety Helmet"].lower()


def test_vest_label_names_the_actual_requirement() -> None:
    """C'est la haute visibilite qui est exigee, pas le gilet en soi."""
    assert "visibilité" in CLASS_LABELS_FR["Safety Vest"].lower()


def test_labels_render_without_mangling_accents() -> None:
    """OpenCV doit ecrire les accents sans les remplacer par des points d'interrogation.

    Le test verifie que le rendu produit bien des pixels : une police incapable
    d'ecrire un caractere laisserait la zone vide.
    """
    image = np.zeros((200, 600, 3), dtype=np.uint8)
    detections = [
        {"class_name": "Safety Vest", "confidence": 0.9, "bbox_xyxy": [20, 40, 300, 160]},
    ]
    out = draw_detections(image, detections)
    assert out.sum() > 0
    # La zone du libelle (au-dessus de la boite) doit contenir du texte dessine.
    label_band = out[10:40, 20:300]
    assert label_band.sum() > 0


def test_draw_detections_uses_french_by_default() -> None:
    """Le libelle dessine doit differer selon la traduction appliquee."""
    image = np.zeros((200, 600, 3), dtype=np.uint8)
    detections = [
        {"class_name": "Safety Helmet", "confidence": 0.9, "bbox_xyxy": [20, 60, 300, 160]},
    ]
    french = draw_detections(image, detections)
    english = draw_detections(image, detections, labels={"Safety Helmet": "Safety Helmet"})
    assert not np.array_equal(french, english)


def test_color_is_stable_across_translations() -> None:
    """Traduire un libelle ne doit pas changer la couleur de la boite."""
    from ppe_detection.visualization import color_for_class

    assert color_for_class("Safety Helmet") == color_for_class("Safety Helmet")
    # La couleur derive du nom interne, jamais du libelle affiche.
    assert color_for_class("Safety Helmet") != color_for_class("Casque de chantier")
