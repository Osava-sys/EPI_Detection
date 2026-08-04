"""Tests de l'association EPI <-> personne fondee sur les points cles.

Aucun de ces tests ne charge de modele de pose : les points cles sont
synthetiques, ce qui permet de verifier la geometrie et la logique de repli
sans dependre d'un telechargement ni d'un GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppe_detection.compliance import (
    STATUS_COMPLIANT,
    STATUS_INDETERMINATE,
    STATUS_NON_COMPLIANT,
    evaluate_compliance,
    person_region,
)
from ppe_detection.config import ComplianceConfig
from ppe_detection.pose import (
    LEFT_ANKLE,
    LEFT_EAR,
    LEFT_EYE,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_EAR,
    RIGHT_EYE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    PersonPose,
    feet_region,
    hands_region,
    head_region,
    regions_from_pose,
    torso_region,
)


def make_pose(points: dict[int, tuple[float, float]], *, score: float = 0.9) -> PersonPose:
    """Construit une pose synthetique; les points absents sont a confiance nulle."""
    keypoints = np.zeros((17, 2), dtype=float)
    scores = np.zeros(17, dtype=float)
    for index, (x, y) in points.items():
        keypoints[index] = (x, y)
        scores[index] = score
    xs = [p[0] for p in points.values()]
    ys = [p[1] for p in points.values()]
    return PersonPose(
        bbox_xyxy=[min(xs), min(ys), max(xs), max(ys)],
        confidence=0.9,
        keypoints=keypoints,
        scores=scores,
    )


def standing_pose() -> PersonPose:
    """Personne debout, tete en haut : le cas ou les fractions fonctionnent."""
    return make_pose(
        {
            NOSE: (100, 50),
            LEFT_EYE: (95, 45),
            RIGHT_EYE: (105, 45),
            LEFT_EAR: (90, 48),
            RIGHT_EAR: (110, 48),
            LEFT_SHOULDER: (80, 90),
            RIGHT_SHOULDER: (120, 90),
            LEFT_HIP: (85, 190),
            RIGHT_HIP: (115, 190),
            LEFT_WRIST: (70, 160),
            RIGHT_WRIST: (130, 160),
            LEFT_ANKLE: (88, 290),
            RIGHT_ANKLE: (112, 290),
        }
    )


def crouching_pose() -> PersonPose:
    """Personne accroquie et penchee : la tete n'est PAS en haut de la boite.

    C'est le cas que le decoupage par fractions traite mal : les epaules sont
    plus hautes que la tete, qui est projetee vers l'avant et vers le bas.
    """
    return make_pose(
        {
            NOSE: (170, 150),
            LEFT_EYE: (168, 145),
            RIGHT_EYE: (175, 145),
            LEFT_EAR: (160, 148),
            RIGHT_EAR: (180, 148),
            LEFT_SHOULDER: (120, 120),
            RIGHT_SHOULDER: (140, 130),
            LEFT_HIP: (80, 200),
            RIGHT_HIP: (100, 205),
            LEFT_WRIST: (190, 210),
            RIGHT_WRIST: (200, 200),
            LEFT_ANKLE: (70, 260),
            RIGHT_ANKLE: (95, 265),
        }
    )


# --------------------------------------------------------------------------- #
# Geometrie des regions
# --------------------------------------------------------------------------- #
def test_head_region_sits_above_face_points() -> None:
    """Un casque couvre le crane : la zone doit remonter au-dessus du visage."""
    pose = standing_pose()
    region = head_region(pose, 0.5)
    assert region is not None
    x1, y1, x2, y2 = region
    assert y1 < 45, "la zone doit remonter au-dessus des yeux"
    assert x1 < 100 < x2, "la zone doit encadrer le nez"


def test_torso_region_spans_shoulders_to_hips() -> None:
    pose = standing_pose()
    region = torso_region(pose, 0.5)
    assert region is not None
    _, y1, _, y2 = region
    assert y1 <= 90 and y2 >= 190


def test_feet_region_surrounds_ankles() -> None:
    pose = standing_pose()
    region = feet_region(pose, 0.5)
    assert region is not None
    x1, y1, x2, y2 = region
    assert x1 < 88 and x2 > 112
    assert y1 < 290 < y2


def test_hands_region_surrounds_wrists() -> None:
    pose = standing_pose()
    region = hands_region(pose, 0.5)
    assert region is not None
    x1, _, x2, _ = region
    assert x1 < 70 and x2 > 130


def test_regions_absent_when_keypoints_missing() -> None:
    """Sans points cles fiables, aucune region n'est produite (repli attendu)."""
    pose = make_pose({NOSE: (100, 50)}, score=0.1)  # confiance sous le seuil
    assert regions_from_pose(pose, 0.5) == {}


def test_partial_pose_yields_partial_regions() -> None:
    """Une pose incomplete ne produit que les regions calculables."""
    pose = make_pose(
        {
            LEFT_SHOULDER: (80, 90),
            RIGHT_SHOULDER: (120, 90),
            LEFT_HIP: (85, 190),
            RIGHT_HIP: (115, 190),
        }
    )
    regions = regions_from_pose(pose, 0.5)
    assert "torso" in regions
    assert "head" not in regions  # aucun point du visage
    assert "feet" not in regions  # aucune cheville


# --------------------------------------------------------------------------- #
# Le cas qui justifie le module
# --------------------------------------------------------------------------- #
def test_pose_locates_head_where_fractions_fail() -> None:
    """Sur une personne accroupie, les fractions cherchent la tete au mauvais endroit.

    La boite englobe [70, 120] a [200, 265]. Le decoupage par fractions place la
    zone « tete » dans les 35 % superieurs, soit y <= 171. Or la tete reelle est
    a y ~ 150 mais surtout tres a droite (x ~ 170), alors que la zone par
    fractions couvre toute la largeur, y compris les hanches a gauche.
    """
    pose = crouching_pose()
    config = ComplianceConfig()
    person_box = pose.bbox_xyxy

    fraction_region = person_region(person_box, "head", config)
    pose_region = person_region(person_box, "head", config, regions_from_pose(pose, 0.5))

    assert fraction_region != pose_region
    # La zone issue de la pose est centree sur le visage reel...
    px1, _, px2, _ = pose_region
    assert px1 <= 170 <= px2
    # ...et bien plus etroite que la pleine largeur du corps.
    assert (px2 - px1) < (fraction_region[2] - fraction_region[0])


def test_helmet_attributed_correctly_on_crouching_person() -> None:
    """Un casque pose sur la tete reelle doit etre attribue grace a la pose."""
    pose = crouching_pose()
    regions = regions_from_pose(pose, 0.5)
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"])

    person = {
        "class_name": "Person",
        "confidence": 0.9,
        "bbox_xyxy": pose.bbox_xyxy,
        "pose_regions": regions,
    }
    # Casque autour du visage reel (x ~ 170, y ~ 140), pas en haut de la boite.
    helmet = {
        "class_name": "Safety Helmet",
        "confidence": 0.8,
        "bbox_xyxy": [158, 128, 186, 152],
    }
    result = evaluate_compliance([person, helmet], config, image_size=(640, 480))[0]
    assert result["status"] == STATUS_COMPLIANT
    assert result["association_method"] == "pose"


# --------------------------------------------------------------------------- #
# Repli et tracabilite
# --------------------------------------------------------------------------- #
def test_falls_back_to_fractions_without_pose() -> None:
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"])
    person = {"class_name": "Person", "confidence": 0.9, "bbox_xyxy": [100, 50, 300, 400]}
    result = evaluate_compliance([person], config, image_size=(640, 480))[0]
    assert result["association_method"] == "bbox_fractions"
    assert result["status"] == STATUS_NON_COMPLIANT


def test_association_method_is_reported() -> None:
    """Le rapport doit indiquer sur quelle methode repose le verdict."""
    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"])
    pose = standing_pose()
    person = {
        "class_name": "Person",
        "confidence": 0.9,
        "bbox_xyxy": pose.bbox_xyxy,
        "pose_regions": regions_from_pose(pose, 0.5),
    }
    result = evaluate_compliance([person], config, image_size=(640, 480))[0]
    assert result["association_method"] == "pose"


def test_missing_head_keypoints_keep_helmet_indeterminate() -> None:
    """Personne de dos : sans visage detecte, aucune conclusion sur le casque.

    La zone tete n'est pas calculable, donc le systeme retombe sur les
    fractions; la personne touchant le bord haut du cadre, le verdict reste
    indetermine plutot que faussement accusateur.
    """
    pose = make_pose(
        {
            LEFT_SHOULDER: (80, 10),
            RIGHT_SHOULDER: (120, 10),
            LEFT_HIP: (85, 190),
            RIGHT_HIP: (115, 190),
        }
    )
    regions = regions_from_pose(pose, 0.5)
    assert "head" not in regions

    config = ComplianceConfig(enabled=True, required_ppe=["Safety Helmet"])
    person = {
        "class_name": "Person",
        "confidence": 0.9,
        "bbox_xyxy": [80, 0, 120, 190],
        "pose_regions": regions,
    }
    result = evaluate_compliance([person], config, image_size=(640, 480))[0]
    assert result["status"] == STATUS_INDETERMINATE
    assert result["indeterminate_ppe"] == ["Safety Helmet"]


@pytest.mark.parametrize("region_name", ["head", "torso", "feet", "hands"])
def test_all_regions_are_valid_boxes(region_name: str) -> None:
    """Toute region produite doit etre une boite non degeneree."""
    regions = regions_from_pose(standing_pose(), 0.5)
    assert region_name in regions
    x1, y1, x2, y2 = regions[region_name]
    assert x2 > x1 and y2 > y1
