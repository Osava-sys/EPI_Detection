"""Association EPI <-> personne fondee sur les points cles du corps.

Pourquoi ce module
------------------
L'heuristique geometrique de :mod:`ppe_detection.compliance` decoupe la boite
d'une personne en tranches fixes : les 35 % superieurs sont « la tete », les
30 % inferieurs « les pieds ». Cette regle suppose que la personne est **debout
et vue de face**. Elle se trompe des que :

* la personne est accroupie, penchee ou assise ;
* la camera est en plongee (cas courant en videosurveillance) ;
* la personne est couchee.

Les points cles du corps donnent la position reelle de la tete, du torse et des
pieds, quelle que soit la posture. Ce module les convertit en regions
utilisables directement par la couche de conformite.

Modele et points cles
---------------------
Le modele de pose (``yolo26n-pose.pt`` par defaut) predit les **17 points cles
COCO**. Il est totalement independant du detecteur d'EPI : il ne connait que
les personnes, et sert uniquement a localiser leurs parties du corps.

Limites
-------
* Le cout d'inference est double (deux modeles au lieu d'un).
* Les points cles sont eux aussi predits : une pose erronee produit des regions
  erronees. Chaque region n'est donc produite que si les points necessaires
  depassent un seuil de confiance.
* Un casque **masque le crane** : le modele s'appuie sur le visage (nez, yeux,
  oreilles) et sur l'echelle du buste pour extrapoler la zone du casque, qui se
  trouve au-dessus des points du visage.
* Quand la pose est indisponible ou incomplete, le systeme revient
  automatiquement au decoupage par fractions : la degradation est douce.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import get_logger, resolve_path

LOGGER = get_logger(__name__)

# Indices des 17 points cles COCO.
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

FACE_POINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)
SHOULDERS = (LEFT_SHOULDER, RIGHT_SHOULDER)
HIPS = (LEFT_HIP, RIGHT_HIP)
ANKLES = (LEFT_ANKLE, RIGHT_ANKLE)
WRISTS = (LEFT_WRIST, RIGHT_WRIST)

DEFAULT_POSE_MODEL = "yolo26n-pose.pt"


class PoseError(RuntimeError):
    """Erreur de chargement ou d'execution du modele de pose."""


@dataclass
class PersonPose:
    """Points cles d'une personne, en pixels."""

    bbox_xyxy: list[float]
    confidence: float
    keypoints: np.ndarray  # (17, 2) en pixels
    scores: np.ndarray  # (17,) confiance par point

    def visible(self, indices: Sequence[int], min_score: float) -> list[int]:
        """Indices des points demandes dont la confiance depasse le seuil."""
        return [i for i in indices if i < len(self.scores) and self.scores[i] >= min_score]


def _bounds(points: np.ndarray) -> tuple[float, float, float, float]:
    """Boite englobante d'un nuage de points ``(n, 2)``."""
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def _body_scale(pose: PersonPose, min_score: float) -> float:
    """Echelle du corps en pixels, servant de reference aux marges.

    Mesuree comme la distance entre le milieu des epaules et le milieu des
    hanches (longueur du buste), qui varie peu avec la posture. A defaut, on
    retombe sur la diagonale de la boite de la personne.
    """
    shoulders = pose.visible(SHOULDERS, min_score)
    hips = pose.visible(HIPS, min_score)
    if shoulders and hips:
        shoulder_mid = pose.keypoints[shoulders].mean(axis=0)
        hip_mid = pose.keypoints[hips].mean(axis=0)
        scale = float(np.linalg.norm(shoulder_mid - hip_mid))
        if scale > 1.0:
            return scale
    x1, y1, x2, y2 = pose.bbox_xyxy
    return max(float(y2 - y1) * 0.3, 1.0)


def head_region(pose: PersonPose, min_score: float) -> tuple[float, float, float, float] | None:
    """Zone ou se trouverait un casque, deduite du visage et de l'echelle du buste.

    Un casque couvre le crane, donc une zone situee **au-dessus** des points du
    visage. La region est centree sur le visage puis etendue vers le haut d'une
    hauteur de tete estimee, sans hypothese sur l'orientation de la personne
    dans l'image.
    """
    face = pose.visible(FACE_POINTS, min_score)
    if not face:
        return None
    points = pose.keypoints[face]
    cx = float(points[:, 0].mean())
    cy = float(points[:, 1].mean())

    scale = _body_scale(pose, min_score)
    # Le buste vaut environ 2,5 tetes : on en deduit la taille de la tete.
    head = max(scale / 2.5, 8.0)

    half_width = head * 0.95
    # Vers le haut : le crane et le casque. Vers le bas : le menton.
    return (cx - half_width, cy - head * 1.35, cx + half_width, cy + head * 0.55)


def torso_region(pose: PersonPose, min_score: float) -> tuple[float, float, float, float] | None:
    """Zone du gilet ou du harnais : des epaules aux hanches."""
    shoulders = pose.visible(SHOULDERS, min_score)
    hips = pose.visible(HIPS, min_score)
    if not shoulders or not hips:
        return None
    points = pose.keypoints[shoulders + hips]
    x1, y1, x2, y2 = _bounds(points)
    # Elargissement lateral : le gilet deborde de la ligne des epaules.
    margin = max((x2 - x1) * 0.25, _body_scale(pose, min_score) * 0.15)
    return (x1 - margin, y1 - margin * 0.5, x2 + margin, y2 + margin * 0.5)


def feet_region(pose: PersonPose, min_score: float) -> tuple[float, float, float, float] | None:
    """Zone des chaussures, autour des chevilles."""
    ankles = pose.visible(ANKLES, min_score)
    if not ankles:
        return None
    points = pose.keypoints[ankles]
    x1, y1, x2, y2 = _bounds(points)
    scale = _body_scale(pose, min_score)
    margin = max(scale * 0.35, 10.0)
    # La chaussure s'etend surtout vers le bas et vers l'avant du pied.
    return (x1 - margin, y1 - margin * 0.5, x2 + margin, y2 + margin)


def hands_region(pose: PersonPose, min_score: float) -> tuple[float, float, float, float] | None:
    """Zone des gants, autour des poignets."""
    wrists = pose.visible(WRISTS, min_score)
    if not wrists:
        return None
    points = pose.keypoints[wrists]
    x1, y1, x2, y2 = _bounds(points)
    scale = _body_scale(pose, min_score)
    margin = max(scale * 0.30, 8.0)
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def regions_from_pose(pose: PersonPose, min_score: float = 0.5) -> dict[str, list[float]]:
    """Construit toutes les regions exploitables a partir d'une pose.

    Args:
        pose: Points cles d'une personne.
        min_score: Confiance minimale d'un point cle pour etre utilise.

    Returns:
        Les regions disponibles, indexees par nom (``head``, ``torso``,
        ``feet``, ``hands``). Une region absente signifie que les points
        necessaires manquaient : la couche de conformite retombera alors sur le
        decoupage par fractions.
    """
    builders = {
        "head": head_region,
        "torso": torso_region,
        "feet": feet_region,
        "hands": hands_region,
    }
    regions: dict[str, list[float]] = {}
    for name, builder in builders.items():
        box = builder(pose, min_score)
        if box is not None:
            regions[name] = [float(v) for v in box]
    return regions


class PoseEstimator:
    """Estimateur de pose charge une seule fois et reutilise."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_POSE_MODEL,
        *,
        device: str = "auto",
        conf: float = 0.25,
        imgsz: int = 640,
        min_keypoint_score: float = 0.5,
    ) -> None:
        """Charge le modele de pose.

        Args:
            weights: Poids du modele de pose.
            device: ``auto``, ``cpu``, ``cuda`` ou un index.
            conf: Confiance minimale de detection d'une personne.
            imgsz: Taille d'inference.
            min_keypoint_score: Confiance minimale d'un point cle.

        Raises:
            PoseError: Si le modele ne peut pas etre charge.
        """
        from .predict import resolve_device

        self.conf = conf
        self.imgsz = imgsz
        self.min_keypoint_score = min_keypoint_score
        self.device = resolve_device(device)

        try:
            from ultralytics import YOLO

            candidate = Path(str(weights))
            target = str(resolve_path(candidate) if candidate.exists() else weights)
            self.model = YOLO(target)
        except Exception as exc:  # noqa: BLE001 - Ultralytics leve des types varies
            raise PoseError(
                f"Chargement du modele de pose impossible ({weights}) : {exc}\n"
                f"Verifiez le nom du modele (par exemple {DEFAULT_POSE_MODEL}) et "
                f"la connexion reseau si le telechargement est necessaire."
            ) from exc
        LOGGER.info("Modele de pose charge : %s (device=%s)", weights, self.device)

    def estimate(self, image: np.ndarray) -> list[PersonPose]:
        """Estime la pose de chaque personne d'une image BGR."""
        try:
            results = list(
                self.model.predict(
                    source=image,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise PoseError(f"Inference de pose impossible : {exc}") from exc

        if not results:
            return []
        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        boxes = getattr(result, "boxes", None)
        if keypoints is None or boxes is None or len(boxes) == 0:
            return []

        xy = keypoints.xy.cpu().numpy()
        raw_scores = getattr(keypoints, "conf", None)
        scores = (
            raw_scores.cpu().numpy()
            if raw_scores is not None
            else np.ones(xy.shape[:2], dtype=float)
        )
        bboxes = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        return [
            PersonPose(
                bbox_xyxy=[float(v) for v in bboxes[i]],
                confidence=float(confidences[i]),
                keypoints=xy[i],
                scores=scores[i],
            )
            for i in range(len(bboxes))
        ]

    def annotate_detections(
        self, image: np.ndarray, detections: Sequence[dict[str, Any]], person_class: str = "Person"
    ) -> int:
        """Attache les regions issues de la pose aux detections de personnes.

        Chaque pose est appariee a la detection « personne » dont la boite
        recouvre le mieux la sienne. Les detections modifiees recoivent une cle
        ``pose_regions``; celles restees sans pose conservent le comportement
        par fractions.

        Args:
            image: Image BGR analysee.
            detections: Detections a enrichir, modifiees sur place.
            person_class: Nom de la classe « personne ».

        Returns:
            Le nombre de personnes effectivement enrichies.
        """
        from .compliance import containment_ratio, intersection_area

        persons = [d for d in detections if d.get("class_name") == person_class]
        if not persons:
            return 0

        poses = self.estimate(image)
        if not poses:
            return 0

        used: set[int] = set()
        enriched = 0
        for person in persons:
            person_box = person.get("bbox_xyxy")
            if person_box is None:
                continue
            best_index, best_score = -1, 0.0
            for index, pose in enumerate(poses):
                if index in used:
                    continue
                # Recouvrement symetrique : evite qu'une pose minuscule incluse
                # dans une grande boite obtienne un score parfait.
                inter = intersection_area(pose.bbox_xyxy, person_box)
                if inter <= 0:
                    continue
                score = min(
                    containment_ratio(pose.bbox_xyxy, person_box),
                    containment_ratio(person_box, pose.bbox_xyxy),
                )
                if score > best_score:
                    best_score, best_index = score, index
            if best_index >= 0 and best_score >= 0.5:
                used.add(best_index)
                regions = regions_from_pose(poses[best_index], self.min_keypoint_score)
                if regions:
                    person["pose_regions"] = regions
                    enriched += 1
        return enriched
