"""Couche metier : association geometrique EPI <-> personne.

AVERTISSEMENT IMPORTANT
=======================
Le modele de detection identifie des objets **independamment** les uns des
autres. Rien dans ses sorties ne relie formellement un casque a une personne.
Ce module applique une heuristique purement geometrique : un EPI est attribue a
une personne si une fraction suffisante de sa boite se trouve dans la region
attendue de la boite de cette personne (haut du corps pour un casque, bas pour
les chaussures, torse pour le gilet et le harnais).

Trois etats, pas deux
---------------------
Un detecteur qui ne voit pas un gilet ne prouve pas son absence. Declarer
« non conforme » une personne dont on ne peut pas observer la zone concernee
produit des fausses alertes en masse — c'est le defaut le plus courant de ce
type de systeme. Ce module distingue donc :

* ``compliant``     — tous les EPI requis sont detectes et attribues ;
* ``non_compliant`` — au moins un EPI requis manque **dans une zone reellement
  observable** ;
* ``indeterminate`` — la zone concernee n'est pas observable (personne tronquee
  par le bord du cadre, ou trop petite pour que l'objet soit resoluble). Aucune
  conclusion n'est tiree.

Limites persistantes
--------------------
Meme avec ces garde-fous, l'heuristique echoue lorsque :

* les personnes se chevauchent (le casque de l'une peut etre attribue a l'autre) ;
* la camera est en plongee/contre-plongee forte, ce qui invalide l'hypothese
  « la tete est en haut de la boite » ;
* un EPI est present dans la scene sans etre porte (casque pose sur une table) ;
* le modele n'a pas detecte un EPI pourtant visible et bien resolu — le verdict
  sera alors « non conforme » a tort.

Un statut « non conforme » reste une **alerte a verifier**, jamais un constat.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import ComplianceConfig
from .utils import get_logger

LOGGER = get_logger(__name__)

Box = Sequence[float]

REGION_ANY = "any"
REGION_HEAD = "head"
REGION_TORSO = "torso"
REGION_FEET = "feet"

STATUS_COMPLIANT = "compliant"
STATUS_NON_COMPLIANT = "non_compliant"
STATUS_INDETERMINATE = "indeterminate"

HEURISTIC_NOTICE = (
    "Association geometrique EPI/personne : resultat indicatif, a verifier "
    "humainement. Un statut 'indeterminate' signifie que la zone concernee "
    "n'etait pas observable, pas que l'EPI est absent."
)


@dataclass
class RegionObservability:
    """Indique si une zone du corps peut servir de base a un verdict."""

    observable: bool
    reason: str = ""


@dataclass
class PersonCompliance:
    """Verdict de conformite pour une personne detectee."""

    person_index: int
    bbox_xyxy: list[float]
    confidence: float
    status: str
    detected_ppe: list[str]
    missing_ppe: list[str]
    indeterminate_ppe: list[str]
    matched: list[dict[str, Any]]
    verdict_confidence: float
    reasons: list[str] = field(default_factory=list)
    track_id: int | None = None

    @property
    def compliant(self) -> bool:
        """Conserve pour compatibilite : vrai uniquement si pleinement conforme."""
        return self.status == STATUS_COMPLIANT

    def to_dict(self) -> dict[str, Any]:
        """Representation serialisable (reponses API, export JSON)."""
        payload: dict[str, Any] = {
            "person_index": self.person_index,
            "bbox_xyxy": [round(float(v), 2) for v in self.bbox_xyxy],
            "confidence": round(float(self.confidence), 4),
            "status": self.status,
            "compliant": self.compliant,
            "detected_ppe": self.detected_ppe,
            "missing_ppe": self.missing_ppe,
            "indeterminate_ppe": self.indeterminate_ppe,
            "matched": self.matched,
            "verdict_confidence": round(float(self.verdict_confidence), 4),
            "heuristic": HEURISTIC_NOTICE,
        }
        if self.reasons:
            payload["reasons"] = self.reasons
        if self.track_id is not None:
            payload["track_id"] = self.track_id
        return payload


# --------------------------------------------------------------------------- #
# Geometrie
# --------------------------------------------------------------------------- #
def box_area(box: Box) -> float:
    """Aire d'une boite ``(x1, y1, x2, y2)`` (0 si degeneree)."""
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def intersection_area(first: Box, second: Box) -> float:
    """Aire d'intersection de deux boites ``(x1, y1, x2, y2)``."""
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def containment_ratio(inner: Box, outer: Box) -> float:
    """Fraction de l'aire de ``inner`` situee dans ``outer`` (0 a 1).

    Contrairement a l'IoU, cette mesure ne penalise pas la difference de taille :
    un petit casque entierement inclus dans une grande personne obtient 1.0.
    """
    area = box_area(inner)
    if area <= 0.0:
        return 0.0
    return intersection_area(inner, outer) / area


def person_region(
    person_box: Box,
    region: str,
    config: ComplianceConfig,
    pose_regions: Mapping[str, Sequence[float]] | None = None,
) -> tuple[float, float, float, float]:
    """Sous-region attendue d'un EPI a l'interieur de la boite d'une personne.

    Si des regions issues de l'estimation de pose sont fournies et couvrent la
    zone demandee, elles priment : elles donnent la position reelle de la partie
    du corps, sans supposer que la personne est debout et vue de face. Sinon on
    retombe sur le decoupage par fractions de la boite.

    Args:
        person_box: Boite de la personne ``(x1, y1, x2, y2)``.
        region: ``"head"``, ``"torso"``, ``"feet"``, ``"hands"`` ou ``"any"``.
        config: Parametres de zones.
        pose_regions: Regions issues des points cles, indexees par nom.

    Returns:
        La sous-boite correspondante.
    """
    if pose_regions and region in pose_regions:
        box = pose_regions[region]
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

    x1, y1, x2, y2 = (float(v) for v in person_box)
    height = y2 - y1
    if region == REGION_HEAD:
        return (x1, y1, x2, y1 + height * config.helmet_region)
    if region == REGION_FEET:
        return (x1, y2 - height * config.shoes_region, x2, y2)
    if region == REGION_TORSO:
        start, end = config.torso_region
        return (x1, y1 + height * start, x2, y1 + height * end)
    return (x1, y1, x2, y2)


def region_observability(
    person_box: Box,
    region: str,
    config: ComplianceConfig,
    image_size: tuple[int, int] | None,
    pose_regions: Mapping[str, Sequence[float]] | None = None,
) -> RegionObservability:
    """Determine si une zone du corps permet de conclure a l'absence d'un EPI.

    Deux causes d'inobservabilite sont testees :

    1. **Troncature** — la zone touche un bord du cadre, donc une partie du
       corps est hors champ. Typiquement une personne dont la tete depasse par
       le haut de l'image : son casque peut etre present sans etre visible.
    2. **Resolution** — la zone est trop petite en pixels pour que le detecteur
       puisse y resoudre un objet. A 10 pixels de haut, l'absence de detection
       d'un casque n'apprend rien.

    Args:
        person_box: Boite de la personne en pixels.
        region: Zone concernee.
        config: Seuils d'observabilite.
        image_size: ``(largeur, hauteur)`` de l'image; si ``None``, le test de
            troncature est ignore (dimensions inconnues).
        pose_regions: Regions issues des points cles, si disponibles.

    Returns:
        L'observabilite de la zone, avec une raison lisible si negative.
    """
    x1, y1, x2, y2 = person_region(person_box, region, config, pose_regions)
    height = y2 - y1

    if height < config.min_region_height_px:
        return RegionObservability(
            observable=False,
            reason=(
                f"zone '{region}' trop petite ({height:.0f} px < "
                f"{config.min_region_height_px:.0f} px) pour resoudre un EPI"
            ),
        )

    if image_size is not None:
        width_img, height_img = image_size
        margin = config.edge_margin_px
        truncated: list[str] = []
        if x1 <= margin:
            truncated.append("gauche")
        if y1 <= margin:
            truncated.append("haut")
        if x2 >= width_img - margin:
            truncated.append("droite")
        if y2 >= height_img - margin:
            truncated.append("bas")
        if truncated:
            return RegionObservability(
                observable=False,
                reason=f"zone '{region}' tronquee par le bord {'/'.join(truncated)} du cadre",
            )

    return RegionObservability(observable=True)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_compliance(
    detections: Iterable[Mapping[str, Any]],
    config: ComplianceConfig,
    *,
    image_size: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Associe les EPI detectes aux personnes et rend un verdict a trois etats.

    Chaque EPI est attribue a **une seule** personne : celle dont la region
    attendue contient la plus grande fraction de la boite de l'EPI. Un EPI dont
    le meilleur taux de recouvrement reste sous ``containment_threshold`` n'est
    attribue a personne.

    Pour chaque EPI requis non attribue, la zone correspondante est ensuite
    testee : si elle n'est pas observable (troncature ou resolution
    insuffisante), l'EPI est classe *indetermine* plutot que *manquant*.

    Args:
        detections: Detections brutes exposant ``class_name``, ``confidence`` et
            ``bbox_xyxy`` (format produit par :mod:`ppe_detection.predict`).
        config: Regles metier.
        image_size: ``(largeur, hauteur)`` de l'image, necessaire au test de
            troncature. Sans cette information le test est simplement ignore.

    Returns:
        Une liste de verdicts serialisables, un par personne detectee. Liste
        vide si aucune personne n'est detectee — dans ce cas aucune conclusion
        de conformite ne peut etre tiree.
    """
    items = list(detections)
    persons: list[tuple[int, Mapping[str, Any]]] = []
    ppe_items: list[tuple[int, Mapping[str, Any]]] = []

    for index, detection in enumerate(items):
        name = str(detection.get("class_name", ""))
        confidence = float(detection.get("confidence", 0.0))
        if name == config.person_class:
            if confidence >= config.min_person_conf:
                persons.append((index, detection))
        elif confidence >= config.min_ppe_conf:
            ppe_items.append((index, detection))

    if not persons:
        return []

    assignments: list[list[dict[str, Any]]] = [[] for _ in persons]

    for ppe_index, ppe in ppe_items:
        ppe_name = str(ppe.get("class_name", ""))
        ppe_box = ppe.get("bbox_xyxy")
        if ppe_box is None:
            continue
        region = config.region_by_class.get(ppe_name, REGION_ANY)

        best_person = -1
        best_ratio = 0.0
        for position, (_, person) in enumerate(persons):
            person_box = person.get("bbox_xyxy")
            if person_box is None:
                continue
            target = person_region(person_box, region, config, person.get("pose_regions"))
            ratio = containment_ratio(ppe_box, target)
            if ratio > best_ratio:
                best_ratio = ratio
                best_person = position

        if best_person >= 0 and best_ratio >= config.containment_threshold:
            assignments[best_person].append(
                {
                    "class_name": ppe_name,
                    "detection_index": ppe_index,
                    "confidence": round(float(ppe.get("confidence", 0.0)), 4),
                    "containment": round(best_ratio, 4),
                    "region": region,
                }
            )

    results: list[dict[str, Any]] = []
    for position, (person_index, person) in enumerate(persons):
        matched = assignments[position]
        detected_names = sorted({item["class_name"] for item in matched})
        person_box = person["bbox_xyxy"]

        missing: list[str] = []
        indeterminate: list[str] = []
        reasons: list[str] = []
        pose_regions = person.get("pose_regions")

        for required in config.required_ppe:
            if required in detected_names:
                continue
            region = config.region_by_class.get(required, REGION_ANY)
            observability = region_observability(
                person_box, region, config, image_size, pose_regions
            )
            if observability.observable:
                missing.append(required)
            else:
                indeterminate.append(required)
                reasons.append(f"{required} : {observability.reason}")

        if missing:
            status = STATUS_NON_COMPLIANT
        elif indeterminate:
            status = STATUS_INDETERMINATE
        else:
            status = STATUS_COMPLIANT

        person_conf = float(person.get("confidence", 0.0))
        if status == STATUS_COMPLIANT and matched:
            required_confidences = [
                item["confidence"] for item in matched if item["class_name"] in config.required_ppe
            ]
            verdict_conf = (
                min([person_conf, *required_confidences]) if required_confidences else person_conf
            )
        else:
            # Pour une non-conformite ou un indetermine, la confiance porte sur
            # la detection de la personne : l'absence d'un EPI peut resulter
            # d'un faux negatif du detecteur.
            verdict_conf = person_conf

        track_id = person.get("track_id")
        entry = PersonCompliance(
            person_index=person_index,
            bbox_xyxy=[float(v) for v in person_box],
            confidence=person_conf,
            status=status,
            detected_ppe=detected_names,
            missing_ppe=missing,
            indeterminate_ppe=indeterminate,
            matched=matched,
            verdict_confidence=verdict_conf,
            reasons=reasons,
            track_id=int(track_id) if track_id is not None else None,
        ).to_dict()
        # Tracabilite : indique si le verdict repose sur les points cles du corps
        # ou sur le decoupage par fractions, dont les hypotheses sont plus fortes.
        entry["association_method"] = "pose" if pose_regions else "bbox_fractions"
        results.append(entry)
    return results


def summarise_compliance(persons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Agrege les verdicts individuels en indicateurs de synthese.

    Le taux de conformite est calcule sur les seules personnes **jugeables** :
    inclure les indetermines au denominateur ferait baisser artificiellement le
    taux a cause de personnes qu'on n'a simplement pas pu observer.
    """
    total = len(persons)
    statuses = Counter(str(person.get("status", STATUS_NON_COMPLIANT)) for person in persons)
    compliant = statuses.get(STATUS_COMPLIANT, 0)
    non_compliant = statuses.get(STATUS_NON_COMPLIANT, 0)
    indeterminate = statuses.get(STATUS_INDETERMINATE, 0)
    decidable = compliant + non_compliant

    missing_counter: Counter[str] = Counter()
    indeterminate_counter: Counter[str] = Counter()
    for person in persons:
        missing_counter.update(str(n) for n in (person.get("missing_ppe") or []))
        indeterminate_counter.update(str(n) for n in (person.get("indeterminate_ppe") or []))

    return {
        "persons_detected": total,
        "compliant": compliant,
        "non_compliant": non_compliant,
        "indeterminate": indeterminate,
        "decidable": decidable,
        "compliance_rate": round(compliant / decidable, 4) if decidable else None,
        "missing_ppe_counts": dict(missing_counter.most_common()),
        "indeterminate_ppe_counts": dict(indeterminate_counter.most_common()),
    }


# --------------------------------------------------------------------------- #
# Lissage temporel (niveau 2)
# --------------------------------------------------------------------------- #
@dataclass
class TrackState:
    """Historique des verdicts d'une personne suivie."""

    history: deque[str] = field(default_factory=deque)
    frames_seen: int = 0
    alerted: bool = False
    last_status: str = STATUS_INDETERMINATE


class ComplianceTracker:
    """Stabilise les verdicts de conformite dans le temps.

    Sur une video, un verdict calcule image par image clignote : une personne
    passe de conforme a non conforme au gre des detections manquees. Ce lisseur
    conserve les derniers verdicts de chaque personne suivie et ne tranche que
    lorsqu'une majorite nette se degage.

    Les verdicts *indetermines* ne sont pas comptes dans la majorite — ils
    n'apportent aucune information — mais ils sont conserves dans l'historique
    pour que la fenetre reste temporelle et non « temporelle utile ».

    Requiert que les detections portent un ``track_id`` (voir
    :meth:`ppe_detection.predict.PPEDetector.track_array`). Les personnes sans
    identifiant de suivi sont renvoyees telles quelles, sans lissage.
    """

    def __init__(self, config: ComplianceConfig) -> None:
        """Initialise le lisseur a partir des parametres temporels de la config."""
        self.config = config
        self.window = config.temporal_window
        self.min_ratio = config.temporal_min_ratio
        self.min_observations = config.temporal_min_observations
        self._tracks: dict[int, TrackState] = {}

    def update(self, persons: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Enrichit chaque verdict d'un statut lisse.

        Args:
            persons: Verdicts instantanes produits par :func:`evaluate_compliance`.

        Returns:
            Les memes verdicts, chacun complete par ``smoothed_status``,
            ``observations``, ``agreement`` et ``is_new_alert``.
        """
        enriched: list[dict[str, Any]] = []
        for person in persons:
            payload = dict(person)
            track_id = payload.get("track_id")
            if track_id is None:
                payload["smoothed_status"] = payload.get("status", STATUS_INDETERMINATE)
                payload["observations"] = 0
                payload["agreement"] = None
                payload["is_new_alert"] = False
                enriched.append(payload)
                continue

            state = self._tracks.setdefault(int(track_id), TrackState(history=deque(maxlen=self.window)))
            state.history.append(str(payload.get("status", STATUS_INDETERMINATE)))
            state.frames_seen += 1

            smoothed, observations, agreement = self._smooth(state)
            state.last_status = smoothed

            is_new_alert = False
            if smoothed == STATUS_NON_COMPLIANT and not state.alerted:
                state.alerted = True
                is_new_alert = True
            elif smoothed == STATUS_COMPLIANT:
                # La personne s'est mise en conformite : on rearme l'alerte.
                state.alerted = False

            payload["smoothed_status"] = smoothed
            payload["observations"] = observations
            payload["agreement"] = agreement
            payload["is_new_alert"] = is_new_alert
            payload["frames_seen"] = state.frames_seen
            enriched.append(payload)
        return enriched

    def _smooth(self, state: TrackState) -> tuple[str, int, float | None]:
        """Calcule le statut lisse d'une personne suivie."""
        counts = Counter(state.history)
        compliant = counts.get(STATUS_COMPLIANT, 0)
        non_compliant = counts.get(STATUS_NON_COMPLIANT, 0)
        observations = compliant + non_compliant

        if observations < self.min_observations:
            return STATUS_INDETERMINATE, observations, None

        non_compliant_ratio = non_compliant / observations
        compliant_ratio = compliant / observations
        if non_compliant_ratio >= self.min_ratio:
            return STATUS_NON_COMPLIANT, observations, round(non_compliant_ratio, 4)
        if compliant_ratio >= self.min_ratio:
            return STATUS_COMPLIANT, observations, round(compliant_ratio, 4)
        return STATUS_INDETERMINATE, observations, round(max(compliant_ratio, non_compliant_ratio), 4)

    def summary(self) -> dict[str, Any]:
        """Bilan par personne suivie, une ligne par identifiant.

        C'est cette vue qui a un sens operationnel : elle compte des personnes,
        pas des detections repetees a chaque frame.
        """
        statuses = Counter(state.last_status for state in self._tracks.values())
        compliant = statuses.get(STATUS_COMPLIANT, 0)
        non_compliant = statuses.get(STATUS_NON_COMPLIANT, 0)
        decidable = compliant + non_compliant
        return {
            "tracked_persons": len(self._tracks),
            "compliant": compliant,
            "non_compliant": non_compliant,
            "indeterminate": statuses.get(STATUS_INDETERMINATE, 0),
            "decidable": decidable,
            "compliance_rate": round(compliant / decidable, 4) if decidable else None,
            # Nombre de personnes actuellement en alerte, a ne pas confondre avec
            # le nombre total d'evenements d'alerte : une personne qui se met en
            # conformite puis redevient non conforme genere deux evenements.
            "persons_currently_alerted": sum(1 for state in self._tracks.values() if state.alerted),
            "window": self.window,
            "min_ratio": self.min_ratio,
            "min_observations": self.min_observations,
        }

    def reset(self) -> None:
        """Vide l'historique (changement de source video)."""
        self._tracks.clear()
