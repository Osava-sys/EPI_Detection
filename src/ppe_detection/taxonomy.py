"""Schemas de classes et taxonomie etendue aux EPI « sosies ».

Pourquoi une taxonomie etendue
==============================
Le schema d'origine ne comporte que des classes **positives** : sept EPI, aucune
classe pour ce qui leur ressemble sans en etre. Un detecteur entraine ainsi
n'apprend pas « casque de chantier », il apprend « coque rigide bombee sur une
tete ». Un casque de velo correspond a cette description et se retrouve donc
classe `Safety Helmet` — verifie sur le modele actuel, avec 0.84 de confiance.

Aucune quantite de casques de chantier supplementaires ne corrige cela : le
modele n'a pas de sortie lui permettant d'exprimer « objet ressemblant a un
casque mais non conforme ». Il faut lui en donner une.

Le mecanisme de contre-preuve
=============================
Une fois ces classes negatives disponibles, la logique de conformite gagne un
niveau de certitude. Il existe trois situations, et non deux :

* **Absence de detection** — on ne voit rien dans la zone. C'est une absence de
  preuve, qui peut venir d'un faux negatif du detecteur.
* **Contre-preuve** — on voit explicitement un couvre-chef non conforme. C'est
  une **preuve positive de violation**, bien plus fiable qu'une absence, et elle
  leve toute ambiguite sur l'observabilite de la zone : si on distingue l'objet,
  c'est que la zone est visible.
* **Detection conforme** — l'EPI attendu est present.

Ce module ne definit que les schemas et leurs relations ; la logique vit dans
:mod:`ppe_detection.compliance`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Schema d'origine
# --------------------------------------------------------------------------- #
BASE_CLASSES: tuple[str, ...] = (
    "Face Mask",
    "Person",
    "Safety Gloves",
    "Safety Harness",
    "Safety Helmet",
    "Safety Shoes",
    "Safety Vest",
)
"""Les sept classes de l'export Roboflow, dans l'ordre des identifiants."""

# --------------------------------------------------------------------------- #
# Extension : classes negatives
# --------------------------------------------------------------------------- #
NEGATIVE_CLASSES: tuple[str, ...] = (
    "Non-Safety Headwear",
    # 'Uncovered Head' precede les deux classes suivantes parce qu'elle est la
    # seule a disposer de donnees. Les classes vides placees avant elle
    # occuperaient des identifiants et feraient entrainer des sorties mortes.
    "Uncovered Head",
    "Non-Safety Vest",
    "Non-Safety Footwear",
)
"""Classes « sosies » : objets portes a la place d'un EPI conforme.

Les identifiants continuent la numerotation de :data:`BASE_CLASSES` (7, 8, 9),
ce qui rend un dataset etendu retro-compatible : un modele entraine sur le
schema etendu reste utilisable avec les seuils et regles definis pour les sept
classes d'origine.
"""

EXTENDED_CLASSES: tuple[str, ...] = BASE_CLASSES + NEGATIVE_CLASSES
"""Schema complet a dix classes."""

# --------------------------------------------------------------------------- #
# Relations entre classes
# --------------------------------------------------------------------------- #
COUNTER_EVIDENCE: dict[str, tuple[str, ...]] = {
    "Safety Helmet": ("Non-Safety Headwear", "Uncovered Head"),
    "Safety Vest": ("Non-Safety Vest",),
    "Safety Shoes": ("Non-Safety Footwear",),
}
"""EPI requis -> classes dont la presence prouve qu'il n'est pas porte.

Detecter un `Non-Safety Headwear` sur une personne etablit qu'elle porte quelque
chose sur la tete qui n'est pas un casque de chantier. C'est une violation
constatee, pas une simple absence.
"""

NEGATIVE_REGIONS: dict[str, str] = {
    "Non-Safety Headwear": "head",
    "Non-Safety Vest": "torso",
    "Non-Safety Footwear": "feet",
    "Uncovered Head": "head",
}
"""Zone du corps ou chaque classe negative est attendue."""

ANNOTATION_CONVENTIONS: dict[str, str] = {
    "Non-Safety Headwear": "object",
    "Uncovered Head": "region",
}
"""Ce que delimite la boite : l'objet porte, ou la region du corps.

Distinction cruciale au moment de fusionner des sources. ``Non-Safety Headwear``
encadre **l'objet** (un casque de velo, un chapeau) ; ``Uncovered Head`` encadre
**la tete** lorsqu'aucun casque de chantier ne la protege.

Melanger les deux conventions sur une meme scene donnerait au modele des
etiquettes contradictoires : une tete coiffee d'une casquette serait annotee
``Uncovered Head`` par une source et ``Non-Safety Headwear`` par une autre, sur
une boite quasi identique. Les datasets doivent donc etre choisis pour que
chaque convention couvre un domaine visuel distinct.
"""


# --------------------------------------------------------------------------- #
# Libelles d'affichage
# --------------------------------------------------------------------------- #
CLASS_LABELS_FR: dict[str, str] = {
    "Face Mask": "Masque",
    "Person": "Personne",
    "Safety Gloves": "Gants de sécurité",
    "Safety Harness": "Harnais de sécurité",
    "Safety Helmet": "Casque de chantier",
    "Safety Shoes": "Chaussures de sécurité",
    "Safety Vest": "Gilet haute visibilité",
    "Non-Safety Headwear": "Couvre-chef non conforme",
    "Non-Safety Vest": "Vêtement non conforme",
    "Non-Safety Footwear": "Chaussures non conformes",
    "Uncovered Head": "Tête sans casque",
}
"""Libelles francais affiches a l'ecran.

Les noms **anglais restent les identifiants internes** : ce sont eux que
contiennent les poids du modele, les cles de ``configs/inference.yaml``
(``required_ppe``, ``region_by_class``, ``counter_evidence``...) et le champ
``class_name`` des exports JSON. Traduire ces identifiants casserait les
configurations et rendrait les exports incomparables d'une version a l'autre.

Seul l'affichage est traduit : etiquettes dessinees sur les images, tableaux de
l'interface, resumes lisibles. Les detections exposent en plus un champ
``class_name_fr`` pour que l'API et les exports restent exploitables sans avoir
a rejouer cette table.

Deux choix de vocabulaire meritent explication :

* « Casque de chantier » plutot que « Casque de sécurité » : depuis l'ajout de
  ``Non-Safety Headwear``, un casque de vélo est aussi un casque. Preciser
  « de chantier » leve l'ambiguite a l'ecran.
* « Gilet haute visibilité » plutot que « Gilet de sécurité » : c'est la
  haute visibilite qui constitue l'exigence reglementaire, pas le gilet en soi.
"""


def display_name(name: str, labels: Mapping[str, str] | None = None) -> str:
    """Retourne le libelle affichable d'une classe.

    Args:
        name: Identifiant interne de la classe (anglais).
        labels: Table de correspondance, par exemple issue de la configuration.
            Par defaut :data:`CLASS_LABELS_FR`.

    Returns:
        Le libelle traduit, ou le nom d'origine si aucune traduction n'existe —
        une classe inconnue reste ainsi visible plutot que d'etre masquee.
    """
    table = CLASS_LABELS_FR if labels is None else labels
    return table.get(name, name)


def display_names(labels: Mapping[str, str] | None = None) -> dict[str, str]:
    """Table complete des libelles, utile pour l'interface et les rapports."""
    table = dict(CLASS_LABELS_FR)
    if labels:
        table.update(labels)
    return table


# --------------------------------------------------------------------------- #
# Faisabilite par EPI
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClassFeasibility:
    """Evaluation honnete de ce qui est distinguable visuellement.

    Toutes les distinctions ne se valent pas. Separer un casque de chantier d'un
    casque de velo repose sur des differences geometriques nettes ; separer une
    chaussure de securite d'une chaussure de travail ordinaire repose sur un
    embout d'acier **interne**, invisible sur une image. Documenter cette limite
    evite d'investir dans une annotation qui ne peut pas aboutir.

    Attributes:
        level: ``"high"``, ``"medium"`` ou ``"low"``.
        rationale: Ce qui rend la distinction possible ou non.
        recommended: Faut-il investir dans cette classe.
    """

    level: str
    rationale: str
    recommended: bool


FEASIBILITY: dict[str, ClassFeasibility] = {
    "Non-Safety Headwear": ClassFeasibility(
        level="high",
        rationale=(
            "Les differences sont geometriques et visibles : aerations, forme de la "
            "coque, jugulaire, bord. Un casque de velo, une casquette et un casque de "
            "chantier se distinguent nettement, meme a taille moderee."
        ),
        recommended=True,
    ),
    "Non-Safety Vest": ClassFeasibility(
        level="medium",
        rationale=(
            "Les bandes retroreflechissantes constituent un signal fort et apprenable. "
            "Mais une veste orange vive sans bandes reste confondue de loin, et de nuit "
            "le retroreflechissant sature et change d'aspect."
        ),
        recommended=True,
    ),
    "Uncovered Head": ClassFeasibility(
        level="high",
        rationale=(
            "Une tete non protegee par un casque de chantier est bien delimitee et "
            "abondamment annotee en contexte industriel (5 000 images CC0 disponibles). "
            "C'est la reponse operationnelle directe a « cette personne porte-t-elle un "
            "casque ? », et elle englobe les casquettes de travail, bonnets et casquettes "
            "anti-heurt — qu'aucun dataset public n'annote comme tels."
        ),
        recommended=True,
    ),
    "Non-Safety Footwear": ClassFeasibility(
        level="low",
        rationale=(
            "Une chaussure de securite et une chaussure de travail ordinaire sont "
            "visuellement quasi identiques : la difference (embout d'acier) est interne. "
            "A la taille mediane observee dans le dataset (73 px), aucun modele ne peut "
            "trancher. Un classificateur entraine la-dessus apprendrait des correlations "
            "trompeuses (couleur, marque, contexte) et echouerait silencieusement."
        ),
        recommended=False,
    ),
}


# --------------------------------------------------------------------------- #
# Manipulation de schemas
# --------------------------------------------------------------------------- #
@dataclass
class ClassSchema:
    """Un schema de classes, avec conversion nom <-> identifiant."""

    names: list[str] = field(default_factory=lambda: list(BASE_CLASSES))

    def __post_init__(self) -> None:
        """Verifie l'unicite des noms."""
        seen: set[str] = set()
        duplicates = [n for n in self.names if n in seen or seen.add(n)]  # type: ignore[func-returns-value]
        if duplicates:
            raise ValueError(f"Noms de classes dupliques : {sorted(set(duplicates))}")

    @property
    def size(self) -> int:
        """Nombre de classes."""
        return len(self.names)

    def id_of(self, name: str) -> int:
        """Identifiant d'une classe.

        Raises:
            KeyError: Si la classe est absente du schema, avec la liste des
                classes disponibles pour faciliter le diagnostic.
        """
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"Classe inconnue : {name!r}. Classes du schema : {', '.join(self.names)}"
            ) from exc

    def name_of(self, class_id: int) -> str:
        """Nom d'une classe a partir de son identifiant."""
        if not 0 <= class_id < len(self.names):
            raise KeyError(
                f"Identifiant hors bornes : {class_id} (schema de {len(self.names)} classes)"
            )
        return self.names[class_id]

    def is_negative(self, name: str) -> bool:
        """Indique si la classe est une classe « sosie »."""
        return name in NEGATIVE_CLASSES

    def extended(self) -> ClassSchema:
        """Retourne le schema complete des classes negatives absentes."""
        missing = [n for n in NEGATIVE_CLASSES if n not in self.names]
        return ClassSchema(names=[*self.names, *missing])

    def to_data_yaml(self) -> dict[str, Any]:
        """Fragment ``nc`` / ``names`` pour un ``data.yaml``."""
        return {"nc": self.size, "names": list(self.names)}


def base_schema() -> ClassSchema:
    """Schema d'origine a sept classes."""
    return ClassSchema(names=list(BASE_CLASSES))


def extended_schema() -> ClassSchema:
    """Schema etendu a dix classes."""
    return ClassSchema(names=list(EXTENDED_CLASSES))


def build_class_mapping(
    source_names: list[str], target: ClassSchema, mapping: dict[str, str]
) -> dict[int, int]:
    """Traduit les identifiants d'un dataset externe vers le schema cible.

    Sert a integrer un dataset public dont les classes portent d'autres noms :
    la classe ``"With Helmet"`` d'un dataset de casques de velo devient
    ``"Non-Safety Headwear"``.

    Args:
        source_names: Classes du dataset source, dans l'ordre de ses identifiants.
        target: Schema de destination.
        mapping: ``{nom_source: nom_cible}``. Une valeur vide ou absente exclut
            la classe : ses annotations seront ignorees.

    Returns:
        ``{id_source: id_cible}`` pour les seules classes conservees.

    Raises:
        KeyError: Si un nom cible n'existe pas dans le schema.
    """
    result: dict[int, int] = {}
    for source_id, source_name in enumerate(source_names):
        target_name = mapping.get(source_name)
        if not target_name:
            continue
        result[source_id] = target.id_of(target_name)
    return result


def describe_feasibility() -> str:
    """Rend un resume lisible de la faisabilite de chaque classe negative."""
    lines = []
    for name in NEGATIVE_CLASSES:
        info = FEASIBILITY[name]
        verdict = "recommandee" if info.recommended else "DECONSEILLEE"
        lines.append(f"{name} [{info.level}] — {verdict}\n    {info.rationale}")
    return "\n".join(lines)
