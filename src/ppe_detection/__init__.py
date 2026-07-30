"""Detection d'equipements de protection individuelle (EPI) avec Ultralytics YOLO.

Modules principaux :

* :mod:`ppe_detection.annotations` — parsing/normalisation des labels YOLO.
* :mod:`ppe_detection.dataset_audit` — audit complet du dataset.
* :mod:`ppe_detection.dataset_cleaner` — construction du dataset de detection derive.
* :mod:`ppe_detection.train` — entrainement.
* :mod:`ppe_detection.evaluate` — evaluation.
* :mod:`ppe_detection.predict` — inference image/dossier/video/webcam.
* :mod:`ppe_detection.compliance` — regles metier EPI (heuristique).
* :mod:`ppe_detection.api` — API REST FastAPI.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
