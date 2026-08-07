# Rapport d'evaluation — split `test`

- **Genere le** : 2026-08-07T16:51:22.343362+00:00
- **Poids** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\models\best_8classes.pt` (19.4 Mo)
- **Parametres** : 9468276
- **Dataset** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\dataset_common_test\data_8cls.yaml`
- **Parametres d'evaluation** : imgsz=640, batch=16, conf=0.001, iou=0.6, device=0

## Metriques globales

| Metrique            | Valeur |
|---------------------|--------|
| Precision (moyenne) | 0.8083 |
| Rappel (moyen)      | 0.7451 |
| **mAP@0.50**        | 0.7988 |
| **mAP@0.50:0.95**   | 0.4339 |
| mAP@0.75            | 0.4189 |

## Metriques par classe

| Classe         | Precision | Rappel | mAP@0.50 | mAP@0.50:0.95 |
|----------------|-----------|--------|----------|---------------|
| Face Mask      | 0.8129    | 0.8548 | 0.8931   | 0.5284        |
| Person         | 0.8717    | 0.888  | 0.896    | 0.5283        |
| Safety Gloves  | 0.6867    | 0.5255 | 0.5607   | 0.2385        |
| Safety Harness | 0.7904    | 0.7117 | 0.8091   | 0.4228        |
| Safety Helmet  | 0.7874    | 0.7697 | 0.8067   | 0.3611        |
| Safety Shoes   | 0.8299    | 0.6591 | 0.7523   | 0.4259        |
| Safety Vest    | 0.8793    | 0.807  | 0.8741   | 0.5325        |

## Vitesse

| Etape       | Temps par image |
|-------------|-----------------|
| preprocess  | 0.728 ms        |
| inference   | 3.887 ms        |
| loss        | 0.0 ms          |
| postprocess | 0.089 ms        |
| **Debit**   | 212.59 images/s |

## Limites connues

- Le split de test provient du meme export Roboflow que l'entrainement : l'audit a identifie des images issues d'une meme photo source reparties entre plusieurs splits, ce qui rend les metriques optimistes par rapport a un deploiement sur un chantier inconnu.
- Les metriques sont calculees sur des annotations dont 349 lignes ont ete converties depuis des polygones de segmentation : la boite englobante d'un polygone est systematiquement au moins aussi grande que l'objet reel.

> Les courbes et la matrice de confusion generees par Ultralytics se trouvent dans `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\runs\val\common_8cls`.
