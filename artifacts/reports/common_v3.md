# Rapport d'evaluation — split `test`

- **Genere le** : 2026-08-17T18:42:14.341106+00:00
- **Poids** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\models\best_v3.pt` (19.4 Mo)
- **Parametres** : 9468663
- **Dataset** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\dataset_common_test\data_9cls.yaml`
- **Parametres d'evaluation** : imgsz=640, batch=16, conf=0.001, iou=0.6, device=0

## Metriques globales

| Metrique            | Valeur |
|---------------------|--------|
| Precision (moyenne) | 0.825  |
| Rappel (moyen)      | 0.7723 |
| **mAP@0.50**        | 0.8077 |
| **mAP@0.50:0.95**   | 0.4394 |
| mAP@0.75            | 0.404  |

## Metriques par classe

| Classe         | Precision | Rappel | mAP@0.50 | mAP@0.50:0.95 |
|----------------|-----------|--------|----------|---------------|
| Face Mask      | 0.9047    | 0.9338 | 0.9592   | 0.5826        |
| Person         | 0.8733    | 0.8894 | 0.8771   | 0.5264        |
| Safety Gloves  | 0.699     | 0.5474 | 0.5535   | 0.239         |
| Safety Harness | 0.8368    | 0.6929 | 0.7969   | 0.4064        |
| Safety Helmet  | 0.7856    | 0.7818 | 0.794    | 0.3455        |
| Safety Shoes   | 0.8177    | 0.697  | 0.7653   | 0.4212        |
| Safety Vest    | 0.858     | 0.864  | 0.9077   | 0.5544        |

## Vitesse

| Etape       | Temps par image |
|-------------|-----------------|
| preprocess  | 0.664 ms        |
| inference   | 2.512 ms        |
| loss        | 0.0 ms          |
| postprocess | 0.14 ms         |
| **Debit**   | 301.57 images/s |

## Limites connues

- Le split de test provient du meme export Roboflow que l'entrainement : l'audit a identifie des images issues d'une meme photo source reparties entre plusieurs splits, ce qui rend les metriques optimistes par rapport a un deploiement sur un chantier inconnu.
- Les metriques sont calculees sur des annotations dont 349 lignes ont ete converties depuis des polygones de segmentation : la boite englobante d'un polygone est systematiquement au moins aussi grande que l'objet reel.

> Les courbes et la matrice de confusion generees par Ultralytics se trouvent dans `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\runs\val\common_v3`.
