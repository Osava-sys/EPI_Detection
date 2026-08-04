# Rapport d'evaluation — split `test`

- **Genere le** : 2026-08-04T15:08:58.852497+00:00
- **Poids** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\models\best.pt` (19.4 Mo)
- **Parametres** : 9467889
- **Dataset** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\dataset_detection\data.yaml`
- **Parametres d'evaluation** : imgsz=960, batch=16, conf=0.001, iou=0.6, device=0

## Metriques globales

| Metrique            | Valeur |
|---------------------|--------|
| Precision (moyenne) | 0.8059 |
| Rappel (moyen)      | 0.7572 |
| **mAP@0.50**        | 0.798  |
| **mAP@0.50:0.95**   | 0.4276 |
| mAP@0.75            | 0.4039 |

## Metriques par classe

| Classe         | Precision | Rappel | mAP@0.50 | mAP@0.50:0.95 |
|----------------|-----------|--------|----------|---------------|
| Face Mask      | 0.8384    | 0.8197 | 0.8892   | 0.5456        |
| Person         | 0.874     | 0.8843 | 0.9152   | 0.5332        |
| Safety Gloves  | 0.7107    | 0.5559 | 0.5757   | 0.2375        |
| Safety Harness | 0.7994    | 0.6823 | 0.7553   | 0.3927        |
| Safety Helmet  | 0.7684    | 0.8111 | 0.8096   | 0.345         |
| Safety Shoes   | 0.8084    | 0.6919 | 0.7673   | 0.4165        |
| Safety Vest    | 0.8418    | 0.8553 | 0.8733   | 0.5228        |

## Vitesse

| Etape       | Temps par image |
|-------------|-----------------|
| preprocess  | 1.724 ms        |
| inference   | 5.832 ms        |
| loss        | 0.0 ms          |
| postprocess | 0.111 ms        |
| **Debit**   | 130.43 images/s |

## Limites connues

- Le split de test provient du meme export Roboflow que l'entrainement : l'audit a identifie des images issues d'une meme photo source reparties entre plusieurs splits, ce qui rend les metriques optimistes par rapport a un deploiement sur un chantier inconnu.
- Les metriques sont calculees sur des annotations dont 349 lignes ont ete converties depuis des polygones de segmentation : la boite englobante d'un polygone est systematiquement au moins aussi grande que l'objet reel.

> Les courbes et la matrice de confusion generees par Ultralytics se trouvent dans `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\runs\val\eval_960_test`.
