# Rapport d'evaluation — split `test`

- **Genere le** : 2026-08-04T15:08:30.398030+00:00
- **Poids** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\models\best_640.pt` (19.4 Mo)
- **Parametres** : 9467889
- **Dataset** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\dataset_detection\data.yaml`
- **Parametres d'evaluation** : imgsz=640, batch=16, conf=0.001, iou=0.6, device=0

## Metriques globales

| Metrique            | Valeur |
|---------------------|--------|
| Precision (moyenne) | 0.8086 |
| Rappel (moyen)      | 0.7539 |
| **mAP@0.50**        | 0.7992 |
| **mAP@0.50:0.95**   | 0.4326 |
| mAP@0.75            | 0.4078 |

## Metriques par classe

| Classe         | Precision | Rappel | mAP@0.50 | mAP@0.50:0.95 |
|----------------|-----------|--------|----------|---------------|
| Face Mask      | 0.8606    | 0.8033 | 0.9224   | 0.5595        |
| Person         | 0.8757    | 0.8754 | 0.8927   | 0.5306        |
| Safety Gloves  | 0.6725    | 0.5547 | 0.5483   | 0.221         |
| Safety Harness | 0.8274    | 0.7117 | 0.7821   | 0.4003        |
| Safety Helmet  | 0.7761    | 0.7842 | 0.7961   | 0.3524        |
| Safety Shoes   | 0.7983    | 0.6793 | 0.7751   | 0.4296        |
| Safety Vest    | 0.8495    | 0.8684 | 0.8777   | 0.5347        |

## Vitesse

| Etape       | Temps par image |
|-------------|-----------------|
| preprocess  | 0.776 ms        |
| inference   | 2.764 ms        |
| loss        | 0.0 ms          |
| postprocess | 0.151 ms        |
| **Debit**   | 270.93 images/s |

## Limites connues

- Le split de test provient du meme export Roboflow que l'entrainement : l'audit a identifie des images issues d'une meme photo source reparties entre plusieurs splits, ce qui rend les metriques optimistes par rapport a un deploiement sur un chantier inconnu.
- Les metriques sont calculees sur des annotations dont 349 lignes ont ete converties depuis des polygones de segmentation : la boite englobante d'un polygone est systematiquement au moins aussi grande que l'objet reel.

> Les courbes et la matrice de confusion generees par Ultralytics se trouvent dans `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\runs\val\eval_640_test`.
