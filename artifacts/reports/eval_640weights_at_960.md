# Rapport d'evaluation — split `test`

- **Genere le** : 2026-08-04T10:23:55.664106+00:00
- **Poids** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\models\best.pt` (19.4 Mo)
- **Parametres** : 9467889
- **Dataset** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\dataset_detection_nofleak\data.yaml`
- **Parametres d'evaluation** : imgsz=960, batch=16, conf=0.001, iou=0.6, device=0

## Metriques globales

| Metrique            | Valeur |
|---------------------|--------|
| Precision (moyenne) | 0.7865 |
| Rappel (moyen)      | 0.7225 |
| **mAP@0.50**        | 0.78   |
| **mAP@0.50:0.95**   | 0.3919 |
| mAP@0.75            | 0.3363 |

## Metriques par classe

| Classe         | Precision | Rappel | mAP@0.50 | mAP@0.50:0.95 |
|----------------|-----------|--------|----------|---------------|
| Face Mask      | 0.81      | 0.8033 | 0.8918   | 0.5033        |
| Person         | 0.861     | 0.8613 | 0.8983   | 0.4983        |
| Safety Gloves  | 0.7106    | 0.4964 | 0.5588   | 0.2197        |
| Safety Harness | 0.7008    | 0.6329 | 0.6943   | 0.3215        |
| Safety Helmet  | 0.7909    | 0.7939 | 0.812    | 0.3535        |
| Safety Shoes   | 0.813     | 0.6843 | 0.7659   | 0.398         |
| Safety Vest    | 0.8195    | 0.7851 | 0.8387   | 0.4488        |

## Vitesse

| Etape       | Temps par image |
|-------------|-----------------|
| preprocess  | 1.687 ms        |
| inference   | 7.906 ms        |
| loss        | 0.0 ms          |
| postprocess | 0.104 ms        |
| **Debit**   | 103.12 images/s |

## Limites connues

- Le split de test provient du meme export Roboflow que l'entrainement : l'audit a identifie des images issues d'une meme photo source reparties entre plusieurs splits, ce qui rend les metriques optimistes par rapport a un deploiement sur un chantier inconnu.
- Les metriques sont calculees sur des annotations dont 349 lignes ont ete converties depuis des polygones de segmentation : la boite englobante d'un polygone est systematiquement au moins aussi grande que l'objet reel.

> Les courbes et la matrice de confusion generees par Ultralytics se trouvent dans `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\runs\val\eval_960`.
