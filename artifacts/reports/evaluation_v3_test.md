# Rapport d'evaluation — split `test`

- **Genere le** : 2026-08-17T18:43:21.907010+00:00
- **Poids** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\models\best_v3.pt` (19.4 Mo)
- **Parametres** : 9468663
- **Dataset** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\dataset_v3\data.yaml`
- **Parametres d'evaluation** : imgsz=640, batch=16, conf=0.001, iou=0.6, device=0

## Metriques globales

| Metrique            | Valeur |
|---------------------|--------|
| Precision (moyenne) | 0.836  |
| Rappel (moyen)      | 0.8009 |
| **mAP@0.50**        | 0.8368 |
| **mAP@0.50:0.95**   | 0.5077 |
| mAP@0.75            | 0.522  |

## Metriques par classe

| Classe              | Precision | Rappel | mAP@0.50 | mAP@0.50:0.95 |
|---------------------|-----------|--------|----------|---------------|
| Face Mask           | 0.9046    | 0.9323 | 0.9584   | 0.5831        |
| Non-Safety Headwear | 0.7757    | 0.7632 | 0.7721   | 0.4894        |
| Person              | 0.8581    | 0.9067 | 0.9169   | 0.6777        |
| Safety Gloves       | 0.6691    | 0.5474 | 0.5251   | 0.2257        |
| Safety Harness      | 0.8368    | 0.6931 | 0.796    | 0.4048        |
| Safety Helmet       | 0.9092    | 0.9076 | 0.9479   | 0.5907        |
| Safety Shoes        | 0.8082    | 0.6944 | 0.762    | 0.4194        |
| Safety Vest         | 0.8512    | 0.864  | 0.903    | 0.5521        |
| Uncovered Head      | 0.9112    | 0.8996 | 0.95     | 0.6262        |

## Vitesse

| Etape       | Temps par image |
|-------------|-----------------|
| preprocess  | 0.71 ms         |
| inference   | 2.146 ms        |
| loss        | 0.0 ms          |
| postprocess | 0.081 ms        |
| **Debit**   | 340.48 images/s |

## Analyse d'erreurs

Realisee au seuil operationnel `conf=0.25` et `IoU>=0.5` sur 1263 image(s).

| Indicateur     | Total |
|----------------|-------|
| Vrais positifs | 6173  |
| Faux positifs  | 1233  |
| Faux negatifs  | 768   |

### Detail par classe

| Classe              | VP   | FP  | FN  | Precision | Rappel | F1     |
|---------------------|------|-----|-----|-----------|--------|--------|
| Face Mask           | 57   | 11  | 4   | 0.8382    | 0.9344 | 0.8837 |
| Non-Safety Headwear | 286  | 103 | 72  | 0.7352    | 0.7989 | 0.7657 |
| Person              | 2328 | 512 | 202 | 0.8197    | 0.9202 | 0.867  |
| Safety Gloves       | 73   | 48  | 64  | 0.6033    | 0.5328 | 0.5659 |
| Safety Harness      | 81   | 23  | 30  | 0.7788    | 0.7297 | 0.7535 |
| Safety Helmet       | 2401 | 333 | 206 | 0.8782    | 0.921  | 0.8991 |
| Safety Shoes        | 279  | 82  | 117 | 0.7729    | 0.7045 | 0.7371 |
| Safety Vest         | 197  | 50  | 31  | 0.7976    | 0.864  | 0.8295 |
| Uncovered Head      | 471  | 71  | 42  | 0.869     | 0.9181 | 0.8929 |

### Confusions entre classes

| Classe reelle       | Predite comme  | Occurrences |
|---------------------|----------------|-------------|
| Safety Helmet       | Uncovered Head | 9           |
| Uncovered Head      | Safety Helmet  | 3           |
| Safety Vest         | Safety Gloves  | 2           |
| Safety Vest         | Safety Harness | 2           |
| Safety Harness      | Person         | 1           |
| Non-Safety Headwear | Person         | 1           |
| Safety Helmet       | Safety Gloves  | 1           |
| Safety Shoes        | Safety Gloves  | 1           |
| Person              | Safety Harness | 1           |
| Safety Harness      | Safety Vest    | 1           |
| Safety Shoes        | Safety Vest    | 1           |

### Pires exemples (a inspecter en priorite)

| Image                                                                                                          | Reference | Predictions | VP | FP | FN | Score |
|----------------------------------------------------------------------------------------------------------------|-----------|-------------|----|----|----|-------|
| hardhat-hf__hard_hat_workers207__2c692339.png                                                                  | 6         | 0           | 0  | 0  | 6  | 0.0   |
| open-images__01012c2e299f281b__39dffa25.jpg                                                                    | 1         | 0           | 0  | 0  | 1  | 0.0   |
| open-images__024a2c77ee7857fe__d24e7a93.jpg                                                                    | 3         | 1           | 0  | 1  | 3  | 0.0   |
| open-images__5d2f872779c9eb77__1902f9e3.jpg                                                                    | 1         | 1           | 0  | 1  | 0  | 0.0   |
| open-images__5e432cf1a6415182__24746144.jpg                                                                    | 1         | 1           | 0  | 1  | 1  | 0.0   |
| ppe-original__-2609-_png_jpg.rf.rl366pGPQbGlgxf1AuZm__07f07686.jpg                                             | 8         | 0           | 0  | 0  | 8  | 0.0   |
| ppe-original__Construction-worker-wearing-safety-shoes-scaled-1_jpeg_jpg.rf.QSoK4kkwN4jlDtUxxl8y__5d3d6ec6.jpg | 2         | 0           | 0  | 0  | 2  | 0.0   |
| ppe-original__Construction-worker-wearing-safety-shoes-scaled-1_jpeg_jpg.rf.Z88wKwYsIgJPMvg56sWB__42422b29.jpg | 3         | 0           | 0  | 0  | 3  | 0.0   |
| ppe-original__Construction-worker-wearing-safety-shoes-scaled-1_jpeg_jpg.rf.mL1277Me2sg2oLKmUCRF__89c1428f.jpg | 3         | 0           | 0  | 0  | 3  | 0.0   |
| ppe-original__construction_jpg.rf.LNRJ3lNOWiB35KfZOqkd__e3c5651b.jpg                                           | 2         | 0           | 0  | 0  | 2  | 0.0   |

### Meilleurs exemples

| Image                                          | Reference | Predictions | VP | FP | FN | Score |
|------------------------------------------------|-----------|-------------|----|----|----|-------|
| hardhat-hf__hard_hat_workers1000__49f6cbf4.png | 3         | 3           | 3  | 0  | 0  | 1.0   |
| hardhat-hf__hard_hat_workers1028__3b1acdc2.png | 4         | 4           | 4  | 0  | 0  | 1.0   |
| hardhat-hf__hard_hat_workers1055__1f29a919.png | 2         | 2           | 2  | 0  | 0  | 1.0   |
| hardhat-hf__hard_hat_workers1086__87b1676d.png | 3         | 3           | 3  | 0  | 0  | 1.0   |
| hardhat-hf__hard_hat_workers1101__0562f878.png | 6         | 6           | 6  | 0  | 0  | 1.0   |
| hardhat-hf__hard_hat_workers1104__f132c0ba.png | 2         | 2           | 2  | 0  | 0  | 1.0   |
| hardhat-hf__hard_hat_workers110__dd2ff4a3.png  | 2         | 2           | 2  | 0  | 0  | 1.0   |
| hardhat-hf__hard_hat_workers1126__c1dce8cb.png | 4         | 4           | 4  | 0  | 0  | 1.0   |
| hardhat-hf__hard_hat_workers1136__342e6f3f.png | 7         | 7           | 7  | 0  | 0  | 1.0   |
| hardhat-hf__hard_hat_workers1169__7760f573.png | 2         | 2           | 2  | 0  | 0  | 1.0   |

## Limites connues

- Le split de test provient du meme export Roboflow que l'entrainement : l'audit a identifie des images issues d'une meme photo source reparties entre plusieurs splits, ce qui rend les metriques optimistes par rapport a un deploiement sur un chantier inconnu.
- Les metriques sont calculees sur des annotations dont 349 lignes ont ete converties depuis des polygones de segmentation : la boite englobante d'un polygone est systematiquement au moins aussi grande que l'objet reel.
- Confusion la plus frequente : 9 objet(s) de classe 'Safety Helmet' predits comme 'Uncovered Head'.

> Les courbes et la matrice de confusion generees par Ultralytics se trouvent dans `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\runs\val\evaluation_v3_test`.
