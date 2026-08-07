# Rapport d'evaluation — split `test`

- **Genere le** : 2026-08-07T16:52:37.744191+00:00
- **Poids** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\models\best_8classes.pt` (19.4 Mo)
- **Parametres** : 9468276
- **Dataset** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\dataset_extended\data.yaml`
- **Parametres d'evaluation** : imgsz=640, batch=16, conf=0.001, iou=0.6, device=0

## Metriques globales

| Metrique            | Valeur |
|---------------------|--------|
| Precision (moyenne) | 0.8126 |
| Rappel (moyen)      | 0.7325 |
| **mAP@0.50**        | 0.7906 |
| **mAP@0.50:0.95**   | 0.4317 |
| mAP@0.75            | 0.4178 |

## Metriques par classe

| Classe              | Precision | Rappel | mAP@0.50 | mAP@0.50:0.95 |
|---------------------|-----------|--------|----------|---------------|
| Face Mask           | 0.8101    | 0.8392 | 0.8955   | 0.532         |
| Non-Safety Headwear | 0.829     | 0.6741 | 0.7373   | 0.4161        |
| Person              | 0.873     | 0.888  | 0.8953   | 0.5279        |
| Safety Gloves       | 0.6913    | 0.5255 | 0.5576   | 0.2373        |
| Safety Harness      | 0.7949    | 0.7117 | 0.8075   | 0.4219        |
| Safety Helmet       | 0.7892    | 0.7697 | 0.8063   | 0.3612        |
| Safety Shoes        | 0.8299    | 0.654  | 0.7515   | 0.4255        |
| Safety Vest         | 0.8834    | 0.7978 | 0.8733   | 0.5318        |

## Vitesse

| Etape       | Temps par image |
|-------------|-----------------|
| preprocess  | 0.669 ms        |
| inference   | 2.19 ms         |
| loss        | 0.001 ms        |
| postprocess | 0.135 ms        |
| **Debit**   | 333.89 images/s |

## Analyse d'erreurs

Realisee au seuil operationnel `conf=0.25` et `IoU>=0.5` sur 698 image(s).

| Indicateur     | Total |
|----------------|-------|
| Vrais positifs | 1945  |
| Faux positifs  | 577   |
| Faux negatifs  | 535   |

### Detail par classe

| Classe              | VP  | FP  | FN  | Precision | Rappel | F1     |
|---------------------|-----|-----|-----|-----------|--------|--------|
| Face Mask           | 54  | 18  | 7   | 0.75      | 0.8852 | 0.812  |
| Non-Safety Headwear | 237 | 73  | 101 | 0.7645    | 0.7012 | 0.7315 |
| Person              | 643 | 139 | 71  | 0.8223    | 0.9006 | 0.8596 |
| Safety Gloves       | 73  | 42  | 64  | 0.6348    | 0.5328 | 0.5794 |
| Safety Harness      | 84  | 31  | 27  | 0.7304    | 0.7568 | 0.7434 |
| Safety Helmet       | 408 | 152 | 87  | 0.7286    | 0.8242 | 0.7735 |
| Safety Shoes        | 260 | 82  | 136 | 0.7602    | 0.6566 | 0.7046 |
| Safety Vest         | 186 | 40  | 42  | 0.823     | 0.8158 | 0.8194 |

### Confusions entre classes

| Classe reelle  | Predite comme  | Occurrences |
|----------------|----------------|-------------|
| Person         | Safety Harness | 5           |
| Safety Vest    | Safety Harness | 5           |
| Safety Harness | Person         | 2           |
| Safety Vest    | Safety Gloves  | 2           |
| Safety Harness | Safety Vest    | 2           |
| Safety Shoes   | Safety Gloves  | 1           |
| Person         | Safety Helmet  | 1           |
| Safety Shoes   | Safety Vest    | 1           |

### Pires exemples (a inspecter en priorite)

| Image                                       | Reference | Predictions | VP | FP | FN | Score |
|---------------------------------------------|-----------|-------------|----|----|----|-------|
| open-images__0008fba87dd4aed2__a95f3b75.jpg | 1         | 0           | 0  | 0  | 1  | 0.0   |
| open-images__00c5fd925839f08d__91ff440e.jpg | 1         | 0           | 0  | 0  | 1  | 0.0   |
| open-images__00d36ecb99c04218__e503cc1f.jpg | 1         | 0           | 0  | 0  | 1  | 0.0   |
| open-images__010f0e8a3eae037f__d6180ec9.jpg | 1         | 1           | 0  | 1  | 1  | 0.0   |
| open-images__01272e1fba48e22b__8db61729.jpg | 5         | 0           | 0  | 0  | 5  | 0.0   |
| open-images__01817498689fd9af__bc525455.jpg | 1         | 1           | 0  | 1  | 1  | 0.0   |
| open-images__01cc341122052975__58c091e2.jpg | 1         | 2           | 0  | 2  | 1  | 0.0   |
| open-images__026cc6d89ba92087__ff107659.jpg | 1         | 1           | 0  | 1  | 1  | 0.0   |
| open-images__028d2474f978f362__52b44f4a.jpg | 1         | 1           | 0  | 1  | 1  | 0.0   |
| open-images__03c1578416f806b5__f528e817.jpg | 1         | 1           | 0  | 1  | 1  | 0.0   |

### Meilleurs exemples

| Image                                       | Reference | Predictions | VP | FP | FN | Score |
|---------------------------------------------|-----------|-------------|----|----|----|-------|
| open-images__000c899427d321da__4649de57.jpg | 1         | 1           | 1  | 0  | 0  | 1.0   |
| open-images__000de2d23a112361__e7dddef1.jpg | 3         | 3           | 3  | 0  | 0  | 1.0   |
| open-images__002e81410ab5f932__faa68cac.jpg | 1         | 1           | 1  | 0  | 0  | 1.0   |
| open-images__0030a88574b9faea__83290c8a.jpg | 1         | 1           | 1  | 0  | 0  | 1.0   |
| open-images__004cbeaeec83badb__7ca7c9db.jpg | 1         | 1           | 1  | 0  | 0  | 1.0   |
| open-images__0056931e46ed0539__ecb3ae6e.jpg | 1         | 1           | 1  | 0  | 0  | 1.0   |
| open-images__00a19985a766d15d__49968f21.jpg | 1         | 1           | 1  | 0  | 0  | 1.0   |
| open-images__00a3cd901721e453__9a8a14c1.jpg | 1         | 1           | 1  | 0  | 0  | 1.0   |
| open-images__00b6d43aa9e4337e__72fe9d63.jpg | 1         | 1           | 1  | 0  | 0  | 1.0   |
| open-images__01449c6c1abed851__35b4b1d9.jpg | 1         | 1           | 1  | 0  | 0  | 1.0   |

## Limites connues

- Le split de test provient du meme export Roboflow que l'entrainement : l'audit a identifie des images issues d'une meme photo source reparties entre plusieurs splits, ce qui rend les metriques optimistes par rapport a un deploiement sur un chantier inconnu.
- Les metriques sont calculees sur des annotations dont 349 lignes ont ete converties depuis des polygones de segmentation : la boite englobante d'un polygone est systematiquement au moins aussi grande que l'objet reel.
- Confusion la plus frequente : 5 objet(s) de classe 'Person' predits comme 'Safety Harness'.

> Les courbes et la matrice de confusion generees par Ultralytics se trouvent dans `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\runs\val\evaluation_extended_test`.
