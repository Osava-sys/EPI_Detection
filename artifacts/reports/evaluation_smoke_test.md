# Rapport d'evaluation — split `test`

- **Genere le** : 2026-07-30T10:14:06.107693+00:00
- **Poids** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\models\smoke_best.pt` (19.4 Mo)
- **Parametres** : 9467889
- **Dataset** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\dataset_detection\data.yaml`
- **Parametres d'evaluation** : imgsz=640, batch=16, conf=0.001, iou=0.6, device=0

## Metriques globales

| Metrique            | Valeur |
|---------------------|--------|
| Precision (moyenne) | 0.2635 |
| Rappel (moyen)      | 0.1068 |
| **mAP@0.50**        | 0.0882 |
| **mAP@0.50:0.95**   | 0.0334 |
| mAP@0.75            | 0.0148 |

## Metriques par classe

| Classe         | Precision | Rappel | mAP@0.50 | mAP@0.50:0.95 |
|----------------|-----------|--------|----------|---------------|
| Face Mask      | 1.0       | 0.0    | 0.0      | 0.0           |
| Person         | 0.359     | 0.6157 | 0.4209   | 0.1649        |
| Safety Gloves  | 0.0       | 0.0    | 0.0003   | 0.0001        |
| Safety Harness | 0.0       | 0.0    | 0.0007   | 0.0001        |
| Safety Helmet  | 0.4855    | 0.1316 | 0.1914   | 0.0674        |
| Safety Shoes   | 0.0       | 0.0    | 0.0014   | 0.0004        |
| Safety Vest    | 0.0       | 0.0    | 0.0025   | 0.0006        |

## Vitesse

| Etape       | Temps par image |
|-------------|-----------------|
| preprocess  | 0.789 ms        |
| inference   | 2.755 ms        |
| loss        | 0.0 ms          |
| postprocess | 0.106 ms        |
| **Debit**   | 273.97 images/s |

## Analyse d'erreurs

Realisee au seuil operationnel `conf=0.25` et `IoU>=0.5` sur 60 image(s).

| Indicateur     | Total |
|----------------|-------|
| Vrais positifs | 184   |
| Faux positifs  | 201   |
| Faux negatifs  | 292   |

### Detail par classe

| Classe         | VP  | FP  | FN  | Precision | Rappel | F1     |
|----------------|-----|-----|-----|-----------|--------|--------|
| Face Mask      | 0   | 0   | 0   | 0.0       | 0.0    | 0.0    |
| Person         | 151 | 150 | 43  | 0.5017    | 0.7784 | 0.6101 |
| Safety Gloves  | 0   | 16  | 15  | 0.0       | 0.0    | 0.0    |
| Safety Harness | 0   | 1   | 0   | 0.0       | 0.0    | 0.0    |
| Safety Helmet  | 33  | 33  | 181 | 0.5       | 0.1542 | 0.2357 |
| Safety Shoes   | 0   | 0   | 30  | 0.0       | 0.0    | 0.0    |
| Safety Vest    | 0   | 1   | 23  | 0.0       | 0.0    | 0.0    |

### Confusions entre classes

| Classe reelle | Predite comme | Occurrences |
|---------------|---------------|-------------|
| Safety Vest   | Person        | 3           |
| Safety Helmet | Safety Gloves | 3           |

### Pires exemples (a inspecter en priorite)

| Image                                      | Reference | Predictions | VP | FP | FN | Score  |
|--------------------------------------------|-----------|-------------|----|----|----|--------|
| -1387-_png_jpg.rf.pkQrmlUcVQzzOEt4QiZW.jpg | 2         | 3           | 0  | 3  | 2  | 0.0    |
| -2641-_png_jpg.rf.fWsNy9yLWRzrtl1lmxJL.jpg | 10        | 5           | 0  | 5  | 10 | 0.0    |
| -1020-_png_jpg.rf.DygJlOObbh6zEatUNApF.jpg | 26        | 6           | 3  | 3  | 23 | 0.1034 |
| -1116-_png_jpg.rf.eO6sjUYKarm5wVVBBDtj.jpg | 4         | 6           | 1  | 5  | 2  | 0.1111 |
| -1441-_png_jpg.rf.ARKacGR9xKjIQxKsfb6x.jpg | 3         | 7           | 1  | 6  | 2  | 0.1111 |
| -2066-_png_jpg.rf.sOymzFTePtLKXhNAhdmt.jpg | 4         | 6           | 1  | 5  | 3  | 0.1111 |
| -1400-_png_jpg.rf.3mG3sJ90WY3OkGox5Yq0.jpg | 4         | 5           | 1  | 4  | 3  | 0.125  |
| -1591-_png_jpg.rf.NMkMcxsLuV0SoCOJ47dE.jpg | 4         | 5           | 1  | 4  | 2  | 0.125  |
| -1916-_png_jpg.rf.aybHvgoswOQ4AXn4Wf07.jpg | 4         | 4           | 1  | 3  | 3  | 0.1429 |
| -1792-_png_jpg.rf.UJ2w0ZuxnOf4AgyNvXSq.jpg | 14        | 8           | 3  | 5  | 11 | 0.1579 |

### Meilleurs exemples

| Image                                      | Reference | Predictions | VP | FP | FN | Score  |
|--------------------------------------------|-----------|-------------|----|----|----|--------|
| -1245-_png_jpg.rf.4g4OwqUdBmImS16jOBPa.jpg | 6         | 4           | 4  | 0  | 2  | 0.6667 |
| -162-_png_jpg.rf.jC5cKJvDhmHvAfI1vLBO.jpg  | 4         | 4           | 3  | 1  | 1  | 0.6    |
| -1154-_png_jpg.rf.TvHLl7ao8gk4oy0jXkF8.jpg | 5         | 4           | 3  | 1  | 2  | 0.5    |
| -134-_png_jpg.rf.wSV4tygIGjmmRpwgQFSQ.jpg  | 2         | 1           | 1  | 0  | 1  | 0.5    |
| -1473-_png_jpg.rf.x4D4mDrRLOcKUAilnB7r.jpg | 2         | 1           | 1  | 0  | 1  | 0.5    |
| -1924-_png_jpg.rf.i30vwjsr5ZcNmnFNv6Ne.jpg | 2         | 1           | 1  | 0  | 1  | 0.5    |
| -2434-_png_jpg.rf.GdrObI6YRqCvyxUyiffk.jpg | 11        | 7           | 6  | 1  | 5  | 0.5    |
| -1253-_png_jpg.rf.dOrZfLiMSmgZzUvD1LDm.jpg | 16        | 15          | 10 | 5  | 6  | 0.4762 |
| -2562-_png_jpg.rf.j6n5oOtlPvCWb9ZTGCKw.jpg | 9         | 8           | 5  | 3  | 4  | 0.4167 |
| -2366-_png_jpg.rf.sgrmaw2cSsrbeiOQoJuK.jpg | 16        | 12          | 8  | 4  | 8  | 0.4    |

## Limites connues

- Le split de test provient du meme export Roboflow que l'entrainement : l'audit a identifie des images issues d'une meme photo source reparties entre plusieurs splits, ce qui rend les metriques optimistes par rapport a un deploiement sur un chantier inconnu.
- Les metriques sont calculees sur des annotations dont 349 lignes ont ete converties depuis des polygones de segmentation : la boite englobante d'un polygone est systematiquement au moins aussi grande que l'objet reel.
- Classes dont la mAP@0.50 reste inferieure a 0.50 : Face Mask, Person, Safety Gloves, Safety Harness, Safety Helmet, Safety Shoes, Safety Vest. Ces classes demandent davantage d'exemples annotes.
- Confusion la plus frequente : 3 objet(s) de classe 'Safety Vest' predits comme 'Person'.

> Les courbes et la matrice de confusion generees par Ultralytics se trouvent dans `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\runs\val\evaluation_smoke_test`.
