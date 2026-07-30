# Rapport d'evaluation — split `test`

- **Genere le** : 2026-07-30T12:42:19.841840+00:00
- **Poids** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\models\best.pt` (19.4 Mo)
- **Parametres** : 9467889
- **Dataset** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\dataset_detection\data.yaml`
- **Parametres d'evaluation** : imgsz=640, batch=16, conf=0.001, iou=0.6, device=0

## Metriques globales

| Metrique            | Valeur |
|---------------------|--------|
| Precision (moyenne) | 0.8184 |
| Rappel (moyen)      | 0.7869 |
| **mAP@0.50**        | 0.8319 |
| **mAP@0.50:0.95**   | 0.4696 |
| mAP@0.75            | 0.4628 |

## Metriques par classe

| Classe         | Precision | Rappel | mAP@0.50 | mAP@0.50:0.95 |
|----------------|-----------|--------|----------|---------------|
| Face Mask      | 0.8673    | 0.769  | 0.9149   | 0.5649        |
| Person         | 0.8696    | 0.884  | 0.9009   | 0.5364        |
| Safety Gloves  | 0.7193    | 0.6784 | 0.6695   | 0.3338        |
| Safety Harness | 0.8285    | 0.7143 | 0.7834   | 0.3962        |
| Safety Helmet  | 0.7734    | 0.7878 | 0.7987   | 0.3589        |
| Safety Shoes   | 0.8527    | 0.7883 | 0.8691   | 0.5545        |
| Safety Vest    | 0.8179    | 0.8865 | 0.8869   | 0.5423        |

## Vitesse

| Etape       | Temps par image |
|-------------|-----------------|
| preprocess  | 0.711 ms        |
| inference   | 2.828 ms        |
| loss        | 0.0 ms          |
| postprocess | 0.087 ms        |
| **Debit**   | 275.79 images/s |

## Analyse d'erreurs

Realisee au seuil operationnel `conf=0.25` et `IoU>=0.5` sur 698 image(s).

| Indicateur     | Total |
|----------------|-------|
| Vrais positifs | 2074  |
| Faux positifs  | 564   |
| Faux negatifs  | 398   |

### Detail par classe

| Classe         | VP  | FP  | FN  | Precision | Rappel | F1     |
|----------------|-----|-----|-----|-----------|--------|--------|
| Face Mask      | 77  | 13  | 8   | 0.8556    | 0.9059 | 0.88   |
| Person         | 657 | 144 | 67  | 0.8202    | 0.9075 | 0.8616 |
| Safety Gloves  | 141 | 66  | 58  | 0.6812    | 0.7085 | 0.6946 |
| Safety Harness | 80  | 23  | 32  | 0.7767    | 0.7143 | 0.7442 |
| Safety Helmet  | 414 | 155 | 95  | 0.7276    | 0.8134 | 0.7681 |
| Safety Shoes   | 500 | 100 | 114 | 0.8333    | 0.8143 | 0.8237 |
| Safety Vest    | 205 | 63  | 24  | 0.7649    | 0.8952 | 0.8249 |

### Confusions entre classes

| Classe reelle  | Predite comme  | Occurrences |
|----------------|----------------|-------------|
| Safety Vest    | Safety Gloves  | 2           |
| Safety Vest    | Safety Harness | 2           |
| Safety Harness | Person         | 1           |
| Safety Vest    | Person         | 1           |
| Safety Shoes   | Safety Gloves  | 1           |
| Person         | Safety Harness | 1           |
| Person         | Safety Helmet  | 1           |
| Safety Harness | Safety Vest    | 1           |
| Safety Helmet  | Safety Vest    | 1           |
| Safety Shoes   | Safety Vest    | 1           |

### Pires exemples (a inspecter en priorite)

| Image                                                                                  | Reference | Predictions | VP | FP | FN | Score |
|----------------------------------------------------------------------------------------|-----------|-------------|----|----|----|-------|
| Construction-worker-wearing-safety-shoes-scaled-1_jpeg_jpg.rf.QSoK4kkwN4jlDtUxxl8y.jpg | 2         | 0           | 0  | 0  | 2  | 0.0   |
| construction_jpg.rf.SMFRMj2KMSOkXT1rBaBp.jpg                                           | 7         | 0           | 0  | 0  | 7  | 0.0   |
| full-body-harness-63-_jpg.rf.SJRgWmzNcAlu5RVSPt4a.jpg                                  | 1         | 3           | 0  | 3  | 1  | 0.0   |
| images (22)_jpg.rf.1zGj1VXyvSUDzNTtHBiB.jpg                                            | 2         | 3           | 0  | 3  | 1  | 0.0   |
| images (27)_jpg.rf.ElFcrFQLkv1GPeQBaZeR.jpg                                            | 1         | 3           | 0  | 3  | 1  | 0.0   |
| images (6)_jpg.rf.mtqUILPYBu7WhA6xKhRh.jpg                                             | 1         | 2           | 0  | 2  | 0  | 0.0   |
| istockphoto-1484904613-612x612_jpg.rf.cP0UMDjL5cIDNOyOiqSU.jpg                         | 2         | 0           | 0  | 0  | 2  | 0.0   |
| istockphoto-1484904613-612x612_jpg.rf.iB382fTOqdI27HmlZSkG.jpg                         | 2         | 0           | 0  | 0  | 2  | 0.0   |
| pos_1854_jpg.rf.ucHPyNN8jDiXvnuP9pmv.jpg                                               | 1         | 0           | 0  | 0  | 1  | 0.0   |
| pos_2083_jpg.rf.tNhPzNcZwCmusUSg6S2X.jpg                                               | 1         | 9           | 0  | 9  | 0  | 0.0   |

### Meilleurs exemples

| Image                                      | Reference | Predictions | VP | FP | FN | Score |
|--------------------------------------------|-----------|-------------|----|----|----|-------|
| -1034-_png_jpg.rf.A37nKPpsXjq6x6XWHXUk.jpg | 6         | 6           | 6  | 0  | 0  | 1.0   |
| -1116-_png_jpg.rf.eO6sjUYKarm5wVVBBDtj.jpg | 4         | 4           | 4  | 0  | 0  | 1.0   |
| -1145-_png_jpg.rf.GMuekIgV6POaTWvfHJ8C.jpg | 4         | 4           | 4  | 0  | 0  | 1.0   |
| -118-_png_jpg.rf.ZwATULKJbfwX96P3j5pE.jpg  | 3         | 3           | 3  | 0  | 0  | 1.0   |
| -1245-_png_jpg.rf.4g4OwqUdBmImS16jOBPa.jpg | 6         | 6           | 6  | 0  | 0  | 1.0   |
| -129-_png_jpg.rf.OetkxIZqgNVieXxJhrsA.jpg  | 7         | 7           | 7  | 0  | 0  | 1.0   |
| -1387-_png_jpg.rf.pkQrmlUcVQzzOEt4QiZW.jpg | 2         | 2           | 2  | 0  | 0  | 1.0   |
| -1546-_png_jpg.rf.rw7kg8Wtz5aUpZReLvVZ.jpg | 3         | 3           | 3  | 0  | 0  | 1.0   |
| -1899-_png_jpg.rf.rIlYrlLFbzNumEj07ph9.jpg | 2         | 2           | 2  | 0  | 0  | 1.0   |
| -1904-_png_jpg.rf.xdLpo5j2DhfEzN91xe9V.jpg | 7         | 7           | 7  | 0  | 0  | 1.0   |

## Limites connues

- Le split de test provient du meme export Roboflow que l'entrainement : l'audit a identifie des images issues d'une meme photo source reparties entre plusieurs splits, ce qui rend les metriques optimistes par rapport a un deploiement sur un chantier inconnu.
- Les metriques sont calculees sur des annotations dont 349 lignes ont ete converties depuis des polygones de segmentation : la boite englobante d'un polygone est systematiquement au moins aussi grande que l'objet reel.
- Confusion la plus frequente : 2 objet(s) de classe 'Safety Vest' predits comme 'Safety Gloves'.

> Les courbes et la matrice de confusion generees par Ultralytics se trouvent dans `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\runs\val\evaluation_test`.
