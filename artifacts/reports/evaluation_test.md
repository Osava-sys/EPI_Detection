# Rapport d'evaluation — split `test`

- **Genere le** : 2026-08-04T15:11:18.700074+00:00
- **Poids** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\models\best.pt` (19.4 Mo)
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
| preprocess  | 0.705 ms        |
| inference   | 2.732 ms        |
| loss        | 0.006 ms        |
| postprocess | 0.164 ms        |
| **Debit**   | 277.24 images/s |

## Analyse d'erreurs

Realisee au seuil operationnel `conf=0.25` et `IoU>=0.5` sur 578 image(s).

| Indicateur     | Total |
|----------------|-------|
| Vrais positifs | 1734  |
| Faux positifs  | 486   |
| Faux negatifs  | 408   |

### Detail par classe

| Classe         | VP  | FP  | FN  | Precision | Rappel | F1     |
|----------------|-----|-----|-----|-----------|--------|--------|
| Face Mask      | 55  | 12  | 6   | 0.8209    | 0.9016 | 0.8594 |
| Person         | 643 | 133 | 71  | 0.8286    | 0.9006 | 0.8631 |
| Safety Gloves  | 78  | 44  | 59  | 0.6393    | 0.5693 | 0.6023 |
| Safety Harness | 79  | 23  | 32  | 0.7745    | 0.7117 | 0.7418 |
| Safety Helmet  | 401 | 148 | 94  | 0.7304    | 0.8101 | 0.7682 |
| Safety Shoes   | 278 | 73  | 118 | 0.792     | 0.702  | 0.7443 |
| Safety Vest    | 200 | 53  | 28  | 0.7905    | 0.8772 | 0.8316 |

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

| Image                                                                                             | Reference | Predictions | VP | FP | FN | Score |
|---------------------------------------------------------------------------------------------------|-----------|-------------|----|----|----|-------|
| Construction-worker-wearing-safety-shoes-scaled-1_jpeg_jpg.rf.QSoK4kkwN4jlDtUxxl8y.jpg            | 2         | 0           | 0  | 0  | 2  | 0.0   |
| Construction-worker-wearing-safety-shoes-scaled-1_jpeg_jpg.rf.Z88wKwYsIgJPMvg56sWB.jpg            | 3         | 0           | 0  | 0  | 3  | 0.0   |
| Construction-worker-wearing-safety-shoes-scaled-1_jpeg_jpg.rf.mL1277Me2sg2oLKmUCRF.jpg            | 3         | 0           | 0  | 0  | 3  | 0.0   |
| construction_jpg.rf.SMFRMj2KMSOkXT1rBaBp.jpg                                                      | 7         | 0           | 0  | 0  | 7  | 0.0   |
| dyRJH7cAFsizFZa0b455i81Kf9x48TXPtM7ApZBBSS4s2A26q3zUCnE9jMEK_jpeg_jpg.rf.Sv7mCwrlQT2axIjgcJBc.jpg | 1         | 2           | 0  | 2  | 1  | 0.0   |
| full-body-harness-63-_jpg.rf.SJRgWmzNcAlu5RVSPt4a.jpg                                             | 1         | 3           | 0  | 3  | 1  | 0.0   |
| images (22)_jpg.rf.1zGj1VXyvSUDzNTtHBiB.jpg                                                       | 2         | 3           | 0  | 3  | 1  | 0.0   |
| images (27)_jpg.rf.ElFcrFQLkv1GPeQBaZeR.jpg                                                       | 1         | 3           | 0  | 3  | 1  | 0.0   |
| images (6)_jpg.rf.mtqUILPYBu7WhA6xKhRh.jpg                                                        | 1         | 2           | 0  | 2  | 0  | 0.0   |
| istockphoto-1484904613-612x612_jpg.rf.RjASIHlOnAC2hag8uloA.jpg                                    | 2         | 0           | 0  | 0  | 2  | 0.0   |

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
