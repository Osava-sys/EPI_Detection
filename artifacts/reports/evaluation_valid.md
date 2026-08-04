# Rapport d'evaluation — split `valid`

- **Genere le** : 2026-08-04T11:10:10.932333+00:00
- **Poids** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\models\best.pt` (19.4 Mo)
- **Parametres** : 9467889
- **Dataset** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\dataset_detection\data.yaml`
- **Parametres d'evaluation** : imgsz=640, batch=16, conf=0.001, iou=0.6, device=0

## Metriques globales

| Metrique            | Valeur |
|---------------------|--------|
| Precision (moyenne) | 0.7855 |
| Rappel (moyen)      | 0.7333 |
| **mAP@0.50**        | 0.7789 |
| **mAP@0.50:0.95**   | 0.4294 |
| mAP@0.75            | 0.4176 |

## Metriques par classe

| Classe         | Precision | Rappel | mAP@0.50 | mAP@0.50:0.95 |
|----------------|-----------|--------|----------|---------------|
| Face Mask      | 0.8462    | 0.7863 | 0.8733   | 0.5694        |
| Person         | 0.867     | 0.8906 | 0.8981   | 0.535         |
| Safety Gloves  | 0.6826    | 0.5244 | 0.5217   | 0.2354        |
| Safety Harness | 0.7705    | 0.6145 | 0.7309   | 0.3753        |
| Safety Helmet  | 0.7764    | 0.7962 | 0.8212   | 0.365         |
| Safety Shoes   | 0.7487    | 0.6905 | 0.7409   | 0.4019        |
| Safety Vest    | 0.8069    | 0.8304 | 0.8664   | 0.5236        |

## Vitesse

| Etape       | Temps par image |
|-------------|-----------------|
| preprocess  | 0.725 ms        |
| inference   | 3.626 ms        |
| loss        | 0.0 ms          |
| postprocess | 0.097 ms        |
| **Debit**   | 224.82 images/s |

## Analyse d'erreurs

Realisee au seuil operationnel `conf=0.25` et `IoU>=0.5` sur 1277 image(s).

| Indicateur     | Total |
|----------------|-------|
| Vrais positifs | 3821  |
| Faux positifs  | 1209  |
| Faux negatifs  | 1027  |

### Detail par classe

| Classe         | VP   | FP  | FN  | Precision | Rappel | F1     |
|----------------|------|-----|-----|-----------|--------|--------|
| Face Mask      | 112  | 30  | 19  | 0.7887    | 0.855  | 0.8205 |
| Person         | 1347 | 290 | 153 | 0.8228    | 0.898  | 0.8588 |
| Safety Gloves  | 216  | 108 | 194 | 0.6667    | 0.5268 | 0.5886 |
| Safety Harness | 154  | 72  | 95  | 0.6814    | 0.6185 | 0.6484 |
| Safety Helmet  | 907  | 333 | 205 | 0.7315    | 0.8156 | 0.7713 |
| Safety Shoes   | 700  | 265 | 292 | 0.7254    | 0.7056 | 0.7154 |
| Safety Vest    | 385  | 111 | 69  | 0.7762    | 0.848  | 0.8105 |

### Confusions entre classes

| Classe reelle  | Predite comme  | Occurrences |
|----------------|----------------|-------------|
| Safety Gloves  | Safety Shoes   | 11          |
| Safety Harness | Safety Vest    | 9           |
| Safety Vest    | Safety Harness | 4           |
| Safety Gloves  | Person         | 3           |
| Safety Vest    | Person         | 3           |
| Safety Shoes   | Safety Gloves  | 3           |
| Person         | Safety Harness | 2           |
| Person         | Face Mask      | 1           |
| Safety Helmet  | Face Mask      | 1           |
| Safety Harness | Person         | 1           |
| Safety Helmet  | Person         | 1           |
| Safety Shoes   | Person         | 1           |
| Person         | Safety Gloves  | 1           |

### Pires exemples (a inspecter en priorite)

| Image                                                                                             | Reference | Predictions | VP | FP | FN | Score |
|---------------------------------------------------------------------------------------------------|-----------|-------------|----|----|----|-------|
| -1325-_png_jpg.rf.PcHQ6a6K2g70N4JRCIry.jpg                                                        | 3         | 0           | 0  | 0  | 3  | 0.0   |
| 6e2c756d74b3282c2bcc0bab4a0ce9b7-jpg_1000x1000q80-jpg_320x32_jpg.rf.1wJZy9wEGKSPOZeeH7Oc.jpg      | 1         | 0           | 0  | 0  | 1  | 0.0   |
| 6e2c756d74b3282c2bcc0bab4a0ce9b7-jpg_1000x1000q80-jpg_320x32_jpg.rf.aEIM0Pcjkve83Zlkuu9Z.jpg      | 1         | 0           | 0  | 0  | 1  | 0.0   |
| 6e2c756d74b3282c2bcc0bab4a0ce9b7-jpg_1000x1000q80-jpg_320x32_jpg.rf.uA7HEf5XqP8QqPFQfrQI.jpg      | 1         | 0           | 0  | 0  | 1  | 0.0   |
| Aitin2742_jpg.rf.9OZ3emMERwyS84EI7UCB.jpg                                                         | 1         | 0           | 0  | 0  | 1  | 0.0   |
| Aitin2742_jpg.rf.EUNPqNW6OI50O9MzSZVi.jpg                                                         | 3         | 0           | 0  | 0  | 3  | 0.0   |
| Aitin2742_jpg.rf.KXfWIkfAhBcc8EI38Wx9.jpg                                                         | 2         | 2           | 0  | 2  | 2  | 0.0   |
| Co-abA8QiKYSZQ9T_FzA_QIcoAxxysFc6lAm6gSnZ6SVGggvVn9bMoOsnCqB_jpeg_jpg.rf.eeXq9UqR8VhRaeIHKOhf.jpg | 1         | 1           | 0  | 1  | 0  | 0.0   |
| Screenshot 2026-06-06 003654_jpg.rf.83G9F8Pj5G1BFV86Pxiw.jpg                                      | 1         | 2           | 0  | 2  | 0  | 0.0   |
| attractive-man-work-clothes-shoes-26075194-1-_jpg.rf.RiA5doMafAzkF6uvQ9IJ.jpg                     | 1         | 2           | 0  | 2  | 1  | 0.0   |

### Meilleurs exemples

| Image                                      | Reference | Predictions | VP | FP | FN | Score |
|--------------------------------------------|-----------|-------------|----|----|----|-------|
| -1040-_png_jpg.rf.fOHJ11n4UdtQWnemGX1b.jpg | 2         | 2           | 2  | 0  | 0  | 1.0   |
| -119-_png_jpg.rf.Eghj34ecjSy9yLnwYmMB.jpg  | 2         | 2           | 2  | 0  | 0  | 1.0   |
| -1257-_png_jpg.rf.VAXGqcZrmlLeRPH9Xnq3.jpg | 6         | 6           | 6  | 0  | 0  | 1.0   |
| -1605-_png_jpg.rf.FGuFfUx7aODP88WMrNYw.jpg | 2         | 2           | 2  | 0  | 0  | 1.0   |
| -168-_png_jpg.rf.pUODXOb0MmChN3ZXJ3vR.jpg  | 3         | 3           | 3  | 0  | 0  | 1.0   |
| -1724-_png_jpg.rf.F2disVXGzwPja7mySokq.jpg | 4         | 4           | 4  | 0  | 0  | 1.0   |
| -1795-_png_jpg.rf.IsIsfY26oGBiPkVNdZIQ.jpg | 6         | 6           | 6  | 0  | 0  | 1.0   |
| -1942-_png_jpg.rf.O6jKKwgF9ikAnC0YSEd5.jpg | 14        | 14          | 14 | 0  | 0  | 1.0   |
| -2012-_png_jpg.rf.HLiHDZfp8V4Onm0xQw3b.jpg | 2         | 2           | 2  | 0  | 0  | 1.0   |
| -216-_png_jpg.rf.5JeCZKWGFmG4Yzvk0qBc.jpg  | 3         | 3           | 3  | 0  | 0  | 1.0   |

## Limites connues

- Le split de test provient du meme export Roboflow que l'entrainement : l'audit a identifie des images issues d'une meme photo source reparties entre plusieurs splits, ce qui rend les metriques optimistes par rapport a un deploiement sur un chantier inconnu.
- Les metriques sont calculees sur des annotations dont 349 lignes ont ete converties depuis des polygones de segmentation : la boite englobante d'un polygone est systematiquement au moins aussi grande que l'objet reel.
- Confusion la plus frequente : 11 objet(s) de classe 'Safety Gloves' predits comme 'Safety Shoes'.

> Les courbes et la matrice de confusion generees par Ultralytics se trouvent dans `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\runs\val\evaluation_valid`.
