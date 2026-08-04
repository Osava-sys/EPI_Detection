# Rapport d'audit du dataset EPI

- **Genere le** : 2026-08-04T11:08:44.610213+00:00
- **data.yaml** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\dataset_detection\data.yaml`
- **Classes (7)** : Face Mask, Person, Safety Gloves, Safety Harness, Safety Helmet, Safety Shoes, Safety Vest
- **Splits trouves** : test, train, valid

> **Verdict : PRET POUR LA DETECTION** — 7000 images, 25542 annotations, 0 ligne(s) polygonale(s) a convertir, 0 erreur(s), 4 avertissement(s), 0 groupe(s) de doublons exacts.

## 1. Vue d'ensemble par split

| Split     | Images | Labels | Annot. | Detection | Polygone | Malformees | Appariement | Taille   |
|-----------|--------|--------|--------|-----------|----------|------------|-------------|----------|
| train     | 5145   | 5145   | 18552  | 18552     | 0        | 0          | oui         | 620.0 Mo |
| valid     | 1277   | 1277   | 4848   | 4848      | 0        | 0          | oui         | 161.5 Mo |
| test      | 578    | 578    | 2142   | 2142      | 0        | 0          | oui         | 75.7 Mo  |
| **TOTAL** | 7000   | 7000   | 25542  | 25542     | 0        | 0          | oui         |          |

## 2. Distribution des classes

| Classe         | ID | train | valid | test | Total | Part (%) |
|----------------|----|-------|-------|------|-------|----------|
| Face Mask      | 0  | 596   | 131   | 61   | 788   | 3.09     |
| Person         | 1  | 5435  | 1500  | 714  | 7649  | 29.95    |
| Safety Gloves  | 2  | 1625  | 410   | 137  | 2172  | 8.50     |
| Safety Harness | 3  | 815   | 249   | 111  | 1175  | 4.60     |
| Safety Helmet  | 4  | 3842  | 1112  | 495  | 5449  | 21.33    |
| Safety Shoes   | 5  | 4487  | 992   | 396  | 5875  | 23.00    |
| Safety Vest    | 6  | 1752  | 454   | 228  | 2434  | 9.53     |

- Classe majoritaire : **Person**
- Classe minoritaire : **Face Mask**
- Ratio max/min : **9.71**

## 3. Integrite des images

| Split | Extensions                   | Illisibles | Resolutions | Min   | Max       | Ratio L/H   |
|-------|------------------------------|------------|-------------|-------|-----------|-------------|
| train | .jpeg:8, .jpg:5033, .png:104 | 0          | 255         | 62x85 | 2160x3840 | 0.45 - 2.80 |
| valid | .jpeg:1, .jpg:1243, .png:33  | 0          | 103         | 55x87 | 5178x3884 | 0.45 - 2.07 |
| test  | .jpg:560, .png:18            | 0          | 72          | 64x82 | 2160x3840 | 0.45 - 2.15 |

## 4. Statistiques geometriques des boites

| Split | Boites | Aire min   | p05      | Mediane  | p95      | Aire max | Petits objets (<1 %) |
|-------|--------|------------|----------|----------|----------|----------|----------------------|
| train | 18552  | 4.307e-05  | 0.001748 | 0.021778 | 0.375784 | 1.0      | 35.51 %              |
| valid | 4848   | 1.157e-05  | 0.001629 | 0.021707 | 0.386156 | 0.97051  | 34.86 %              |
| test  | 2142   | 0.00026109 | 0.001366 | 0.021538 | 0.405535 | 0.970368 | 34.55 %              |

## 5. Doublons et risques de fuite

| Indicateur                                    | Valeur |
|-----------------------------------------------|--------|
| Groupes de doublons binaires exacts           | 0      |
| Fichiers en trop (doublons exacts)            | 0      |
| Groupes de doublons exacts inter-splits       | 0      |
| Hash perceptuel disponible                    | non    |
| Clusters quasi identiques (Hamming <= 3)      | 0      |
| Images concernees                             | 0      |
| Plus grand cluster                            | 0      |
| Clusters s'etendant sur plusieurs splits      | 0      |
| Images dans ces clusters inter-splits         | 0      |
| Images source Roboflow partagees entre splits | 0      |
| Fichiers concernes                            | 0      |
| Sequences numerotees reparties entre splits   | 19     |
| Images de ces sequences                       | 3109   |

> Le nombre brut de *paires* quasi identiques (0) n'est pas un bon indicateur : un groupe de N images similaires produit N*(N-1)/2 paires. Les clusters ci-dessus refletent la realite.

Principales sequences reparties entre splits :

| Prefixe                    | Images | Splits             |
|----------------------------|--------|--------------------|
| frame_                     | 1785   | test, train, valid |
| pos_                       | 582    | test, train, valid |
| image_                     | 355    | test, train, valid |
| S2-N2301M                  | 97     | test, train, valid |
| front_crawling_            | 68     | test, train, valid |
| ppe_                       | 68     | test, train, valid |
| with_mask_                 | 65     | test, train, valid |
| Manoj_Herness2_annotation_ | 33     | test, train, valid |
| img_                       | 11     | test, train, valid |
| Onsite PhotoVideo_         | 8      | train, valid       |

## 6. Problemes detectes

| Severite | Code                          | Description                                                                                                                                                                |
|----------|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AVERT.   | clamped_minor                 | [train] 254 boite(s) corrigee(s) : clamped_minor.                                                                                                                          |
| AVERT.   | clamped_minor                 | [valid] 66 boite(s) corrigee(s) : clamped_minor.                                                                                                                           |
| AVERT.   | clamped_minor                 | [test] 18 boite(s) corrigee(s) : clamped_minor.                                                                                                                            |
| AVERT.   | video_sequences_across_splits | 19 sequence(s) d'images numerotees (3109 images de type 'frame_000324') reparties entre plusieurs splits : les frames consecutives d'une meme video sont quasi identiques. |

## 7. Detail des codes d'anomalie par split

| Code          | train | valid | test |
|---------------|-------|-------|------|
| clamped_minor | 254   | 66    | 18   |
| tiny_box      | 3     | 1     | 0    |

## 8. Graphiques et exemples

### Frequence des classes par split

![Frequence des classes par split](dataset_audit_detection_assets/class_distribution.png)

### Distribution des tailles de boites

![Distribution des tailles de boites](dataset_audit_detection_assets/box_size_distribution.png)

### Exemples d'annotations (verification visuelle)

![Exemples d'annotations (verification visuelle)](dataset_audit_detection_assets/annotation_samples.png)

## Glossaire des codes

| Code                   | Signification                                                        |
|------------------------|----------------------------------------------------------------------|
| polygon_lines          | Ligne de segmentation (class_id + paires x/y). A convertir en boite. |
| converted_from_polygon | Boite obtenue par min/max sur les sommets d'un polygone.             |
| clamped_minor          | Derive numerique <= tolerance ramenee dans [0, 1].                   |
| clipped_to_bounds      | Objet partiellement hors cadre : boite rognee sur l'image.           |
| class_id_out_of_range  | Identifiant de classe absent de data.yaml.                           |
| non_numeric            | Champ non convertible en nombre.                                     |
| non_positive_size      | Largeur ou hauteur nulle/negative.                                   |
| center_out_of_range    | Centre de la boite hors de l'image.                                  |
| bad_field_count        | Nombre de champs incompatible avec detection ou polygone.            |
| very_small_box         | Boite d'aire normalisee < 1e-5.                                      |
