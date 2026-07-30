# Rapport d'export — format `onnx`

- **Genere le** : 2026-07-30T10:18:56.388352+00:00
- **Poids source** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\models\smoke_best.pt` (19.4 Mo)
- **Fichier produit** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\exports\smoke_best.onnx` (36.4 Mo)
- **Duree** : 11.77 s

## Parametres d'export

| Parametre | Valeur |
|-----------|--------|
| imgsz     | 640    |
| simplify  | True   |
| opset     | None   |
| dynamic   | False  |
| half      | False  |
| device    | cpu    |

## Verification

**Resultat : REUSSIE** — le modele exporte a reellement ete execute.

| Controle            | Resultat |
|---------------------|----------|
| file_exists         | True     |
| file_size_bytes     | 38177215 |
| file_size_human     | 36.4 Mo  |
| onnx_checker        | True     |
| ir_version          | 9        |
| onnxruntime_session | True     |
| dummy_inference     | True     |
| end_to_end_output   | True     |

### Entrees / sorties du graphe

| Sens   | Nom     | Forme            | Type          |
|--------|---------|------------------|---------------|
| entree | images  | [1, 3, 640, 640] | tensor(float) |
| sortie | output0 | [1, 300, 6]      | tensor(float) |

### Comparaison fonctionnelle PyTorch vs ONNX Runtime

Les deux moteurs ont traite la meme image reelle (`-1015-_png_jpg.rf.VPTTTZgPrWtCLZ5u6Baw.jpg`) au seuil `conf=0.25`.

| Indicateur                 | Valeur   |
|----------------------------|----------|
| Detections PyTorch         | 9        |
| Detections ONNX            | 9        |
| Detections appariees       | 9        |
| Classes divergentes        | 0        |
| IoU moyen                  | 0.999999 |
| IoU minimal                | 0.999999 |
| Ecart max de confiance     | 6e-06    |
| Decalage max de boite (px) | 0.0002   |
| **Concordance**            | oui      |

> Ce modele s'exporte **end-to-end** : la sortie ONNX `(1, N, 6)` contient deja des detections filtrees et triees par confiance. Une comparaison terme a terme des tenseurs bruts serait sans signification (un ecart numerique infime reordonne les lignes), d'ou la comparaison des detections effectivement produites.

## Differences de post-traitement a connaitre

- Le modele exporte attend une image normalisee dans [0, 1], au format NCHW (1x3xHxW), en RGB, redimensionnee avec letterbox vers la taille figee a l'export.
- La sortie brute n'est pas filtree de la meme maniere que l'API Python : selon le format, la suppression des non-maxima (NMS) et le decodage des boites peuvent devoir etre reimplementes cote client.
- Les coordonnees produites se rapportent a l'image redimensionnee : il faut annuler le letterbox (echelle et decalage) pour revenir aux coordonnees de l'image d'origine.
- L'ecart numerique entre PyTorch et ONNX Runtime est normalement de l'ordre de 1e-4 a 1e-3 : il provient des noyaux de convolution differents entre les deux moteurs, pas d'une erreur de conversion.
