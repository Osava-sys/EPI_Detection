# Combler l'ecart entre les metriques et le terrain

## Le probleme, mesure

Le modele obtient **0.878 de mAP@0.50 sur `Safety Vest`** sur le split de test
assaini. Sur une video de chantier reelle recuperee sur Pexels, il n'a detecte
un gilet que **15 fois sur 122 images**, alors que la quasi-totalite des
ouvriers en portaient un.

Ce n'est pas une contradiction : les deux mesures portent sur des donnees
differentes. Le test provient du meme export Roboflow que l'entrainement, donc
de la meme distribution. La video, non.

C'est le principal risque du projet en l'etat : **les metriques actuelles ne
predisent pas la performance sur un chantier inconnu.**

## Ce que contient reellement le dataset

L'audit a mis en evidence deux populations d'images tres differentes :

| Population | Volume | Caracteristiques |
|------------|--------|------------------|
| Images web | ~5 200 | Photos de stock, plans rapproches, eclairage soigne, sujets poses |
| Frames video | ~1 785 | Extraites de quelques sequences, tres redondantes entre elles |

Les 1 785 frames video se nomment `frame_000324`, `frame_000325`... et forment
78 clusters de quasi-doublons qui **traversent encore les splits** apres le
regroupement par photo source. Le regroupement par nom ne peut rien y faire :
deux frames voisines sont, formellement, deux images sources distinctes.

Autrement dit, la diversite reelle du dataset est nettement inferieure a ses
7 000 images.

## Conditions absentes du dataset

Ce que le modele n'a jamais vu, et qu'il rencontrera en production :

- **Vues de videosurveillance** : camera en hauteur, angle plongeant, personnes
  petites dans le cadre. Le dataset est domine par des plans a hauteur d'homme.
- **Eclairage difficile** : contre-jour, aube et crepuscule, halogenes de
  chantier, zones d'ombre dure.
- **Intemperies** : pluie, poussiere, buee sur l'objectif, boue sur les EPI.
- **Variantes regionales d'EPI** : les gilets, casques et harnais different
  fortement d'un pays et d'un fournisseur a l'autre.
- **Occlusions partielles** : personnes derriere un echafaudage, un vehicule,
  une palette.
- **Mouvement** : flou de bouge, particulierement penalisant sur les petits
  objets deja fragiles (casques, gants).

## Plan de collecte

### Etape 1 — Definir la cible (avant toute annotation)

Collecter **dans les conditions exactes du deploiement** : memes cameras, meme
site, memes horaires. 500 images issues de votre site valent davantage que
5 000 images web supplementaires.

Si plusieurs sites sont vises, echantillonner chacun : un modele entraine sur un
seul site generalise mal aux autres.

### Etape 2 — Echantillonner intelligemment

Ne pas extraire une frame toutes les secondes : la redondance est le defaut
principal du dataset actuel. Deux approches complementaires :

- **Echantillonnage temporel espace** : une frame toutes les 30 a 60 secondes,
  pour maximiser la diversite de scenes a volume constant.
- **Apprentissage actif** : faire tourner le modele actuel sur les images
  candidates et annoter en priorite celles ou il hesite (detections entre 0.3 et
  0.6 de confiance) ou echoue. Ce sont elles qui apportent de l'information.

La commande suivante produit les predictions exploitables pour ce tri :

```powershell
python -m ppe_detection.predict --weights artifacts/models/best.pt `
  --source chemin\vers\images_candidates --conf 0.20 --save-json `
  --name selection_active
```

### Etape 3 — Volume et repartition

| Objectif | Images annotees | Effet attendu |
|----------|-----------------|---------------|
| Validation du domaine | 150 - 300 | Mesurer honnetement l'ecart, sans encore le combler |
| Amelioration mesurable | 500 - 1 000 | Gain net sur le site cible |
| Robustesse multi-sites | 2 000+ | Generalisation au-dela du site d'origine |

Priorite absolue aux classes faibles : `Safety Gloves` (0.55 de mAP@0.50) et
`Safety Helmet` en petite taille (25 % des casques font moins de 32 px a
640 px).

### Etape 4 — Constituer un jeu de test terrain separe

**C'est l'etape la plus importante, et la plus souvent negligee.**

Reserver 150 a 200 images de terrain qui n'entrent **jamais** dans
l'entrainement. Elles constituent le seul juge honnete de la performance reelle.
Les garder dans un repertoire distinct :

```
artifacts/dataset_terrain/
├── data.yaml
├── train/          # ajoute au dataset principal
└── test/           # JAMAIS fusionne — mesure de reference
```

Evaluer les deux jeux a chaque iteration :

```powershell
python -m ppe_detection.evaluate --weights artifacts/models/best.pt `
  --data artifacts/dataset_detection/data.yaml --split test --name eval_roboflow
python -m ppe_detection.evaluate --weights artifacts/models/best.pt `
  --data artifacts/dataset_terrain/data.yaml --split test --name eval_terrain
```

L'ecart entre les deux chiffres **est** la mesure de l'ecart au terrain. Le
suivre dans le temps vaut mieux que d'optimiser le seul score Roboflow.

### Etape 5 — Integrer sans casser le protocole

Le pipeline existant absorbe les nouvelles images sans modification :

```powershell
# 1. Auditer les nouvelles annotations
python -m ppe_detection.dataset_audit --data chemin\nouveau\data.yaml `
  --output artifacts/reports/audit_terrain.json

# 2. Normaliser (le regroupement anti-fuite est applique par defaut)
python -m ppe_detection.dataset_cleaner --source chemin\nouveau\data.yaml `
  --output artifacts/dataset_terrain

# 3. Reentrainer en repartant des poids actuels plutot que de zero
python -m ppe_detection.train --config configs/train.yaml `
  --model artifacts/models/best.pt --name ppe_terrain_v1
```

Partir de `best.pt` conserve ce que le modele a deja appris et converge bien
plus vite que depuis les poids COCO.

## Consignes d'annotation

L'incoherence entre annotateurs plafonne la performance atteignable. A fixer
**avant** de commencer :

- **Gilet** : annoter le vetement visible, ou la personne entiere si le gilet
  est majoritairement masque ?
- **Casque** : la coque seule, ou coque + jugulaire ?
- **Chaussures** : une boite par pied, ou une pour les deux ?
- **Personne partiellement visible** : a partir de quelle proportion visible
  annote-t-on ?
- **Occlusion** : annoter la partie visible, ou la boite complete extrapolee ?

Le dataset actuel melange deja des boites natives et **349 lignes converties
depuis des polygones de segmentation**. Une boite issue d'un polygone est
systematiquement au moins aussi grande que l'objet reel, ce qui contribue au
faible mAP@0.50:0.95 (0.433 contre 0.799 a IoU 0.50).

## Ce qu'il ne faut pas attendre de cette etape

Ajouter des donnees de terrain ne corrigera pas :

- la **taille des objets** : un casque a 20 px reste indetectable, quelle que
  soit la quantite de donnees. C'est le sujet de la resolution d'entrainement.
- les **erreurs d'annotation** heritees du dataset Roboflow d'origine.
- l'heuristique d'**association EPI/personne**, qui releve de la couche metier
  et non du detecteur.

## Ordre recommande

1. Constituer d'abord le **jeu de test terrain** (150-200 images). Sans lui, on
   optimise a l'aveugle.
2. Mesurer l'ecart reel. Il sera probablement important sur `Safety Vest`.
3. Puis seulement collecter les donnees d'entrainement, guidees par les erreurs
   effectivement constatees.

Mesurer avant d'optimiser evite d'annoter massivement des cas qui n'etaient pas
le probleme.
