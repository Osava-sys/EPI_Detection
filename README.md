# Detection d'equipements de protection individuelle (EPI)

Projet complet de detection d'objets pour la securite au travail : audit du
dataset, normalisation des annotations, entrainement, evaluation, inference
(image / dossier / video / webcam), logique de conformite, API REST, interface
locale et export ONNX.

Construit avec **Ultralytics YOLO26** et **PyTorch**, testé sous **Windows 11**
avec une **NVIDIA RTX 5080 Laptop (16 Go)**.

---

## Table des matieres

1. [Presentation](#1-presentation)
2. [Architecture](#2-architecture)
3. [Dataset et licence](#3-dataset-et-licence)
4. [Prerequis](#4-prerequis)
5. [Installation](#5-installation)
6. [Audit du dataset](#6-audit-du-dataset)
7. [Conversion des annotations](#7-conversion-des-annotations)
8. [Smoke test](#8-smoke-test)
9. [Entrainement complet](#9-entrainement-complet)
10. [Evaluation](#10-evaluation)
11. [Inference](#11-inference)
12. [Conformite EPI](#12-conformite-epi)
13. [API REST](#13-api-rest)
14. [Interface Streamlit](#14-interface-streamlit)
15. [Export ONNX](#15-export-onnx)
16. [Structure des sorties](#16-structure-des-sorties)
17. [Qualite du code et tests](#17-qualite-du-code-et-tests)
18. [Depannage](#18-depannage)
19. [Limites connues](#19-limites-connues)
20. [Pistes d&#39;amelioration](#20-pistes-damelioration)

---

## 1. Presentation

Le systeme detecte **7 classes** d'equipements et de personnes sur des images
de chantier ou de site industriel :

| ID | Classe         | Instances | Part    |
| -- | -------------- | --------- | ------- |
| 0  | Face Mask      | 788       | 3,09 %  |
| 1  | Person         | 7 649     | 29,95 % |
| 2  | Safety Gloves  | 2 172     | 8,50 %  |
| 3  | Safety Harness | 1 175     | 4,60 %  |
| 4  | Safety Helmet  | 5 449     | 21,33 % |
| 5  | Safety Shoes   | 5 875     | 23,00 % |
| 6  | Safety Vest    | 2 434     | 9,53 %  |

Une couche metier **optionnelle** associe ensuite les EPI aux personnes
detectees pour produire un statut de conformite. Cette association est une
heuristique geometrique, pas une mesure certaine : voir la
[section 12](#12-conformite-epi).

### Decisions d'ingenierie notables

| Sujet                | Decision                                     | Raison                                                                                                                                                                                       |
| -------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Modele               | `yolo26s.pt` (Ultralytics 8.4)             | Roboflow annonce un export « YOLO26 » : verification faite,**YOLO26 existe reellement** dans Ultralytics 8.4 (`yolo26n/s/m/l/x`). Les annotations restent du YOLO standard.        |
| Python               | 3.12 dans un venv dedie                      | Python 3.14 est installe globalement, mais PyTorch ne publie pas encore de roues pour cette version.                                                                                         |
| PyTorch              | `2.9.1+cu128`                              | La RTX 5080 est une puce**Blackwell (sm_120)**. Les builds CUDA anterieures a 12.8 ne contiennent aucun noyau pour cette architecture. Verifie via `torch.cuda.get_arch_list()`.     |
| Polygones            | Conversion en boites dans une**copie** | 349 lignes de segmentation coexistent avec les boites. Le dataset original n'est jamais modifie.                                                                                             |
| Chemins`data.yaml` | Resolution multi-candidats                   | L'export Roboflow ecrit`../train/images`, qui ne se resout pas correctement depuis la racine du dataset. Le code teste plusieurs interpretations et retient celle qui existe.              |
| Verification ONNX    | Comparaison**fonctionnelle**           | YOLO26 s'exporte « end-to-end » (sortie`(1, 300, 6)` deja filtree). Comparer les tenseurs bruts terme a terme n'a pas de sens : on compare les detections produites sur une vraie image. |

---

## 2. Architecture

```
.
├── data.yaml                     # Export Roboflow original (jamais modifie)
├── train/ valid/ test/           # Images et labels originaux (intacts)
│
├── configs/
│   ├── train.yaml                # Hyperparametres d'entrainement
│   └── inference.yaml            # Seuils d'inference + regles de conformite
│
├── src/ppe_detection/
│   ├── annotations.py            # Parsing/validation/conversion des labels (sans dependance lourde)
│   ├── config.py                 # Dataclasses de configuration typees
│   ├── utils.py                  # Logging, seed, device, E/S, securite des noms de fichiers
│   ├── dataset_audit.py          # Audit complet du dataset
│   ├── dataset_cleaner.py        # Construction du dataset de detection normalise
│   ├── train.py                  # Entrainement
│   ├── evaluate.py               # Evaluation + analyse d'erreurs
│   ├── calibrate.py              # Calibration des seuils par classe (validation)
│   ├── pose.py                   # Association EPI/personne par points cles
│   ├── predict.py                # Inference unifiee (image/dossier/video/webcam/flux)
│   ├── video.py                  # Boucle video et webcam
│   ├── compliance.py             # Association geometrique EPI <-> personne
│   ├── visualization.py          # Rendu des detections et graphiques
│   ├── export.py                 # Export ONNX + verification reelle
│   └── api.py                    # API REST FastAPI
│
├── app/streamlit_app.py          # Interface locale
├── docs/plan_ecart_terrain.md    # Plan de collecte pour le deploiement reel
├── scripts/*.ps1                 # Scripts PowerShell de bout en bout
├── tests/                        # 116 tests unitaires et d'integration
└── artifacts/                    # Sorties generees (hors Git)
    ├── dataset_detection/        # Dataset normalise (labels 5 champs)
    ├── models/                   # best.pt, last.pt
    ├── runs/                     # Runs Ultralytics
    ├── reports/                  # Rapports JSON + Markdown
    ├── predictions/              # Resultats d'inference
    └── exports/                  # Modeles exportes
```

---

## 3. Dataset et licence

- **Source** : [Roboflow Universe — PPE Detection Project](https://universe.roboflow.com/ousmane-savadogo/ppe-detection-project-jeezl-p9ncg)
- **Licence** : **CC BY 4.0** — reutilisation permise avec attribution.
- **Export** : 30 juillet 2026, format annonce « YOLO26 ».
- **Volume** : 7 000 images, 25 542 annotations, 857 Mo.

| Split | Images | Labels | Annotations |
| ----- | ------ | ------ | ----------- |
| train | 4 903  | 4 903  | 17 873      |
| valid | 1 399  | 1 399  | 5 197       |
| test  | 698    | 698    | 2 472       |

L'appariement image ↔ label est **parfait** : aucune image orpheline, aucun
label sans image.

---

## 4. Prerequis

- **Windows 10/11** avec PowerShell (le code reste portable Linux/macOS).
- **Python 3.10 a 3.13** (3.12 recommande). Python 3.14 n'est pas encore
  supporte par PyTorch.
- **GPU NVIDIA** optionnel mais fortement recommande. Le CPU fonctionne mais
  l'entrainement complet y serait deraisonnablement long.
- ~10 Go d'espace disque (dataset original + copie normalisee + poids).

Environnement de reference valide :

```
Python        3.12.10
torch         2.9.1+cu128
ultralytics   8.4.112
opencv        5.0.0
onnxruntime   1.28.0
GPU           NVIDIA GeForce RTX 5080 Laptop (16 Go, sm_120)
Pilote        596.36 (CUDA 13.2)
```

---

## 5. Installation

### Installation automatique (recommandee)

```powershell
.\scripts\setup.ps1
```

Le script cree le venv, detecte le GPU, installe la variante PyTorch adaptee,
installe le projet, puis **verifie que la build PyTorch contient bien des
noyaux pour votre GPU**.

Variantes :

```powershell
.\scripts\setup.ps1 -Cuda cpu                    # machine sans GPU
.\scripts\setup.ps1 -PythonVersion 3.11 -Force   # autre version, recreation
```

### Installation manuelle

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**GPU (Blackwell / RTX 50xx — CUDA 12.8) :**

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**GPU (Ampere / Ada — RTX 30xx, 40xx) :**

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**CPU uniquement :**

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Puis le projet :

```powershell
python -m pip install -e ".[api,ui,export,audit,dev]"
```

Verification :

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

---

## 6. Audit du dataset

```powershell
python -m ppe_detection.dataset_audit --data data.yaml --output artifacts/reports/dataset_audit_original.json
```

Options utiles :

```powershell
# Audit rapide, sans recherche de quasi-doublons visuels
python -m ppe_detection.dataset_audit --data data.yaml --output artifacts/reports/audit.json --skip-perceptual-hash

# Echoue si des erreurs bloquantes ou des polygones subsistent (utile en CI)
python -m ppe_detection.dataset_audit --data data.yaml --output artifacts/reports/audit.json --fail-on-error --fail-on-polygon
```

L'audit verifie : existence des splits, validite du `data.yaml`, appariement
image/label, extensions, images corrompues, dimensions et rapports d'aspect,
labels vides, identifiants de classes, valeurs non numeriques, coordonnees hors
`[0, 1]`, tailles nulles ou negatives, boites debordant du cadre, boites
minuscules, lignes a 5 champs, lignes polygonales, distribution par classe,
desequilibre, doublons binaires, quasi-doublons perceptuels et fuites entre
splits.

### Resultats sur ce dataset

| Constat                                     | Valeur                                       |
| ------------------------------------------- | -------------------------------------------- |
| Images / annotations                        | 7 000 / 25 542                               |
| Appariement image ↔ label                  | parfait sur les 3 splits                     |
| Images illisibles ou corrompues             | 0                                            |
| Lignes malformees                           | 0                                            |
| **Lignes polygonales (segmentation)** | **349** (285 train, 46 valid, 18 test) |
| Derives numeriques infimes corrigees        | 457                                          |
| Doublons binaires exacts                    | 0                                            |
| Desequilibre (max/min)                      | **9,71** (Person vs Face Mask)         |
| Petits objets (aire < 1 % de l'image)       | ~35 % des boites                             |
| Resolutions distinctes                      | 256 (train), de 55×87 a 5178×3884          |

> Les 349 lignes polygonales ne sont **pas** comptees comme des erreurs :
> ce sont des annotations de segmentation valides qui doivent etre converties
> en boites englobantes pour une tache de detection.

### Fuite entre splits — constat important

L'audit met en evidence un probleme reel et significatif :

| Indicateur                                                                             | Valeur                         |
| -------------------------------------------------------------------------------------- | ------------------------------ |
| Groupes d'images issues d'une**meme photo source** repartis sur plusieurs splits | **390** (1 252 fichiers) |
| Clusters d'images visuellement quasi identiques                                        | 296 (2 654 images)             |
| dont clusters s'etendant sur plusieurs splits                                          | 173 (2 368 images)             |
| Plus grand cluster                                                                     | 1 032 images                   |
| Sequences numerotees (`frame_000324`, …) reparties sur plusieurs splits             | 58 prefixes (3 241 images)     |

Deux mecanismes distincts sont a l'oeuvre :

1. **Variantes augmentees d'une meme photo.** Roboflow nomme les fichiers
   `photo_jpg.rf.<hash>.jpg` ; le prefixe avant `.rf.` identifie la photo
   source. 390 photos sources apparaissent dans plusieurs splits. Une
   comparaison pixel a pixel confirme qu'il s'agit bien de transformations
   geometriques de la meme image (rotation de 180° verifiee sur
   `101307074_544e234e97`), **alors que le README Roboflow affirme
   « No pre-processing or augmentation was applied »**.
2. **Frames video consecutives.** 1 785 images se nomment `frame_NNNNNN` et
   proviennent de sequences video. Deux frames voisines sont quasi identiques ;
   reparties aleatoirement entre train et test, elles rendent l'evaluation
   optimiste.

**Consequence : les metriques mesurees sur ce decoupage surestiment la
performance reelle sur un chantier jamais vu.** Voir la
[section 7](#option-anti-fuite) pour l'attenuation disponible.

---

## 7. Conversion des annotations

Le dataset original n'est **jamais** modifie. Une copie normalisee est
construite, ne contenant que des lignes YOLO detection a 5 champs.

```powershell
python -m ppe_detection.dataset_cleaner --source data.yaml --output artifacts/dataset_detection --mode copy
```

Ou, en une seule commande avec audit avant/apres :

```powershell
.\scripts\audit_dataset.ps1
```

### Regroupement anti-fuite (actif par defaut)

L'export Roboflow place des **variantes augmentees d'une meme photo** dans des
splits differents : 390 photos sources se retrouvaient reparties entre train,
valid et test. Un modele evalue dans ces conditions est note sur des images
qu'il a deja apprises.

Le nettoyeur regroupe donc, **par defaut**, toutes les variantes d'une meme
photo source dans un seul split. Le split retenu est celui ou reside deja la
majorite des fichiers ; les egalites sont tranchees par un hachage stable, donc
reproductible.

Effet mesure sur ce projet :

| | Decoupage Roboflow | Regroupe (defaut) |
|---|---|---|
| Repartition train/valid/test | 4903 / 1399 / 698 | 5145 / 1277 / 578 |
| Photos sources a cheval sur plusieurs splits | 390 | **0** |
| mAP@0.50 rapportee sur le test | 0.8319 | **0.7992** |
| mAP@0.50:0.95 rapportee | 0.4696 | **0.4326** |

Les chiffres de droite sont les vrais. L'ecart de +0.033 mesurait une fuite, pas
une performance : 139 des 698 images de test (19,9 %) figuraient dans le train.

Pour reproduire le decoupage d'origine — par exemple afin de comparer a des
resultats publies sur le dataset Roboflow — utilisez `--allow-source-leak`, en
sachant que les metriques obtenues seront optimistes.

**Fuite residuelle assumee** : 78 clusters de quasi-doublons traversent encore
les splits. Ce sont des frames video consecutives (`frame_000324`,
`frame_000325`...), formellement des images sources distinctes que le
regroupement par nom ne peut pas rapprocher. Les eliminer demanderait de
re-stratifier par sequence video, ce qui releve d'une decision de protocole.

### Regle de conversion

Pour une ligne polygonale `class_id x1 y1 x2 y2 …` :

```
xmin = min(x)      center_x = (xmin + xmax) / 2
ymin = min(y)      center_y = (ymin + ymax) / 2
xmax = max(x)      width    = xmax - xmin
ymax = max(y)      height   = ymax - ymin
```

### Politique de correction

| Situation                                                                      | Traitement                                                                     |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Ligne a 5 champs valide                                                        | conservee telle quelle                                                         |
| Polygone (≥ 3 points, coordonnees paires)                                     | converti en boite englobante, journalise                                       |
| Ecart hors`[0, 1]` ≤ 1e-3                                                   | ramene aux bornes (derive numerique de l'exporteur)                            |
| Boite depassant le cadre, centre valide                                        | rognee sur l'image, journalisee                                                |
| Centre hors image, taille nulle/negative, classe inconnue, champ non numerique | **ligne exclue** et journalisee — aucune boite plausible n'est inventee |

### Options

```
--source          data.yaml du dataset original
--output          repertoire du dataset derive
--mode            copy | symlink
--overwrite       remplace explicitement un dataset existant
--dry-run         analyse sans rien ecrire
--strict          echoue si au moins une ligne doit etre exclue
--regroup-by-source  regroupe les variantes d'une meme photo dans un seul split
--no-audit        n'execute pas l'audit de verification
```

Toujours faire un essai a blanc avant :

```powershell
python -m ppe_detection.dataset_cleaner --source data.yaml --output artifacts/dataset_detection --dry-run
```

### Resultat obtenu

```
25 542 lignes lues → 25 542 conservées (0 perdue)
   349 converties depuis un polygone
   457 derives numeriques corrigees
     0 exclues
```

L'audit de verification relance automatiquement confirme : **0 ligne
polygonale, 0 ligne malformee, 100 % de lignes a 5 champs**.

### Option anti-fuite

Pour supprimer la fuite decrite en section 6 :

```powershell
python -m ppe_detection.dataset_cleaner --source data.yaml --output artifacts/dataset_detection_nofleak --mode copy --regroup-by-source
```

Toutes les variantes d'une meme photo source sont regroupees dans un seul
split (celui ou reside deja la majorite des fichiers ; les egalites sont
tranchees par un hachage stable, donc de facon reproductible).

Effet mesure :

| Indicateur                                          | Splits d'origine     | Apres regroupement  |
| --------------------------------------------------- | -------------------- | ------------------- |
| Groupes source repartis sur plusieurs splits        | 390 (1 252 fichiers) | **0**         |
| Sequences numerotees reparties sur plusieurs splits | 58 prefixes          | 19 prefixes         |
| Repartition train / valid / test                    | 4 903 / 1 399 / 698  | 5 145 / 1 277 / 578 |

488 fichiers sont deplaces. **Cette option n'est pas activee par defaut** :
elle modifie le protocole d'evaluation et rend les chiffres non comparables a
ceux publies sur le decoupage Roboflow d'origine. Le choix est laisse explicite.

> Limite : le regroupement par photo source ne resout **pas** la fuite due aux
> sequences video, car deux frames voisines sont des images sources
> differentes. Apres regroupement, 78 clusters de quasi-doublons s'etendent
> encore sur plusieurs splits.

---

## 8. Smoke test

**A executer systematiquement avant tout entrainement long.**

```powershell
.\scripts\smoke_train.ps1
```

ou :

```powershell
python -m ppe_detection.train --config configs/train.yaml --smoke
```

Le smoke test lance 2 epoques sur 4 % des donnees et verifie que la chaine
complete produit bien poids et metriques.

**Resultat obtenu sur la machine de reference : reussi en 70 secondes**, poids
ecrits dans `artifacts/models/smoke_best.pt`. Les metriques associees
(mAP@0.50 = 0,095) n'ont aucune valeur predictive — c'est attendu apres
2 epoques sur 4 % des donnees.

---

## 9. Entrainement complet

```powershell
.\scripts\train.ps1
```

ou, en commande directe :

```powershell
python -m ppe_detection.train --config configs/train.yaml
```

Surcharges frequentes :

```powershell
python -m ppe_detection.train --config configs/train.yaml --epochs 150 --batch 24
python -m ppe_detection.train --config configs/train.yaml --model yolo26m.pt --imgsz 768
python -m ppe_detection.train --config configs/train.yaml --device cpu
```

### Configuration retenue (`configs/train.yaml`)

| Parametre                              | Valeur         | Justification                                                                  |
| -------------------------------------- | -------------- | ------------------------------------------------------------------------------ |
| `model`                              | `yolo26s.pt` | Compromis vitesse/precision comme baseline                                     |
| `imgsz`                              | 640            | Standard ; ~35 % des objets sont petits, une taille inferieure les degraderait |
| `epochs`                             | 100            | Avec early stopping (`patience: 25`)                                         |
| `batch`                              | `-1` (auto)  | Ultralytics calibre a ~60 % de la VRAM                                         |
| `workers`                            | 8              | Sous Windows chaque worker est un processus complet                            |
| `amp`                                | `true`       | Indispensable sur Blackwell                                                    |
| `seed` / `deterministic`           | 42 /`true`   | Reproductibilite                                                               |
| `degrees`, `flipud`                | 0.0            | Les EPI ont une orientation stable (casque en haut)                            |
| `mixup`, `erasing`, `copy_paste` | 0.0            | Risquent de faire disparaitre les petits EPI                                   |
| `close_mosaic`                       | 10             | Desactive la mosaique en fin d'entrainement                                    |

### Reprise apres interruption

`Ctrl+C` interrompt proprement ; `last.pt` reste exploitable.

```powershell
python -m ppe_detection.train --resume artifacts/runs/ppe_yolo26s/weights/last.pt
```

### Elements archives a chaque run

`best.pt`, `last.pt`, `resolved_train_config.yaml`, `run_metadata.json`
(environnement, `pip freeze`, seed, device), `results.csv`, courbes,
matrice de confusion, exemples de predictions, `training_summary.json`.

---

## 10. Evaluation

```powershell
python -m ppe_detection.evaluate --weights artifacts/models/best.pt --data artifacts/dataset_detection/data.yaml --split test
```

ou, validation puis test :

```powershell
.\scripts\evaluate.ps1
```

> **Protocole** : les seuils et hyperparametres se choisissent sur le split de
> **validation**. Le split de test ne sert qu'une fois les choix arretes.

Le rapport contient : precision, rappel, mAP@0.50, mAP@0.50:0.95, mAP@0.75,
metriques par classe, matrice de confusion, temps de pretraitement /
inference / post-traitement, debit en images/s, taille des poids, nombre de
parametres, analyse d'erreurs (VP / FP / FN par classe), confusions entre
classes, meilleurs et pires exemples, et limites connues.

### Resultats de reference (`yolo26s`, 640 px, dataset sans fuite)

| Metrique | Validation | Test |
|----------|-----------|------|
| mAP@0.50 | 0.7789 | 0.7992 |
| mAP@0.50:0.95 | 0.4294 | 0.4326 |
| Precision | 0.7855 | 0.8086 |
| Rappel | 0.7333 | 0.7539 |

Par classe, sur le test :

| Classe | Instances | Taille mediane @640 | mAP@0.50 | mAP@0.50:0.95 |
|--------|-----------|--------------------|----------|---------------|
| Face Mask | 788 | 46 px | **0.922** | 0.560 |
| Person | 7 649 | 200 px | 0.893 | 0.531 |
| Safety Vest | 2 434 | 136 px | 0.878 | 0.535 |
| Safety Helmet | 5 449 | 50 px | 0.796 | 0.352 |
| Safety Harness | 1 175 | 155 px | 0.782 | 0.400 |
| Safety Shoes | 5 875 | 73 px | 0.775 | 0.430 |
| Safety Gloves | 2 172 | 50 px | **0.548** | 0.221 |

**La performance suit la taille des objets, pas leur frequence.** `Face Mask` est
la classe la plus rare (3,1 % des annotations) et la mieux detectee ; `Safety
Helmet` est la deuxieme plus frequente (21,3 %) et plafonne, car 25 % des casques
font moins de 32 px a 640. Rééquilibrer les classes serait donc inutile ici — le
levier est la resolution.

### Le reentrainement a 960 px : hypothese testee, resultat negatif

L'hypothese etait qu'entrainer a 960 px ferait progresser les classes a petits
objets. Elle a ete testee jusqu'au bout — un entrainement complet de 3 h 36
(97 epoques, early stopping, meilleure epoque 72) — et **elle n'est pas
confirmee au niveau global**.

Comparaison sur le meme split de test, chaque modele evalue a sa resolution
d'entrainement :

| | 640 px | 960 px | Ecart |
|---|--------|--------|-------|
| mAP@0.50 | **0.7992** | 0.7980 | −0.0012 |
| mAP@0.50:0.95 | **0.4326** | 0.4276 | −0.0050 |
| Precision | **0.8086** | 0.8059 | −0.0027 |
| Rappel | 0.7539 | **0.7572** | +0.0033 |
| Inference | **2.76 ms** | 5.83 ms | ×2.1 |
| Debit | **271 img/s** | 130 img/s | ÷2.1 |

Par classe, la prediction se verifie **partiellement** — les deux classes que
l'analyse designait progressent bien :

| Classe | % objets < 32 px | mAP@0.50 640 | mAP@0.50 960 | Ecart |
|--------|------------------|--------------|--------------|-------|
| Safety Gloves | 13 % | 0.5483 | **0.5757** | **+0.0274** |
| Person | 0 % | 0.8927 | **0.9152** | +0.0225 |
| Safety Helmet | 25 % | 0.7961 | **0.8096** | +0.0135 |
| Safety Vest | 2 % | 0.8777 | 0.8733 | −0.0044 |
| Safety Shoes | 9 % | 0.7751 | 0.7673 | −0.0078 |
| Safety Harness | 2 % | 0.7821 | 0.7553 | −0.0268 |
| Face Mask | 18 % | 0.9224 | 0.8892 | −0.0332 |

`Safety Gloves`, la classe la plus faible, gagne 5 % en relatif. Mais le signal
reste faible et bruite : les classes a petits objets gagnent +0.0026 en moyenne,
les autres perdent −0.0041. Surtout, **`Face Mask` regresse le plus fortement
alors que 18 % de ses objets sont minuscules**, ce qui contredit une explication
purement fondee sur la taille.

**Decision : `best.pt` reste le modele 640 px.** Il est meilleur ou equivalent
sur toutes les metriques globales et deux fois plus rapide. Le modele 960 est
conserve sous `artifacts/models/best_960.pt` : il peut se justifier si la
detection des gants devient prioritaire, au prix du debit.

Ce que cela apprend : **la resolution seule ne compense pas un manque de
diversite dans les donnees.** Le levier restant est la collecte de donnees de
terrain — voir [`docs/plan_ecart_terrain.md`](docs/plan_ecart_terrain.md).

À noter egalement : evaluer les poids 640 px a 960 px sans reentrainer degrade
le resultat (0.780 vs 0.799). **Augmenter la resolution a l'inference seule ne
fonctionne pas** — le modele attend l'echelle sur laquelle il a ete entraine.

---

### Calibration des seuils par classe

Un seuil unique pour toutes les classes est un compromis mediocre : chaque
classe a sa propre distribution de scores. La commande suivante balaie les
seuils et retient, pour chaque classe, celui qui maximise le F1 :

```powershell
python -m ppe_detection.calibrate --weights artifacts/models/best.pt `
  --data artifacts/dataset_detection/data.yaml --split valid
```

Ajoutez `--apply` pour ecrire directement les seuils dans
`configs/inference.yaml` (attention : la reecriture YAML supprime les
commentaires du fichier ; le rapport fournit toujours l'extrait a recopier).

**Protocole** : la calibration s'effectue sur la **validation** uniquement.
Choisir des seuils sur le test reviendrait a ajuster le modele sur les donnees
censees l'evaluer — la commande refuse d'ailleurs `--split test` sauf
`--allow-test-split` assume explicitement.

Resultats obtenus sur ce projet (seuils deja reportes dans `inference.yaml`) :

| Classe | Seuil retenu | Gain de F1 sur le **test** |
|--------|--------------|---------------------------|
| Face Mask | 0.25 | +0.0000 |
| Person | 0.35 | +0.0076 |
| Safety Gloves | 0.30 | +0.0025 |
| Safety Harness | 0.40 | **+0.0235** |
| Safety Helmet | 0.30 | +0.0116 |
| Safety Shoes | 0.30 | −0.0001 |
| Safety Vest | 0.45 | +0.0127 |
| **F1 macro** | | **+0.0083** |

Concretement sur le split de test : **120 faux positifs en moins** pour 52 vrais
positifs perdus. Les seuils choisis sur la validation generalisent donc bien
(+0.0067 attendu, +0.0083 constate).

Une contrainte metier de rappel minimal est disponible via `--min-recall` :
utile lorsque manquer un EPI coute plus cher qu'une fausse alerte.

## 11. Inference

Interface unique pour toutes les sources.

```powershell
# Image
python -m ppe_detection.predict --weights artifacts/models/best.pt --source chemin\image.jpg --save --save-json

# Dossier
python -m ppe_detection.predict --weights artifacts/models/best.pt --source chemin\dossier --save --save-txt --save-json --save-csv

# Video
python -m ppe_detection.predict --weights artifacts/models/best.pt --source chemin\video.mp4 --save --save-json

# Webcam (fenetre temps reel, 'q' ou Echap pour quitter)
python -m ppe_detection.predict --weights artifacts/models/best.pt --source 0 --show

# Flux RTSP
python -m ppe_detection.predict --weights artifacts/models/best.pt --source "rtsp://user:motdepasse@192.168.1.10:554/stream" --show
```

Options principales :

```
--conf / --iou       seuils de confiance et de NMS
--device             auto | cpu | cuda | 0
--imgsz              taille d'inference
--max-det            detections maximales par image
--half               FP16 (GPU uniquement)
--save               images/video annotees
--save-txt           labels YOLO (class cx cy w h conf)
--save-json          rapport JSON structure
--save-csv           rapport CSV a plat
--recursive          parcourt les sous-dossiers
--hide-labels        masque les noms de classes
--hide-conf          masque les scores
--compliance         active la conformite EPI
--track              suivi d'objets + lissage temporel des verdicts
--tracker            bytetrack.yaml (rapide) | botsort.yaml (occlusions)
--show               fenetre temps reel (video/webcam)
--frame-skip N       n'infere qu'une frame sur N+1
--max-frames N       limite le nombre de frames
```

Les seuils **par classe** se definissent dans `configs/inference.yaml` :

```yaml
inference:
  conf: 0.20
  class_conf:
    Face Mask: 0.35
    Safety Gloves: 0.40
```

> Un seuil par classe ne peut que **durcir** le seuil global : Ultralytics
> filtre deja a `conf` avant que ces seuils s'appliquent. Pour reellement
> abaisser un seuil, baissez `conf` puis remontez les autres classes.

Comportement video verifie : le modele n'est charge qu'une fois, l'ordre des
frames est preserve, les FPS sont affiches en incrustation, la camera et le
`VideoWriter` sont liberes dans un bloc `finally`, et les proprietes de la
video de sortie (resolution, FPS, nombre de frames) sont conservees.

---

## 12. Conformite EPI

> **Avertissement.** Le modele detecte des objets **independamment**. Rien dans
> ses sorties ne relie formellement un casque a une personne. La couche de
> conformite applique une **heuristique geometrique** : un EPI est attribue a
> la personne dont la region attendue contient la plus grande fraction de la
> boite de l'EPI. Un statut « non conforme » est une **alerte a verifier**,
> jamais un constat automatique.

```powershell
python -m ppe_detection.predict --weights artifacts/models/best.pt --source chemin\image.jpg --compliance --save --save-json
python -m ppe_detection.predict --weights artifacts/models/best.pt --source 0 --show --compliance --required-ppe "Safety Helmet" "Safety Vest"
```

Configuration (`configs/inference.yaml`) :

```yaml
compliance:
  enabled: false
  person_class: Person
  required_ppe:
    - Safety Helmet
    - Safety Vest
  association:
    containment_threshold: 0.50   # fraction de la boite EPI dans la zone attendue
    helmet_region: 0.35           # 35 % superieurs de la personne
    shoes_region: 0.30            # 30 % inferieurs
    torso_region: [0.20, 0.80]
  region_by_class:
    Safety Helmet: head
    Face Mask: head
    Safety Vest: torso
    Safety Harness: torso
    Safety Gloves: any
    Safety Shoes: feet

  # Observabilite : quand peut-on AFFIRMER qu'un EPI manque ?
  min_region_height_px: 24   # sous ce seuil, l'objet n'est pas resoluble
  edge_margin_px: 2          # zone touchant le bord = tronquee

  # Lissage temporel (option --track)
  temporal_window: 15
  temporal_min_ratio: 0.70
  temporal_min_observations: 5
```

### Association par points cles du corps (recommande)

Le decoupage par fractions suppose une personne **debout et vue de face**. Cette
hypothese tombe des que la personne est accroupie, penchee, assise, ou filmee en
plongee — le cas courant en videosurveillance.

L'option `--pose` remplace ce decoupage par la position **reelle** des parties du
corps, obtenue via un modele d'estimation de pose (17 points cles COCO,
`yolo26n-pose.pt`, telecharge automatiquement) :

```powershell
python -m ppe_detection.predict --weights artifacts/models/best.pt `
  --source chemin\image.jpg --compliance --pose --save --save-json
```

| Zone | Points cles utilises | Remplace |
|------|---------------------|----------|
| `head` | nez, yeux, oreilles + echelle du buste | 35 % superieurs de la boite |
| `torso` | epaules et hanches | tranche 20–80 % |
| `feet` | chevilles | 30 % inferieurs |
| `hands` | poignets | boite entiere |

Un casque masque le crane : la zone « tete » est donc extrapolee **au-dessus**
des points du visage, a partir de la longueur du buste.

Mesure sur une ouvriere accroupie (frame reelle) — la zone du torse passe de
`x[387-1133] y[144-576]` (fractions) a `x[719-999] y[262-679]` (pose), soit un
recentrage conforme a sa posture.

**Repli automatique** : si les points cles necessaires manquent (personne de dos,
trop petite, occultee), le systeme revient au decoupage par fractions pour cette
zone. Le champ `association_method` de chaque verdict indique la methode
reellement employee, `pose` ou `bbox_fractions`.

Cout : un second modele en memoire et une inference supplementaire par image.

### Trois etats, pas deux

Un detecteur qui ne voit pas un gilet ne prouve pas son absence. Declarer
« non conforme » une personne dont la zone concernee n'est pas observable
produit des fausses alertes en masse. Le systeme distingue donc :

| Statut            | Signification                                                     | Couleur |
| ----------------- | ----------------------------------------------------------------- | ------- |
| `compliant`     | Tous les EPI requis sont detectes et attribues                    | vert    |
| `non_compliant` | Un EPI requis manque**dans une zone reellement observable** | rouge   |
| `indeterminate` | La zone n'est pas observable — aucune conclusion                 | ambre   |

Une zone est jugee non observable dans deux cas : elle **touche un bord du
cadre** (personne tronquee, typiquement la tete qui depasse par le haut), ou
elle est **trop petite en pixels** pour que le detecteur y resolve un objet.

Le champ `reasons` de chaque personne indique precisement pourquoi un EPI est
indetermine, par exemple `Safety Helmet : zone 'head' tronquee par le bord haut du cadre`.

Le taux de conformite est calcule sur les seules personnes **jugeables** :
inclure les indetermines au denominateur ferait baisser artificiellement le taux
a cause de gens qu'on n'a simplement pas pu observer.

### Suivi et lissage temporel (video)

```powershell
python -m ppe_detection.predict --weights artifacts/models/best.pt --source chemin\video.mp4 --compliance --track --save --save-json
```

Avec `--track`, chaque personne recoit un identifiant persistant (ByteTrack par
defaut, `botsort.yaml` disponible pour plus de robustesse aux occlusions), et le
verdict est lisse sur une fenetre glissante : il ne bascule qu'apres une
majorite nette d'observations concordantes. Une alerte est levee **une fois par
personne**, pas a chaque frame.

Le rapport distingue alors deux vues :

- `tracked_compliance` — le bilan **par personne**, seule vue ayant un sens
  operationnel ;
- `per_detection_compliance` — l'ancien decompte par detection, conserve pour
  comparaison.

Effet mesure sur une video de chantier de 122 frames :

| Vue                                    | Sans suivi                 | Avec`--track`               |
| -------------------------------------- | -------------------------- | ----------------------------- |
| Unites comptees                        | 194 detections de personne | **6 personnes suivies** |
| Non conformes                          | 147                        | 4                             |
| Indetermines (retenus par le niveau 1) | 35                         | 2                             |
| Alertes emises                         | 147                        | **5**                   |

### Distinguer les vrais EPI de leurs sosies

Le modele actuel etiquette un **casque de velo** comme `Safety Helmet` avec
**0.84 de confiance** (verifie sur images de test). Il n'a pas appris « casque
de chantier » mais « coque rigide bombee sur une tete ».

Ce n'est pas un manque de donnees : le schema ne comporte que des classes
**positives**, donc aucune sortie ne permet d'exprimer « ressemble a un casque
mais n'en est pas un ». Ajouter des casques de chantier n'y changera rien.

Le projet fournit l'outillage pour y remedier :

- [`taxonomy.py`](src/ppe_detection/taxonomy.py) definit un **schema etendu a
  10 classes** (`Non-Safety Headwear`, `Non-Safety Vest`, `Non-Safety Footwear`).
  Les sept classes d'origine gardent leurs identifiants : un dataset etendu
  reste retro-compatible.
- [`dataset_merge.py`](src/ppe_detection/dataset_merge.py) assemble des datasets
  publics en remappant leurs classes, ce qui evite d'annoter de zero.
- Le mecanisme de **contre-preuve** distingue deux niveaux de preuve dans le
  verdict : `evidence: absence` (rien detecte, peut etre un faux negatif) et
  `evidence: observed` (couvre-chef non conforme vu — violation constatee).
  Une contre-preuve prime sur le test d'observabilite : apercevoir l'objet
  prouve que la zone est visible.

Volumes a annoter, sources gratuites et regles d'annotation :
[`docs/plan_donnees_epi_sosies.md`](docs/plan_donnees_epi_sosies.md).

#### Resultats du modele a 8 classes

Dataset etendu : 8 204 images, 29 079 annotations, dont **3 537 instances de
`Non-Safety Headwear`** provenant d'Open Images V7 — sans annotation manuelle.
Entrainement de 2 h 09 (91 epoques, early stopping, meilleure epoque 66).

**Test 1 — les sosies.** C'est l'objectif poursuivi, et il est atteint :

| Image | 7 classes | 8 classes |
|-------|-----------|-----------|
| Casque VTT | `Safety Helmet 0.32` | **`Non-Safety Headwear 0.94`** |
| Casque velo route | `Safety Helmet 0.84` | **`Non-Safety Headwear 0.50`** |
| Casquette baseball | *rien* | **`Non-Safety Headwear 0.41`** |
| Casquette sport | *rien* | **`Non-Safety Headwear 0.95`** |

Plus aucun faux `Safety Helmet`. Les casquettes, auparavant simplement ignorees,
sont desormais **detectees activement**, ce qui permet a la contre-preuve de
fonctionner.

**Test 2 — non-regression sur les 578 memes images.** Comparer les mAP globales
de deux modeles a nombre de classes different n'aurait aucun sens : la moyenne
ne porte pas sur les memes classes. La comparaison se fait donc classe par
classe, sur des images identiques.

| Classe | mAP@0.50 7 cls | mAP@0.50 8 cls | Ecart |
|--------|----------------|----------------|-------|
| Safety Harness | 0.7821 | **0.8091** | +0.0270 |
| Safety Gloves | 0.5483 | **0.5607** | +0.0124 |
| Safety Helmet | 0.7961 | **0.8067** | +0.0106 |
| Person | 0.8927 | **0.8960** | +0.0033 |
| Safety Vest | 0.8777 | 0.8741 | −0.0036 |
| Safety Shoes | 0.7751 | 0.7523 | −0.0228 |
| Face Mask | 0.9224 | 0.8931 | −0.0293 |
| **Global (7 classes)** | **0.7992** | **0.7988** | **−0.0004** |

L'ecart global est dans le bruit. `Safety Helmet` **progresse** de +0.0106,
contrairement a la degradation qu'on pouvait redouter : discriminer n'a pas
coute en detection. Deux classes reculent au-dela de 0.02, `Face Mask` et
`Safety Shoes`, sans lien evident avec le couvre-chef.

**Test 3 — qualite de la nouvelle classe** : `Non-Safety Headwear` atteint
**0.7373 de mAP@0.50** avec 0.829 de precision, soit la 5e classe sur 8 —
devant `Safety Shoes` et loin devant `Safety Gloves`. Assez fiable pour fonder
une alerte, d'ou l'activation de `counter_evidence` dans
[`configs/inference.yaml`](configs/inference.yaml).

#### Limite connue : personnes non detectees en contexte sportif

Sur les photos issues d'Open Images — portraits de cyclistes, scenes de sport —
le modele a 8 classes **ne detecte plus les personnes**. Sur l'image du
cycliste, le modele a 7 classes voyait `Person 0.886` ; le nouveau ne voit que
le couvre-chef.

Cause : les images d'Open Images ont ete telechargees avec `only_matching=True`,
qui ne conserve que les etiquettes de couvre-chef. Les personnes y figurent donc
**sans annotation**, et le modele apprend a ne pas les detecter dans ce contexte
visuel.

Portee reelle : **nulle sur le domaine cible**. Sur 60 frames de chantier
identiques, le modele a 8 classes detecte meme *plus* de personnes que le
precedent (72 contre 65). La regression se limite a l'imagerie sportive.

Correction pour une prochaine iteration : retelecharger avec
`only_matching=False` et mapper aussi `Person=Person`, afin que les personnes
des images Open Images soient annotees.

### Limites persistantes

Les garde-fous ci-dessus suppriment une large part des fausses alertes, mais
l'heuristique reste faillible :

- **Personnes proches ou qui se chevauchent** : le casque de l'une peut etre
  attribue a l'autre.
- **Plongee ou contre-plongee forte** : l'hypothese « la tete est en haut de la
  boite » ne tient plus. Seule une approche par pose corrigerait ce cas.
- **EPI non porte** : un casque pose sur une table dans la zone tete d'une
  personne assise sera compte comme porte.
- **Faux negatif de detection dans une zone observable** : un gilet bien visible
  mais non detecte produit toujours un « non conforme » a tort. C'est
  aujourd'hui la principale source d'erreur restante, et elle releve du
  detecteur, pas de la regle metier. L'option `--pose` ne la corrige pas : en
  rendant l'evaluation de l'observabilite plus juste, elle a meme tendance a
  **exposer** ces echecs plutot qu'a les masquer derriere un « indetermine ».
- **Classes peu fiables** : `Safety Gloves` plafonne a 0.55 de mAP@0.50 et 0.57
  de rappel. Elle figure dans `unreliable_ppe` : l'inscrire dans `required_ppe`
  declenche un avertissement au chargement, car la regle produirait une majorite
  de fausses alertes.
- Le champ `verdict_confidence` porte sur la **detection**, pas sur la justesse
  de la regle metier.
- Une personne qui se met en conformite puis redevient non conforme genere
  **deux** evenements d'alerte : c'est voulu, mais cela distingue
  `n_alerts` (evenements) de `persons_currently_alerted` (etat final).

---

## 13. API REST

```powershell
.\scripts\run_api.ps1
.\scripts\run_api.ps1 -Weights artifacts/models/best.pt -Port 8080 -Compliance
```

ou :

```powershell
python -m uvicorn ppe_detection.api:app --host 127.0.0.1 --port 8000
```

Documentation interactive : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

| Methode | Route              | Description                             |
| ------- | ------------------ | --------------------------------------- |
| GET     | `/health`        | Etat du service (`ok` / `degraded`) |
| GET     | `/model-info`    | Metadonnees du modele charge            |
| POST    | `/predict/image` | Inference sur une image                 |
| POST    | `/predict/batch` | Inference sur plusieurs images          |

Exemple de reponse :

```json
{
  "request_id": "3f1c...",
  "filename": "scene.jpg",
  "image": { "width": 640, "height": 640 },
  "detections": [
    {
      "class_id": 1,
      "class_name": "Person",
      "confidence": 0.8177,
      "bbox_xyxy": [411.99, 140.43, 605.24, 581.97]
    }
  ],
  "compliance": [],
  "timing_ms": { "preprocess": 1.48, "inference": 79.1, "postprocess": 0.27, "total_request": 103.22 }
}
```

Configuration par variables d'environnement (voir `.env.example`) :
`PPE_API_WEIGHTS`, `PPE_API_DEVICE`, `PPE_API_CONF`, `PPE_API_IOU`,
`PPE_API_COMPLIANCE`, `PPE_API_MAX_FILE_MB`, `PPE_API_MAX_BATCH`.

### Choix de securite

- Le modele n'est charge **qu'une fois**, au demarrage.
- Les fichiers recus sont decodes **entierement en memoire** : aucun contenu
  fourni par un client n'atteint le disque, ce qui elimine tout risque
  d'ecriture arbitraire via un nom de fichier hostile.
- Le nom renvoye dans la reponse est neutralise (`../../etc/passwd` devient
  un nom inoffensif).
- Type MIME et taille sont valides **avant** tout decodage.
- Les journaux ne contiennent ni le contenu des images ni le nom brut soumis.
- Une exception non geree renvoie un JSON avec un identifiant d'erreur, sans
  divulguer de trace interne.
- Si les poids sont absents, le service demarre quand meme : `/health` repond
  `degraded` et les routes d'inference renvoient **503** avec un message
  actionnable, plutot que de faire echouer le demarrage.

---

## 14. Interface Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Puis [http://localhost:8501](http://localhost:8501).

L'interface comporte quatre onglets.

| Onglet                    | Fonction                                                                                                                                                                     |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Image**           | Televersement d'une photo, image annotee, tableau des detections, conformite, telechargement JSON/JPEG.                                                                      |
| **Video**           | Televersement d'un fichier, traitement,**lecture directe de la video annotee**, journal des alertes, telechargements.                                                  |
| **Webcam (direct)** | Inference**en continu** sur la camera locale, avec FPS, suivi, alertes en temps reel et enregistrement optionnel. Un mode « photo ponctuelle » est aussi disponible. |
| **Resultats**       | Relecture de toutes les videos annotees deja produites dans`artifacts/predictions/`.                                                                                       |

Reglages communs dans la barre laterale : poids, device, seuils de confiance et
d'IoU, activation de la conformite, choix des EPI obligatoires et activation du
suivi temporel.

### Webcam en direct

La camera est ouverte **cote serveur**, par le processus Streamlit. C'est adapte
a un usage local, ou le navigateur et la camera sont sur la meme machine ; un
deploiement distant necessiterait WebRTC (`streamlit-webrtc`).

Le bouton « Demarrer la camera » lance une boucle d'inference continue ;
« Arreter » l'interrompt. Un garde-fou de duree (120 s par defaut) coupe
automatiquement la boucle, et la camera est toujours liberee dans un bloc
`finally`, meme en cas d'erreur ou de rerun Streamlit.

Debit mesure sur la machine de reference : **~14 FPS en 640×480** avec suivi et
conformite actives, annotation comprise.

### Lecture des videos annotees — codec

Si vos videos annotees restaient noires dans le navigateur, c'est un probleme de
codec, desormais corrige.

OpenCV ecrivait en `mp4v`, qui produit un flux **MPEG-4 Part 2** (FOURCC
`FMP4`) qu'aucun navigateur ne decode nativement. Le projet demande maintenant
`avc1` (**H.264**), lu partout.

Deux precautions ont ete necessaires :

- OpenCV affiche parfois `Could not open codec libopenh264` sur la sortie
  d'erreur puis **bascule silencieusement sur un autre encodeur H.264**. Le
  fichier produit est valide : ce message est benin et peut etre ignore.
- OpenCV **substitue un codec sans le signaler** lorsque celui demande est
  indisponible : `isOpened()` renvoie `True` meme avec un FOURCC fantaisiste.
  Le projet relit donc le fichier apres fermeture pour connaitre le codec
  reellement ecrit (`probe_video_codec`). Le champ `output_codec` du resume
  reflete la realite, et `browser_playable` en decoule.

Les videos produites avant cette correction restent en `FMP4` : l'onglet
« Resultats » les signale explicitement et propose leur telechargement.

**L'interface n'entraine jamais de modele.** Si aucun poids n'est disponible,
elle affiche les commandes exactes a executer.

---

## 15. Export ONNX

```powershell
python -m ppe_detection.export --weights artifacts/models/best.pt --format onnx --imgsz 640 --simplify
```

Formats optionnels : `torchscript`, `openvino`, `engine` (TensorRT, necessite
une installation dediee).

### Verification reellement effectuee

Un export n'est jamais considere comme reussi au seul motif qu'un fichier
existe. La verification comprend :

1. fichier present et non vide ;
2. graphe valide (`onnx.checker`) ;
3. session ONNX Runtime chargeable ;
4. inference sur une entree factice de forme attendue ;
5. **comparaison des detections** entre PyTorch et ONNX Runtime sur une vraie
   image, avec appariement par IoU.

Resultat mesure sur les poids du smoke test :

| Controle                   | Resultat                  |
| -------------------------- | ------------------------- |
| Graphe ONNX (opset 20)     | valide                    |
| Session ONNX Runtime       | chargeable                |
| Forme de sortie            | `(1, 300, 6)`           |
| Detections PyTorch vs ONNX | **9 / 9 appariees** |
| IoU moyen                  | **0,999999**        |
| Ecart max de confiance     | 6 × 10⁻⁶               |
| Decalage max de boite      | 0,0002 px                 |
| Classes divergentes        | 0                         |

### Differences de post-traitement a connaitre

- Le modele exporte attend une image normalisee dans `[0, 1]`, en NCHW
  (`1×3×H×W`), RGB, redimensionnee par letterbox vers la taille figee.
- YOLO26 s'exporte **end-to-end** : la sortie contient deja des detections
  filtrees et triees par confiance. Comparer les tenseurs bruts terme a terme
  n'aurait pas de sens, car un ecart numerique infime reordonne les lignes.
- Les coordonnees se rapportent a l'image redimensionnee : il faut annuler le
  letterbox (echelle et decalage) pour revenir a l'image d'origine.

---

## 16. Structure des sorties

```
artifacts/
├── reports/
│   ├── dataset_audit_original.{json,md}    # Audit du dataset original
│   ├── dataset_audit_detection.{json,md}   # Audit du dataset normalise
│   ├── dataset_cleaning.{json,md}          # Journal de conversion
│   ├── evaluation_{valid,test}.{json,md}   # Rapports d'evaluation
│   ├── export.{json,md}                    # Rapport d'export
│   └── *_assets/                           # Graphiques et exemples annotes
├── dataset_detection/          # Dataset normalise (data.yaml + train/valid/test)
├── models/                     # best.pt, last.pt, smoke_best.pt
├── runs/
│   ├── <experience>/           # Poids, courbes, matrice de confusion
│   └── val/                    # Sorties de validation
├── predictions/<nom>/
│   ├── images/                 # Images annotees
│   ├── labels/                 # Labels YOLO predits
│   ├── predictions.{json,csv}
│   └── video_{summary,predictions}.json
├── exports/                    # *.onnx, *.torchscript
└── logs/                       # Journaux par commande
```

---

## 17. Qualite du code et tests

```powershell
python -m pytest tests -q          # 116 tests
python -m ruff check src tests app # linting
python -m mypy                     # verification de types
```

Etat verifie : **116 tests passent, ruff sans erreur, mypy sans erreur**.

Les tests n'exigent aucun entrainement complet : ils s'appuient sur des
fixtures synthetiques (mini dataset couvrant tous les cas limites d'annotation)
et ignorent automatiquement les tests d'inference si aucun poids n'est present.

Couverture : parsing et conversion des annotations, resolution des chemins
`data.yaml` (y compris la convention Roboflow `../train/images`), audit,
nettoyage, preservation du dataset source, geometrie de la conformite,
neutralisation des noms de fichiers, formats de sortie, et les quatre routes
de l'API via le client de test FastAPI.

---

## 18. Depannage

### CUDA indisponible alors qu'un GPU est present

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Si `torch.__version__` ne contient pas `+cuXXX`, la version CPU est installee :

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### « no kernel image is available for execution on the device »

La build PyTorch ne contient pas de noyaux pour votre GPU. Typique des
RTX 50xx (Blackwell, `sm_120`) avec une build anterieure a CUDA 12.8.

```powershell
python -c "import torch; print(torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
```

Si `sm_120` est absent de la liste, reinstallez en `cu128`.

### Memoire GPU insuffisante (CUDA out of memory)

Par ordre d'efficacite :

```powershell
python -m ppe_detection.train --config configs/train.yaml --batch 16   # puis 8
python -m ppe_detection.train --config configs/train.yaml --imgsz 512
python -m ppe_detection.train --config configs/train.yaml --model yolo26n.pt
```

Verifiez aussi `cache: false` dans `configs/train.yaml` et fermez les autres
applications utilisant le GPU (`nvidia-smi`).

### Entrainement tres lent sous Windows

Reduisez `workers` (chaque worker est un processus complet) et activez
`cache: disk` si le disque le permet.

### Webcam inaccessible

Verifiez qu'aucune autre application n'utilise la camera et que l'acces est
autorise dans **Parametres > Confidentialite > Camera**. Essayez un autre
index (`--source 1`).

### La video annotee n'est pas produite

Le codec `mp4v` peut manquer selon la build d'OpenCV. Le message est explicite
et les detections restent exportables en JSON.

### Chemins contenant des espaces

Encadrez-les de guillemets :

```powershell
python -m ppe_detection.predict --weights "artifacts/models/best.pt" --source "C:\Mes Images\chantier.jpg"
```

---

## 19. Limites connues

Cette section liste ce qui est **reellement** limitant. Rien n'est passe sous
silence.

### Donnees

1. **Fuite residuelle par sequence video.** Le regroupement par photo source est
   desormais actif par defaut et supprime les 390 groupes concernes. Il reste
   **78 clusters de quasi-doublons a cheval sur les splits** : des frames video
   consecutives, formellement distinctes, que le regroupement par nom ne peut
   pas rapprocher. Les metriques restent donc legerement optimistes.
2. **Documentation source inexacte.** Le README Roboflow annonce « No
   pre-processing or augmentation was applied », ce que l'analyse pixel
   contredit (variantes pivotees d'une meme photo).
3. **Diversite reelle inferieure au volume affiche.** Sur 7 000 images, environ
   1 785 sont des frames extraites de quelques sequences video, tres
   redondantes entre elles.
4. **Boites issues de polygones.** 349 boites proviennent d'une conversion
   min/max : la boite englobante d'un polygone est toujours au moins aussi
   grande que l'objet reel. Cela contribue au faible mAP@0.50:0.95 (0.433
   contre 0.799 a IoU 0.50).
5. **Petits objets — facteur limitant principal.** 25 % des casques et 13 % des
   gants font moins de 32 px a 640. Cela pese davantage que le desequilibre des
   classes : `Face Mask`, la classe la plus rare, est la mieux detectee (0.922),
   tandis que `Safety Helmet`, la deuxieme plus frequente, plafonne a 0.796.
6. **Ecart au terrain non mesure.** Le dataset ne couvre ni la nuit, ni la
   pluie, ni le contre-jour, ni les angles de videosurveillance. Sur une video
   de chantier reelle, `Safety Vest` (0.878 de mAP@0.50 en test) n'a ete detecte
   que 15 fois sur 122 images. Plan de correction :
   [`docs/plan_ecart_terrain.md`](docs/plan_ecart_terrain.md).

### Modele et pipeline

7. **`Safety Gloves` n'est pas exploitable en production** : 0.548 de mAP@0.50 et
   0.555 de rappel — plus d'un gant sur quatre est manque. La classe figure dans
   `unreliable_ppe` et declenche un avertissement si elle est inscrite dans
   `required_ppe`.
8. **La conformite EPI reste une heuristique**, meme avec `--pose`. Les points
   cles suppriment l'hypothese « personne debout vue de face », mais ni la pose
   ni la geometrie ne prouvent qu'un EPI est effectivement **porte** : un casque
   pose sur une table dans la zone tete sera compte comme porte. Limites
   detaillees en [section 12](#12-conformite-epi).
9. **La verification ONNX porte sur une seule image.** Elle prouve la fidelite
   de la conversion, pas l'equivalence sur toute distribution d'entrees.
10. **Les exports TensorRT et OpenVINO ne sont pas verifies automatiquement**
    (seul ONNX l'est) et n'ont pas ete testes ici.
11. **Le mode `symlink` retombe silencieusement sur la copie** sous Windows si
    les droits ne permettent pas la creation de liens (mode developpeur requis).
12. **Couverture de tests a 51 %.** Les chemins lourds (entrainement, export,
    audit complet) sont peu couverts : les exercer demanderait un GPU et le
    dataset complet.
13. **Incoherence de l'option `--output`.** Elle designe un **repertoire** dans
    `evaluate`, `export`, `predict` et `dataset_cleaner`, mais un **fichier**
    dans `dataset_audit`. Piege a eviter tant que ce n'est pas uniformise.

---

## 20. Pistes d'amelioration

### Deja fait

- Regroupement anti-fuite par photo source, **actif par defaut** (section 7).
- Etat `indeterminate` : ne pas accuser une personne qu'on ne peut pas observer.
- Suivi multi-objets et lissage temporel des verdicts (`--track`).
- Association par points cles du corps (`--pose`).
- Calibration des seuils par classe sur la validation (`calibrate`).

### Priorites restantes

**1. Mesurer l'ecart au terrain.** Constituer un jeu de test de 150 a 200
images issues des conditions reelles de deploiement, jamais melangees a
l'entrainement. C'est le seul juge honnete de la performance en production, et
le prealable a toute autre optimisation. Voir
[`docs/plan_ecart_terrain.md`](docs/plan_ecart_terrain.md).

**2. Resolution : piste testee, a ne pas relancer telle quelle.** Un
entrainement complet a 960 px n'a apporte aucun gain global (voir section 10).
Inutile d'y revenir sans changer autre chose. Restent a essayer :
l'entrainement multi-echelle (`multi_scale: true`), qui expose le modele aux
deux regimes plutot que d'en privilegier un, et un modele plus capacitaire
(`yolo26m`) a 640 px, moins couteux a l'inference qu'un `yolo26s` a 960 px.

**3. `Safety Gloves`.** Campagne d'annotation dediee, ou retrait assume des
regles de conformite. En l'etat, la classe ne supporte aucune decision.

**4. Stratification par sequence video.** Eliminer les 78 clusters de
quasi-doublons restants exige de repartir les splits par sequence source, et non
par nom de fichier.

**5. Conformite avancee.** Verifier qu'un EPI est *porte* et non simplement
present dans la zone (coherence temporelle du port, orientation du casque).

**6. Industrialisation.** Quantification INT8 pour l'embarque ; export TensorRT
verifie ; conteneurisation de l'API ; supervision de la derive du modele.

**7. Dette technique.** Uniformiser la semantique de `--output` ; monter la
couverture de tests sur `train.py` et `export.py`.

---

## Licence et attribution

Code sous licence MIT. Le dataset provient de Roboflow Universe sous licence
**CC BY 4.0** et doit etre attribue a son auteur :
[https://universe.roboflow.com/ousmane-savadogo/ppe-detection-project-jeezl-p9ncg](https://universe.roboflow.com/ousmane-savadogo/ppe-detection-project-jeezl-p9ncg)

Ce systeme est une aide a la detection. **Il ne remplace pas l'inspection
humaine de securite** et ne doit pas etre utilise comme unique mecanisme de
controle de conformite sur un site reel.
