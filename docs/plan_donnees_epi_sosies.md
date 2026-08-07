# Plan d'acquisition — distinguer les vrais EPI de leurs sosies

## Le probleme, mesure

Le modele actuel etiquette un **casque de velo** comme `Safety Helmet` avec
**0.84 de confiance**. Verifie le 7 août 2026 sur quatre images de test :

| Image | Ce qui est porte | Verdict du modele |
|-------|------------------|-------------------|
| `casque_velo2.jpg` | Casque de velo aerodynamique | **Safety Helmet 0.84** |
| `casque_velo.jpg` | Casque de VTT | **Safety Helmet 0.32** |
| `casquette_baseball.jpg` | Casquette de baseball | rien (correct) |
| `casquette_sport.jpg` | Casquette de sport | rien (correct) |

Les casquettes sont correctement ignorees, pas les casques de velo. Le modele
n'a pas appris « casque de chantier » mais **« coque rigide bombee sur une
tete »** — description que le casque de velo satisfait parfaitement.

Avec la couche de conformite active, ce cycliste serait declare **conforme**.

## Pourquoi plus de donnees positives ne suffit pas

Le schema comporte sept classes, **toutes positives**. Aucune sortie ne permet
d'exprimer « objet ressemblant a un casque mais non conforme ». Face a un casque
de velo, le modele doit choisir une classe : il prend la plus proche.

Ajouter des milliers de casques de chantier ameliorera la detection des vrais
casques, sans jamais deplacer cette frontiere — **parce qu'il n'y a pas de
frontiere, il n'y a qu'une classe**. Il faut passer d'une detection a une
discrimination, ce qui suppose des classes negatives explicites.

---

## Combien d'images annoter

Les volumes ci-dessous valent **par classe negative**, en nombre d'instances
annotees (pas d'images : une image peut en contenir plusieurs).

| Palier | Instances | Ce qu'on obtient |
|--------|-----------|------------------|
| Minimum viable | **300 – 500** | Le modele commence a separer les deux classes. Suffisant pour valider que l'approche fonctionne avant d'investir davantage. |
| Exploitable | **800 – 1 200** | Performance utilisable en production sur les cas courants. C'est la cible recommandee. |
| Robuste | **2 000+** | Tient les cas difficiles : eclairage faible, objet partiel, angles inhabituels. |

Pour situer, `Safety Harness` compte 1 175 instances dans votre dataset actuel
et atteint 0.78 de mAP@0.50 — l'ordre de grandeur « exploitable » est donc
coherent avec ce que vous observez deja.

### Ce qui compte davantage que le volume

**Les exemples proches de la frontiere.** Un casque de velo blanc a cote d'un
casque de chantier blanc apprend infiniment plus que cent casques de velo sur
fond de route. Une collecte aleatoire n'en contient presque pas : il faut les
chercher deliberement.

Repartition conseillee pour 1 000 instances de `Non-Safety Headwear` :

| Type | Instances | Pourquoi |
|------|-----------|----------|
| Casques de velo / VTT | 350 | Le sosie le plus dangereux — c'est lui qui trompe le modele aujourd'hui |
| Casquettes, bonnets | 300 | Frequents sur chantier, deja bien geres, a consolider |
| Casques de moto | 200 | Coque integrale, forme distincte |
| Casques sport (ski, escalade, equitation) | 150 | Cas limites qui affinent la frontiere |

**Le contexte doit varier.** Si tous vos casques de velo sont photographies sur
des cyclistes en Lycra et tous vos casques de chantier sur des ouvriers en
gilet, le modele apprendra a reconnaitre le **vetement**, pas le casque. Il
echouera des qu'un ouvrier portera un casque de velo. Il faut donc des casques
de velo en contexte chantier — quitte a les mettre en scene.

---

## Ou trouver ces donnees

### 1. Open Images V7 — la source principale, gratuite

**Verifie le 7 août 2026** : le fichier officiel des classes boxables contient
exactement les classes utiles.

| Identifiant | Classe | Usage |
|-------------|--------|-------|
| `/m/03p3bw` | **Bicycle helmet** | Coeur de `Non-Safety Headwear` |
| `/m/07qxg_` | **Football helmet** | Sosie supplementaire |
| `/m/02dl1y` | **Hat** | Casquettes, bonnets, chapeaux |
| `/m/04tn4x` | **Swim cap** | Cas limite |
| `/m/0zvk5` | Helmet | Generique — a trier manuellement |
| `/m/025rp__` | Cowboy hat | Complement |

- Site : <https://storage.googleapis.com/openimages/web/index.html>
- Liste des classes : <https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv>
- Licence : images sous **CC BY 2.0**, annotations sous **CC BY 4.0**. Usage
  commercial autorise avec attribution.

Telechargement d'un sous-ensemble par classe, via FiftyOne :

```powershell
pip install fiftyone
```

```python
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=["Bicycle helmet", "Football helmet", "Hat"],
    max_samples=1500,
)
dataset.export(
    export_dir="datasets/open_images_headwear",
    dataset_type=foz.types.YOLOv5Dataset,
)
```

C'est de loin le meilleur rapport effort/rendement : **plus de 1 000 instances
annotees sans annoter une seule image**.

### 2. Hard Hat Workers — pour renforcer les vrais casques

- URL : <https://public.roboflow.com/object-detection/hard-hat-workers>
- **7 041 images**, licence **CC0 (domaine public)** — aucune contrainte
- Classes : `hard hat`, `head` (personne sans casque), `person`

La classe `head` est precieuse : elle fournit des tetes nues annotees, ce qui
aide le modele a ne pas halluciner de casque sur un crane.

### 3. Datasets de casques de velo sur Roboflow Universe

- Bike Helmet Detection : <https://universe.roboflow.com/bike-helmets/bike-helmet-detection-2vdjo>
  (~1 371 images, classes `With Helmet` / `Without Helmet`)
- Recherche par classe : <https://universe.roboflow.com/search?q=class%3Ahelmet>

La classe `With Helmet` se remappe directement en `Non-Safety Headwear`.

### 4. Ce qui n'existe pas

Deux recherches sur Roboflow Universe le confirment : **aucun dataset public ne
separe casque de chantier et casque de velo en classes distinctes.** Tous font
du binaire casque / pas-casque. C'est precisement le vide que ce plan comble,
et c'est pourquoi l'assemblage de sources est necessaire.

> Note : les pages Roboflow Universe renvoient une erreur 403 aux outils
> automatiques. Les effectifs et classes ci-dessus proviennent des resultats de
> recherche et non d'une lecture directe des pages — a reverifier dans un
> navigateur avant de lancer un telechargement.

---

## Faisabilite : toutes les distinctions ne se valent pas

| Classe | Faisabilite | Recommandation |
|--------|-------------|----------------|
| `Non-Safety Headwear` | **Elevee** — differences geometriques nettes (aerations, jugulaire, forme de coque) | **Commencer par la** |
| `Non-Safety Vest` | **Moyenne** — les bandes retroreflechissantes sont un signal fort, mais une veste orange sans bandes reste confondue de loin, et de nuit le retroreflechissant sature | Deuxieme etape |
| `Non-Safety Footwear` | **Faible** — l'embout d'acier est **interne**. A 73 px de cote (mediane de votre dataset), aucun modele ne peut trancher ; un humain non plus | **Ne pas investir** |

Un classificateur entraine sur les chaussures apprendrait des correlations
trompeuses — couleur, marque, contexte — et echouerait silencieusement en
production. C'est pire qu'une absence de detection, parce que cela donne une
fausse assurance.

---

## Decisions a figer AVANT d'annoter

Sans reponses ecrites, vos annotateurs seront incoherents et le modele
apprendra du bruit. Ces questions relevent de votre politique HSE, pas du modele :

- [ ] Un **casque de moto** sur chantier est-il conforme ?
- [ ] Une **casquette anti-heurt** (bump cap) compte-t-elle comme casque ?
- [ ] Un **gilet orange sans bandes retroreflechissantes** est-il conforme ?
- [ ] Un **bonnet porte sous le casque** doit-il etre annote ?
- [ ] Que faire d'un casque **tenu a la main** ou **pose** ?
- [ ] Un casque de chantier **non attache** est-il conforme ?

## Regles d'annotation recommandees

1. Annoter le couvre-chef **porte sur la tete**, pas celui pose ou tenu.
2. Une seule classe par objet : jamais `Safety Helmet` **et**
   `Non-Safety Headwear` sur la meme boite.
3. Boite serree sur l'objet, sans inclure le visage.
4. En cas de doute sur le type, **ne pas annoter** plutot que deviner : une
   annotation incertaine nuit plus qu'elle n'apporte.
5. Annoter meme les objets partiellement occultes s'ils restent identifiables.

---

## Mise en oeuvre

Le projet est deja outille. Le schema etendu a dix classes est defini dans
[`src/ppe_detection/taxonomy.py`](../src/ppe_detection/taxonomy.py), avec les
sept classes d'origine conservant leurs identifiants — un dataset etendu reste
donc retro-compatible.

### Assembler les sources

```powershell
# 1. Casques de velo depuis un dataset Roboflow
python -m ppe_detection.dataset_merge `
  --source datasets/bike-helmets/data.yaml `
  --target artifacts/dataset_extended `
  --map "With Helmet=Non-Safety Headwear" `
  --name bike-helmets

# 2. Couvre-chefs depuis Open Images
python -m ppe_detection.dataset_merge `
  --source datasets/open_images_headwear/data.yaml `
  --target artifacts/dataset_extended `
  --map "Bicycle helmet=Non-Safety Headwear" `
  --map "Football helmet=Non-Safety Headwear" `
  --map "Hat=Non-Safety Headwear" `
  --name open-images

# 3. Votre dataset EPI existant
python -m ppe_detection.dataset_merge `
  --source artifacts/dataset_detection/data.yaml `
  --target artifacts/dataset_extended `
  --map "Person=Person" --map "Safety Helmet=Safety Helmet" `
  --map "Safety Vest=Safety Vest" --map "Safety Shoes=Safety Shoes" `
  --map "Safety Gloves=Safety Gloves" --map "Safety Harness=Safety Harness" `
  --map "Face Mask=Face Mask" `
  --name ppe-original
```

Ajoutez `--dry-run` pour verifier le remappage sans rien ecrire. Le rapport
signale les classes restees sans aucune instance — un modele entraine ainsi ne
pourrait jamais les predire.

### Activer la contre-preuve

Une fois le modele reentraine sur les dix classes, decommentez dans
[`configs/inference.yaml`](../configs/inference.yaml) :

```yaml
counter_evidence:
  Safety Helmet: [Non-Safety Headwear]
  Safety Vest:   [Non-Safety Vest]
```

Le verdict de conformite distingue alors deux niveaux de preuve :

- `evidence: absence` — aucun casque detecte. **Peut** resulter d'un faux
  negatif du detecteur.
- `evidence: observed` — un couvre-chef non conforme a ete vu. **Violation
  constatee**, avec le detail dans le champ `violations`.

Une contre-preuve prime sur le test d'observabilite : apercevoir l'objet prouve
que la zone est visible. Une personne tronquee par le bord du cadre, qui serait
autrement classee `indeterminate`, devient `non_compliant` des lors qu'on
distingue sa casquette.

---

## Sequence recommandee

1. **Figer les definitions de classes** (reunion HSE, une heure).
2. **Telecharger Open Images** — casques de velo, casques de football, chapeaux.
   Environ 1 000 instances sans annoter.
3. **Fusionner** avec votre dataset via `dataset_merge`.
4. **Reentrainer** sur huit classes (les sept d'origine + `Non-Safety Headwear`).
5. **Evaluer specifiquement** sur les images sosies : le casque de velo
   doit-il encore etre detecte comme `Safety Helmet` ?
6. Si concluant, **etendre au gilet**. Laisser les chaussures de cote.

Le point 5 est le juge de paix. Les quatre images de test sont conservees dans
`artifacts/samples/lookalikes/` — elles constituent le noyau d'un jeu de
validation dedie, a completer jusqu'a une cinquantaine d'images.
