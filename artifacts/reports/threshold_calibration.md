# Calibration des seuils de confiance par classe

- **Genere le** : 2026-08-04T11:14:19.502589+00:00
- **Poids** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\models\best.pt`
- **Split** : `valid` (1277 images)
- **Seuil de reference** : 0.25
- **IoU d'appariement** : 0.5

> Les seuils sont choisis sur la **validation** uniquement. Les appliquer
> puis mesurer sur le test reste un protocole valide ; les choisir sur le
> test ne le serait pas.

## Seuils retenus

| Classe | Objets | Seuil retenu | F1 obtenu | F1 au seuil uniforme | Gain | Precision | Rappel |
|--------|--------|--------------|-----------|----------------------|------|-----------|--------|
| Face Mask | 131 | **0.25** | 0.8205 | 0.8205 | +0.0000 | 0.7887 | 0.8550 |
| Person | 1500 | **0.35** | 0.8649 | 0.8594 | +0.0055 | 0.8572 | 0.8727 |
| Safety Gloves | 410 | **0.3** | 0.5912 | 0.5886 | +0.0027 | 0.7037 | 0.5098 |
| Safety Harness | 249 | **0.4** | 0.6745 | 0.6695 | +0.0050 | 0.8090 | 0.5783 |
| Safety Helmet | 1112 | **0.3** | 0.7888 | 0.7713 | +0.0175 | 0.7801 | 0.7977 |
| Safety Shoes | 992 | **0.3** | 0.7154 | 0.7154 | +0.0000 | 0.7481 | 0.6855 |
| Safety Vest | 454 | **0.45** | 0.8369 | 0.8211 | +0.0159 | 0.8676 | 0.8084 |

## Synthese

- F1 macro au seuil uniforme 0.25 : **0.7494**
- F1 macro avec seuils calibres : **0.7560**
- Gain macro : **+0.0067**

## Extrait YAML a reporter dans `configs/inference.yaml`

```yaml
inference:
  class_conf:
    Face Mask: 0.25
    Person: 0.35
    Safety Gloves: 0.3
    Safety Harness: 0.4
    Safety Helmet: 0.3
    Safety Shoes: 0.3
    Safety Vest: 0.45
```
