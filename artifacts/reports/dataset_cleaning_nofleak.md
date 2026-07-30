# Rapport de normalisation du dataset (detection)

- **Genere le** : 2026-07-30T10:45:03.191887+00:00
- **Source** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\data.yaml` (non modifiee)
- **Destination** : `C:\Users\LENOVO LEGION\Downloads\PPE Detection Project.yolo26\artifacts\dataset_detection_nofleak`
- **Mode de copie** : copy
- **Simulation (dry-run)** : non

## Statistiques de conversion

| Indicateur                               | Valeur |
|------------------------------------------|--------|
| Images ecrites                           | 7000   |
| Images ignorees (label absent)           | 0      |
| Fichiers de labels ecrits                | 7000   |
| Lignes lues                              | 25542  |
| Lignes conservees                        | 25542  |
| **Lignes converties depuis un polygone** | 349    |
| Lignes avec derive numerique corrigee    | 457    |
| Boites rognees sur les bords de l'image  | 0      |
| Lignes exclues                           | 0      |
| Fichiers de labels vides (negatifs)      | 0      |

## Repartition des images produites

| Split | Images |
|-------|--------|
| test  | 578    |
| train | 5145   |
| valid | 1277   |

## Instances par classe et par split

| Classe         | test | train | valid | Total |
|----------------|------|-------|-------|-------|
| Face Mask      | 61   | 596   | 131   | 788   |
| Person         | 714  | 5435  | 1500  | 7649  |
| Safety Gloves  | 137  | 1625  | 410   | 2172  |
| Safety Harness | 111  | 815   | 249   | 1175  |
| Safety Helmet  | 495  | 3842  | 1112  | 5449  |
| Safety Shoes   | 396  | 4487  | 992   | 5875  |
| Safety Vest    | 228  | 1752  | 454   | 2434  |

## Regroupement anti-fuite

Le regroupement par image source a ete applique : **488 fichier(s)** deplace(s) pour que toutes les variantes d'une meme photo source restent dans un seul split (390 groupe(s) concerne(s)).

## Evenements par type

| Evenement         | Occurrences |
|-------------------|-------------|
| converted_polygon | 349         |

## Journal detaille (premiers 349 evenements)

| Type              | Split | Fichier                                              | Ligne | Detail               |
|-------------------|-------|------------------------------------------------------|-------|----------------------|
| converted_polygon | train | -1013-_png_jpg.rf.TgF97cwTfIx3l5TcgxIm.txt           | 12    | 48 sommets -> boite  |
| converted_polygon | train | -3058-_png_jpg.rf.oySRYRnbJFEojJSK6PUa.txt           | 3     | 9 sommets -> boite   |
| converted_polygon | train | -3058-_png_jpg.rf.oySRYRnbJFEojJSK6PUa.txt           | 4     | 14 sommets -> boite  |
| converted_polygon | train | -3058-_png_jpg.rf.oySRYRnbJFEojJSK6PUa.txt           | 5     | 12 sommets -> boite  |
| converted_polygon | train | -3136-_png_jpg.rf.39nBpvq3BEn7pHy0ntoK.txt           | 2     | 27 sommets -> boite  |
| converted_polygon | train | -3163-_png_jpg.rf.7rI5peLPBqrxa9orIjcc.txt           | 4     | 77 sommets -> boite  |
| converted_polygon | train | -3175-_png_jpg.rf.i93Vr4ciRdELCUcnyuzh.txt           | 6     | 9 sommets -> boite   |
| converted_polygon | train | -3175-_png_jpg.rf.i93Vr4ciRdELCUcnyuzh.txt           | 7     | 11 sommets -> boite  |
| converted_polygon | train | -3175-_png_jpg.rf.i93Vr4ciRdELCUcnyuzh.txt           | 8     | 9 sommets -> boite   |
| converted_polygon | train | -3175-_png_jpg.rf.i93Vr4ciRdELCUcnyuzh.txt           | 9     | 7 sommets -> boite   |
| converted_polygon | train | -3175-_png_jpg.rf.i93Vr4ciRdELCUcnyuzh.txt           | 10    | 10 sommets -> boite  |
| converted_polygon | train | -3175-_png_jpg.rf.i93Vr4ciRdELCUcnyuzh.txt           | 11    | 10 sommets -> boite  |
| converted_polygon | train | -3175-_png_jpg.rf.i93Vr4ciRdELCUcnyuzh.txt           | 12    | 13 sommets -> boite  |
| converted_polygon | train | -3175-_png_jpg.rf.i93Vr4ciRdELCUcnyuzh.txt           | 13    | 11 sommets -> boite  |
| converted_polygon | train | -3175-_png_jpg.rf.i93Vr4ciRdELCUcnyuzh.txt           | 14    | 8 sommets -> boite   |
| converted_polygon | train | -3181-_png_jpg.rf.JsAinO6lKc2QInVAxhbF.txt           | 2     | 37 sommets -> boite  |
| converted_polygon | train | -3181-_png_jpg.rf.JsAinO6lKc2QInVAxhbF.txt           | 3     | 25 sommets -> boite  |
| converted_polygon | train | -3181-_png_jpg.rf.JsAinO6lKc2QInVAxhbF.txt           | 4     | 192 sommets -> boite |
| converted_polygon | train | -3181-_png_jpg.rf.JsAinO6lKc2QInVAxhbF.txt           | 5     | 40 sommets -> boite  |
| converted_polygon | train | -3187-_png_jpg.rf.RRZdIexKgdqV9YW7SpgU.txt           | 2     | 37 sommets -> boite  |
| converted_polygon | train | -3187-_png_jpg.rf.RRZdIexKgdqV9YW7SpgU.txt           | 3     | 19 sommets -> boite  |
| converted_polygon | train | -323-_png_jpg.rf.hOu1TZrD996mYV7uD2pX.txt            | 6     | 14 sommets -> boite  |
| converted_polygon | train | -323-_png_jpg.rf.hOu1TZrD996mYV7uD2pX.txt            | 7     | 21 sommets -> boite  |
| converted_polygon | train | -323-_png_jpg.rf.hOu1TZrD996mYV7uD2pX.txt            | 8     | 41 sommets -> boite  |
| converted_polygon | train | -323-_png_jpg.rf.hOu1TZrD996mYV7uD2pX.txt            | 9     | 17 sommets -> boite  |
| converted_polygon | train | -323-_png_jpg.rf.hOu1TZrD996mYV7uD2pX.txt            | 10    | 18 sommets -> boite  |
| converted_polygon | train | -323-_png_jpg.rf.hOu1TZrD996mYV7uD2pX.txt            | 11    | 23 sommets -> boite  |
| converted_polygon | train | -323-_png_jpg.rf.hOu1TZrD996mYV7uD2pX.txt            | 12    | 23 sommets -> boite  |
| converted_polygon | train | -323-_png_jpg.rf.hOu1TZrD996mYV7uD2pX.txt            | 13    | 41 sommets -> boite  |
| converted_polygon | train | -323-_png_jpg.rf.hOu1TZrD996mYV7uD2pX.txt            | 14    | 25 sommets -> boite  |
| converted_polygon | train | -323-_png_jpg.rf.hOu1TZrD996mYV7uD2pX.txt            | 15    | 11 sommets -> boite  |
| converted_polygon | train | -3235-_png_jpg.rf.Dfg788d6HMffmAQjPkqZ.txt           | 6     | 19 sommets -> boite  |
| converted_polygon | train | -3235-_png_jpg.rf.Dfg788d6HMffmAQjPkqZ.txt           | 7     | 13 sommets -> boite  |
| converted_polygon | train | -3235-_png_jpg.rf.Dfg788d6HMffmAQjPkqZ.txt           | 8     | 11 sommets -> boite  |
| converted_polygon | train | -3235-_png_jpg.rf.Dfg788d6HMffmAQjPkqZ.txt           | 10    | 25 sommets -> boite  |
| converted_polygon | train | -3436-_png_jpg.rf.wZChl8Ifsj8XxAU5vuqR.txt           | 5     | 9 sommets -> boite   |
| converted_polygon | train | -3436-_png_jpg.rf.wZChl8Ifsj8XxAU5vuqR.txt           | 6     | 13 sommets -> boite  |
| converted_polygon | train | -3436-_png_jpg.rf.wZChl8Ifsj8XxAU5vuqR.txt           | 7     | 44 sommets -> boite  |
| converted_polygon | train | -3436-_png_jpg.rf.wZChl8Ifsj8XxAU5vuqR.txt           | 8     | 14 sommets -> boite  |
| converted_polygon | train | -3436-_png_jpg.rf.wZChl8Ifsj8XxAU5vuqR.txt           | 9     | 33 sommets -> boite  |
| converted_polygon | train | -3436-_png_jpg.rf.wZChl8Ifsj8XxAU5vuqR.txt           | 10    | 21 sommets -> boite  |
| converted_polygon | train | -3567-_png_jpg.rf.9zui5AnJUdHqJbwwfxnK.txt           | 8     | 15 sommets -> boite  |
| converted_polygon | train | -3567-_png_jpg.rf.9zui5AnJUdHqJbwwfxnK.txt           | 9     | 67 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 11    | 35 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 12    | 23 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 13    | 40 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 14    | 29 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 15    | 42 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 16    | 34 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 17    | 12 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 18    | 26 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 19    | 11 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 20    | 10 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 21    | 65 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 22    | 15 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 23    | 91 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 24    | 154 sommets -> boite |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 25    | 83 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 26    | 91 sommets -> boite  |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 27    | 100 sommets -> boite |
| converted_polygon | train | -3799-_png_jpg.rf.ljfxhQjfY6mf33Cymqg3.txt           | 28    | 9 sommets -> boite   |
| converted_polygon | train | -3854-_png_jpg.rf.j2LamVjBhB9bdL23CTWt.txt           | 6     | 5 sommets -> boite   |
| converted_polygon | train | -3854-_png_jpg.rf.j2LamVjBhB9bdL23CTWt.txt           | 7     | 17 sommets -> boite  |
| converted_polygon | train | -3854-_png_jpg.rf.j2LamVjBhB9bdL23CTWt.txt           | 8     | 20 sommets -> boite  |
| converted_polygon | train | -3854-_png_jpg.rf.j2LamVjBhB9bdL23CTWt.txt           | 9     | 26 sommets -> boite  |
| converted_polygon | train | -3854-_png_jpg.rf.j2LamVjBhB9bdL23CTWt.txt           | 10    | 22 sommets -> boite  |
| converted_polygon | train | -4394-_png_jpg.rf.NiXta7QlCD6G7YcSaD4X.txt           | 3     | 64 sommets -> boite  |
| converted_polygon | train | -4394-_png_jpg.rf.NiXta7QlCD6G7YcSaD4X.txt           | 4     | 18 sommets -> boite  |
| converted_polygon | train | -4394-_png_jpg.rf.NiXta7QlCD6G7YcSaD4X.txt           | 5     | 26 sommets -> boite  |
| converted_polygon | train | -4596-_png_jpg.rf.UJ5alnIcSUbPmKlyZixO.txt           | 1     | 99 sommets -> boite  |
| converted_polygon | train | -4596-_png_jpg.rf.UJ5alnIcSUbPmKlyZixO.txt           | 2     | 44 sommets -> boite  |
| converted_polygon | train | -4596-_png_jpg.rf.UJ5alnIcSUbPmKlyZixO.txt           | 3     | 29 sommets -> boite  |
| converted_polygon | train | -4814-_png_jpg.rf.n5z6bhvYWhdzSzQ28XYI.txt           | 4     | 46 sommets -> boite  |
| converted_polygon | train | -4814-_png_jpg.rf.n5z6bhvYWhdzSzQ28XYI.txt           | 5     | 32 sommets -> boite  |
| converted_polygon | train | -4814-_png_jpg.rf.n5z6bhvYWhdzSzQ28XYI.txt           | 6     | 50 sommets -> boite  |
| converted_polygon | train | -4814-_png_jpg.rf.n5z6bhvYWhdzSzQ28XYI.txt           | 7     | 31 sommets -> boite  |
| converted_polygon | train | -4814-_png_jpg.rf.n5z6bhvYWhdzSzQ28XYI.txt           | 8     | 24 sommets -> boite  |
| converted_polygon | train | -4814-_png_jpg.rf.n5z6bhvYWhdzSzQ28XYI.txt           | 9     | 36 sommets -> boite  |
| converted_polygon | train | -4814-_png_jpg.rf.n5z6bhvYWhdzSzQ28XYI.txt           | 10    | 27 sommets -> boite  |
| converted_polygon | train | -4814-_png_jpg.rf.n5z6bhvYWhdzSzQ28XYI.txt           | 11    | 27 sommets -> boite  |
| converted_polygon | train | -4814-_png_jpg.rf.n5z6bhvYWhdzSzQ28XYI.txt           | 12    | 23 sommets -> boite  |
| converted_polygon | train | -605-_png_jpg.rf.xh42DofgSm9HEgOBBsmE.txt            | 4     | 23 sommets -> boite  |
| converted_polygon | train | -605-_png_jpg.rf.xh42DofgSm9HEgOBBsmE.txt            | 5     | 27 sommets -> boite  |
| converted_polygon | train | -638-_png_jpg.rf.JEzgjRNNkCUg1jl0YzQt.txt            | 7     | 51 sommets -> boite  |
| converted_polygon | train | -638-_png_jpg.rf.JEzgjRNNkCUg1jl0YzQt.txt            | 8     | 14 sommets -> boite  |
| converted_polygon | train | -638-_png_jpg.rf.JEzgjRNNkCUg1jl0YzQt.txt            | 9     | 60 sommets -> boite  |
| converted_polygon | train | -638-_png_jpg.rf.JEzgjRNNkCUg1jl0YzQt.txt            | 10    | 40 sommets -> boite  |
| converted_polygon | train | -638-_png_jpg.rf.JEzgjRNNkCUg1jl0YzQt.txt            | 11    | 32 sommets -> boite  |
| converted_polygon | train | -638-_png_jpg.rf.JEzgjRNNkCUg1jl0YzQt.txt            | 12    | 29 sommets -> boite  |
| converted_polygon | train | -638-_png_jpg.rf.JEzgjRNNkCUg1jl0YzQt.txt            | 13    | 49 sommets -> boite  |
| converted_polygon | train | -638-_png_jpg.rf.JEzgjRNNkCUg1jl0YzQt.txt            | 14    | 14 sommets -> boite  |
| converted_polygon | train | -638-_png_jpg.rf.JEzgjRNNkCUg1jl0YzQt.txt            | 15    | 19 sommets -> boite  |
| converted_polygon | train | -638-_png_jpg.rf.JEzgjRNNkCUg1jl0YzQt.txt            | 16    | 17 sommets -> boite  |
| converted_polygon | train | 000037_jpg.rf.QmVt1COcTHNRtkShLSut.txt               | 2     | 59 sommets -> boite  |
| converted_polygon | train | 000037_jpg.rf.QmVt1COcTHNRtkShLSut.txt               | 3     | 97 sommets -> boite  |
| converted_polygon | train | 000441_jpg.rf.1L1rgTZ5eFjyW20xhpCb.txt               | 5     | 59 sommets -> boite  |
| converted_polygon | train | 000441_jpg.rf.1L1rgTZ5eFjyW20xhpCb.txt               | 6     | 25 sommets -> boite  |
| converted_polygon | train | 001020_jpg.rf.RoIVSIkEF7g4BqfyGlVv.txt               | 3     | 38 sommets -> boite  |
| converted_polygon | train | 001020_jpg.rf.RoIVSIkEF7g4BqfyGlVv.txt               | 4     | 90 sommets -> boite  |
| converted_polygon | train | 001020_jpg.rf.RoIVSIkEF7g4BqfyGlVv.txt               | 5     | 23 sommets -> boite  |
| converted_polygon | train | 001020_jpg.rf.RoIVSIkEF7g4BqfyGlVv.txt               | 6     | 29 sommets -> boite  |
| converted_polygon | train | 001020_jpg.rf.RoIVSIkEF7g4BqfyGlVv.txt               | 7     | 32 sommets -> boite  |
| converted_polygon | train | 001859_jpg.rf.GXYhB03PAMa3wO1UEncD.txt               | 4     | 35 sommets -> boite  |
| converted_polygon | train | 001859_jpg.rf.GXYhB03PAMa3wO1UEncD.txt               | 5     | 30 sommets -> boite  |
| converted_polygon | train | 08165739_jpg.rf.OVB7uZYjCthujn1kqbld.txt             | 1     | 18 sommets -> boite  |
| converted_polygon | train | 08165739_jpg.rf.OVB7uZYjCthujn1kqbld.txt             | 2     | 27 sommets -> boite  |
| converted_polygon | train | 08165739_jpg.rf.OVB7uZYjCthujn1kqbld.txt             | 3     | 18 sommets -> boite  |
| converted_polygon | train | 08165739_jpg.rf.OVB7uZYjCthujn1kqbld.txt             | 4     | 14 sommets -> boite  |
| converted_polygon | train | 08165739_jpg.rf.OVB7uZYjCthujn1kqbld.txt             | 5     | 19 sommets -> boite  |
| converted_polygon | train | 08165739_jpg.rf.OVB7uZYjCthujn1kqbld.txt             | 6     | 12 sommets -> boite  |
| converted_polygon | train | 1000x-1_jpg.rf.bX4XfUVUKwg1VQOH9lHR.txt              | 1     | 14 sommets -> boite  |
| converted_polygon | train | 1000x-1_jpg.rf.bX4XfUVUKwg1VQOH9lHR.txt              | 2     | 14 sommets -> boite  |
| converted_polygon | train | 1000x-1_jpg.rf.bX4XfUVUKwg1VQOH9lHR.txt              | 3     | 18 sommets -> boite  |
| converted_polygon | train | 1000x-1_jpg.rf.bX4XfUVUKwg1VQOH9lHR.txt              | 5     | 37 sommets -> boite  |
| converted_polygon | train | 1000x-1_jpg.rf.bX4XfUVUKwg1VQOH9lHR.txt              | 6     | 22 sommets -> boite  |
| converted_polygon | train | 1000x-1_jpg.rf.bX4XfUVUKwg1VQOH9lHR.txt              | 7     | 24 sommets -> boite  |
| converted_polygon | train | 101307074_544e234e97_jpg.rf.F6lJ6g1qramzCGawXf6o.txt | 3     | 203 sommets -> boite |
| converted_polygon | train | 101307074_544e234e97_jpg.rf.JSavjjYtVYlD9DSWw5Ij.txt | 3     | 196 sommets -> boite |
| converted_polygon | train | 101307074_544e234e97_jpg.rf.p5sd9cjoPnkYwFYeXvEQ.txt | 3     | 252 sommets -> boite |
| converted_polygon | train | 101307074_544e234e97_jpg.rf.QZYoCvTbTz6NFhpguB3E.txt | 3     | 195 sommets -> boite |
