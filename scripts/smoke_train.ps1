<#
.SYNOPSIS
    Valide le pipeline d'entrainement en quelques minutes.

.DESCRIPTION
    Lance un entrainement tres court (2 epoques sur 4 % du jeu d'entrainement)
    dont le seul but est de prouver que la chaine complete fonctionne :
    chargement du dataset, entrainement, validation, sauvegarde des poids.

    Les metriques obtenues n'ont AUCUNE valeur predictive : elles servent
    uniquement a verifier que le pipeline produit bien des resultats.

    A executer systematiquement avant un entrainement complet.

.EXAMPLE
    .\scripts\smoke_train.ps1
#>
[CmdletBinding()]
param(
    [string]$Config = "configs/train.yaml"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Environnement introuvable. Executez d'abord : .\scripts\setup.ps1"
}

Write-Host "=== SMOKE TEST — validation du pipeline d'entrainement ===" -ForegroundColor Cyan
Write-Host "2 epoques sur une petite fraction des donnees. Duree attendue : 1 a 3 minutes.`n"

& $Python -m ppe_detection.train --config $Config --smoke
if ($LASTEXITCODE -ne 0) {
    throw "Le smoke test a echoue. Corrigez l'erreur avant de lancer l'entrainement complet."
}

$SmokeWeights = Join-Path $ProjectRoot "artifacts\models\smoke_best.pt"
if (-not (Test-Path $SmokeWeights)) {
    throw "Le smoke test s'est termine sans produire de poids : $SmokeWeights est absent."
}

$size = [math]::Round((Get-Item $SmokeWeights).Length / 1MB, 1)
Write-Host "`n=== SMOKE TEST REUSSI ===" -ForegroundColor Green
Write-Host "Poids produits : artifacts\models\smoke_best.pt ($size Mo)"
Write-Host "Run complet    : artifacts\runs\smoke_test\"
Write-Host ""
Write-Host "Les metriques du smoke test ne sont pas representatives." -ForegroundColor Yellow
Write-Host "Lancez maintenant l'entrainement complet : .\scripts\train.ps1"
