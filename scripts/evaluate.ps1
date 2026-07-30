<#
.SYNOPSIS
    Evalue un modele entraine sur le split de validation puis de test.

.DESCRIPTION
    Respecte le protocole methodologique : la validation sert au choix des
    seuils et hyperparametres, le test n'est utilise qu'une fois ces choix
    arretes, pour l'estimation finale.

.PARAMETER Weights
    Poids a evaluer (defaut : artifacts/models/best.pt).

.PARAMETER Split
    Split a evaluer : valid, test ou both (defaut : both).

.PARAMETER Data
    data.yaml du dataset (defaut : artifacts/dataset_detection/data.yaml).

.EXAMPLE
    .\scripts\evaluate.ps1
    .\scripts\evaluate.ps1 -Split valid
    .\scripts\evaluate.ps1 -Weights artifacts/models/smoke_best.pt -Split test
#>
[CmdletBinding()]
param(
    [string]$Weights = "artifacts/models/best.pt",
    [ValidateSet("valid", "test", "both")]
    [string]$Split = "both",
    [string]$Data = "artifacts/dataset_detection/data.yaml",
    [int]$Batch = 16
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Environnement introuvable. Executez d'abord : .\scripts\setup.ps1"
}
if (-not (Test-Path $Weights)) {
    throw @"
Poids introuvables : $Weights
Entrainez d'abord un modele (.\scripts\train.ps1) ou indiquez -Weights.
"@
}

$splits = if ($Split -eq "both") { @("valid", "test") } else { @($Split) }

foreach ($s in $splits) {
    Write-Host "`n=== Evaluation sur le split '$s' ===" -ForegroundColor Cyan
    & $Python -m ppe_detection.evaluate `
        --weights $Weights `
        --data $Data `
        --split $s `
        --batch $Batch
    if ($LASTEXITCODE -ne 0) { throw "L'evaluation sur '$s' a echoue." }
}

Write-Host "`n=== Evaluation terminee ===" -ForegroundColor Green
Write-Host "Rapports : artifacts\reports\evaluation_*.md"
Write-Host "Courbes  : artifacts\runs\val\"
