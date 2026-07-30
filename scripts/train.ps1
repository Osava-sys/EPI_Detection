<#
.SYNOPSIS
    Lance l'entrainement complet du detecteur EPI.

.DESCRIPTION
    Entrainement long (plusieurs heures selon le materiel). Verifie au prealable
    que le dataset normalise existe et avertit si aucun smoke test n'a ete
    execute.

.PARAMETER Config
    Fichier de configuration (defaut : configs/train.yaml).

.PARAMETER Epochs
    Surcharge le nombre d'epoques.

.PARAMETER Batch
    Surcharge la taille de lot (-1 = automatique).

.PARAMETER Model
    Surcharge le modele de depart (yolo26n.pt, yolo26s.pt, yolo26m.pt...).

.PARAMETER Name
    Nom de l'experience.

.PARAMETER Resume
    Chemin d'un last.pt a reprendre.

.EXAMPLE
    .\scripts\train.ps1
    .\scripts\train.ps1 -Epochs 150 -Batch 24
    .\scripts\train.ps1 -Resume artifacts/runs/ppe_yolo26s/weights/last.pt
#>
[CmdletBinding()]
param(
    [string]$Config = "configs/train.yaml",
    [int]$Epochs = 0,
    [double]$Batch = 0,
    [string]$Model = "",
    [string]$Name = "",
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Environnement introuvable. Executez d'abord : .\scripts\setup.ps1"
}

if (-not (Test-Path "artifacts\dataset_detection\data.yaml")) {
    throw @"
Dataset normalise introuvable.
Executez d'abord : .\scripts\audit_dataset.ps1
"@
}

if (-not (Test-Path "artifacts\models\smoke_best.pt")) {
    Write-Host "AVERTISSEMENT : aucun smoke test n'a ete detecte." -ForegroundColor Yellow
    Write-Host "Il est fortement recommande de valider le pipeline avant un entrainement long :" -ForegroundColor Yellow
    Write-Host "  .\scripts\smoke_train.ps1" -ForegroundColor Yellow
    Write-Host ""
}

$trainArgs = @("-m", "ppe_detection.train", "--config", $Config)
if ($Epochs -gt 0)                  { $trainArgs += @("--epochs", $Epochs) }
if ($Batch -ne 0)                   { $trainArgs += @("--batch", $Batch) }
if (-not [string]::IsNullOrWhiteSpace($Model))  { $trainArgs += @("--model", $Model) }
if (-not [string]::IsNullOrWhiteSpace($Name))   { $trainArgs += @("--name", $Name) }
if (-not [string]::IsNullOrWhiteSpace($Resume)) { $trainArgs += @("--resume", $Resume) }

Write-Host "=== ENTRAINEMENT COMPLET ===" -ForegroundColor Cyan
Write-Host "Cette operation peut durer plusieurs heures."
Write-Host "Ctrl+C interrompt proprement ; reprenez ensuite avec -Resume .../weights/last.pt`n"

$started = Get-Date
& $Python @trainArgs
$code = $LASTEXITCODE
$elapsed = (Get-Date) - $started

if ($code -ne 0) {
    throw "L'entrainement a echoue (code $code) apres $($elapsed.ToString('hh\:mm\:ss'))."
}

Write-Host "`n=== ENTRAINEMENT TERMINE en $($elapsed.ToString('hh\:mm\:ss')) ===" -ForegroundColor Green
Write-Host "Poids : artifacts\models\best.pt"
Write-Host "Etape suivante : .\scripts\evaluate.ps1"
