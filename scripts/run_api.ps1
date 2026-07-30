<#
.SYNOPSIS
    Demarre l'API REST de detection d'EPI.

.DESCRIPTION
    Lance uvicorn avec le modele indique. La configuration passe par des
    variables d'environnement, ce qui evite tout secret ou chemin machine
    code en dur dans le depot.

    Le service demarre meme si les poids sont absents : /health repond alors
    « degraded » et les routes d'inference renvoient 503 avec un message
    explicite, plutot que de faire echouer le demarrage.

.PARAMETER Weights
    Poids a charger (defaut : artifacts/models/best.pt).

.PARAMETER BindAddress
    Adresse d'ecoute (defaut : 127.0.0.1, donc accessible uniquement en local).

.PARAMETER Port
    Port d'ecoute (defaut : 8000).

.PARAMETER Compliance
    Active la couche de conformite EPI dans les reponses.

.PARAMETER Reload
    Rechargement automatique du code (developpement uniquement).

.EXAMPLE
    .\scripts\run_api.ps1
    .\scripts\run_api.ps1 -Weights artifacts/models/smoke_best.pt -Port 8080 -Compliance
#>
[CmdletBinding()]
param(
    [string]$Weights = "artifacts/models/best.pt",
    [string]$BindAddress = "127.0.0.1",
    [int]$Port = 8000,
    [string]$Device = "auto",
    [double]$Conf = 0.25,
    [switch]$Compliance,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Environnement introuvable. Executez d'abord : .\scripts\setup.ps1"
}

if (-not (Test-Path $Weights)) {
    Write-Host "AVERTISSEMENT : poids introuvables ($Weights)." -ForegroundColor Yellow
    Write-Host "L'API demarrera en mode degrade : /health repondra 'degraded'" -ForegroundColor Yellow
    Write-Host "et les routes d'inference renverront 503." -ForegroundColor Yellow
    Write-Host ""
}

$env:PPE_API_WEIGHTS = $Weights
$env:PPE_API_DEVICE = $Device
$env:PPE_API_CONF = $Conf
if ($Compliance) { $env:PPE_API_COMPLIANCE = "true" }

Write-Host "=== API de detection EPI ===" -ForegroundColor Cyan
Write-Host "Poids       : $Weights"
Write-Host "Device      : $Device"
Write-Host "Conformite  : $(if ($Compliance) { 'activee' } else { 'desactivee' })"
Write-Host "URL         : http://${BindAddress}:$Port"
Write-Host "Doc OpenAPI : http://${BindAddress}:$Port/docs"
Write-Host "`nCtrl+C pour arreter.`n"

$uvicornArgs = @(
    "-m", "uvicorn", "ppe_detection.api:app",
    "--host", $BindAddress,
    "--port", $Port
)
if ($Reload) { $uvicornArgs += "--reload" }

& $Python @uvicornArgs
