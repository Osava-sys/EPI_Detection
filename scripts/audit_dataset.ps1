<#
.SYNOPSIS
    Audite le dataset original puis construit et verifie le dataset de detection.

.DESCRIPTION
    Enchaine les trois etapes de preparation des donnees :
      1. audit de l'export Roboflow original (jamais modifie) ;
      2. construction du dataset derive avec labels normalises a 5 champs ;
      3. audit de verification du dataset derive, qui echoue s'il subsiste la
         moindre annotation polygonale.

.PARAMETER Data
    data.yaml du dataset original (defaut : data.yaml).

.PARAMETER Output
    Repertoire du dataset derive (defaut : artifacts/dataset_detection).

.PARAMETER SkipPerceptualHash
    Desactive la recherche de quasi-doublons visuels (audit plus rapide).

.PARAMETER AllowSourceLeak
    Reproduit la repartition train/valid/test exacte de l'export Roboflow, au
    lieu du regroupement anti-fuite applique par defaut. Les metriques obtenues
    seront optimistes : a n'utiliser que pour comparer a des resultats publies
    sur le decoupage d'origine.

.PARAMETER Overwrite
    Remplace un dataset derive existant.

.EXAMPLE
    .\scripts\audit_dataset.ps1
    .\scripts\audit_dataset.ps1 -Overwrite
    .\scripts\audit_dataset.ps1 -AllowSourceLeak -Output artifacts/dataset_detection_roboflow
#>
[CmdletBinding()]
param(
    [string]$Data = "data.yaml",
    [string]$Output = "artifacts/dataset_detection",
    [switch]$SkipPerceptualHash,
    [switch]$AllowSourceLeak,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Environnement introuvable. Executez d'abord : .\scripts\setup.ps1"
}

Write-Host "=== 1/3 — Audit du dataset ORIGINAL ===" -ForegroundColor Cyan
$auditArgs = @(
    "-m", "ppe_detection.dataset_audit",
    "--data", $Data,
    "--output", "artifacts/reports/dataset_audit_original.json"
)
if ($SkipPerceptualHash) { $auditArgs += "--skip-perceptual-hash" }
& $Python @auditArgs
if ($LASTEXITCODE -ne 0) { throw "L'audit du dataset original a echoue." }

Write-Host "`n=== 2/3 — Construction du dataset de DETECTION ===" -ForegroundColor Cyan
$cleanArgs = @(
    "-m", "ppe_detection.dataset_cleaner",
    "--source", $Data,
    "--output", $Output,
    "--mode", "copy"
)
if ($Overwrite)        { $cleanArgs += "--overwrite" }
if ($AllowSourceLeak)  { $cleanArgs += "--allow-source-leak" }
& $Python @cleanArgs
if ($LASTEXITCODE -ne 0) { throw "La normalisation du dataset a echoue." }

Write-Host "`n=== 3/3 — Verification du dataset derive ===" -ForegroundColor Cyan
& $Python -m ppe_detection.dataset_audit `
    --data "$Output/data.yaml" `
    --output "artifacts/reports/dataset_audit_detection.json" `
    --skip-perceptual-hash `
    --fail-on-polygon `
    --fail-on-error
if ($LASTEXITCODE -ne 0) { throw "Le dataset derive contient encore des annotations non conformes." }

Write-Host "`n=== Preparation des donnees terminee ===" -ForegroundColor Green
Write-Host "Rapports : artifacts\reports\"
Write-Host "Dataset  : $Output"
Write-Host "Etape suivante : .\scripts\smoke_train.ps1"
