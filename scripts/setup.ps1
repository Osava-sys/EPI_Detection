<#
.SYNOPSIS
    Cree l'environnement Python isole et installe les dependances du projet.

.DESCRIPTION
    Cree un venv avec une version de Python officiellement supportee par
    PyTorch (3.10 a 3.13), installe PyTorch (CPU ou CUDA), puis le projet en
    mode editable.

    Le choix de la roue CUDA n'est pas arbitraire : il est deduit de
    l'architecture du GPU detecte. Les GPU Blackwell (RTX 50xx, capacite 12.0)
    exigent une compilation cu128 ou superieure.

.PARAMETER PythonVersion
    Version de Python a utiliser pour le venv (defaut : 3.12).

.PARAMETER Cuda
    Variante CUDA a installer : cu128, cu126 ou cpu (defaut : cu128).

.PARAMETER Force
    Recree l'environnement meme s'il existe deja.

.EXAMPLE
    .\scripts\setup.ps1
    .\scripts\setup.ps1 -Cuda cpu
    .\scripts\setup.ps1 -PythonVersion 3.11 -Force
#>
[CmdletBinding()]
param(
    [string]$PythonVersion = "3.12",
    [ValidateSet("cu128", "cu126", "cpu")]
    [string]$Cuda = "cu128",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "=== Configuration de l'environnement PPE Detection ===" -ForegroundColor Cyan
Write-Host "Racine du projet : $ProjectRoot"

# --- 1. Verification de l'interpreteur demande -----------------------------
$launcher = Get-Command py -ErrorAction SilentlyContinue
if (-not $launcher) {
    throw "Le lanceur Python 'py' est introuvable. Installez Python depuis https://www.python.org/downloads/"
}

$available = & py -0p 2>&1 | Out-String
if ($available -notmatch [regex]::Escape("-V:$PythonVersion")) {
    Write-Host "Versions disponibles :" -ForegroundColor Yellow
    Write-Host $available
    throw @"
Python $PythonVersion est introuvable.
PyTorch ne publie pas encore de roues pour les toutes dernieres versions de
Python : installez Python 3.12 (recommande) depuis python.org, puis relancez.
"@
}

# --- 2. Creation du venv ---------------------------------------------------
$VenvPath = Join-Path $ProjectRoot ".venv"
if ((Test-Path $VenvPath) -and $Force) {
    Write-Host "Suppression de l'environnement existant..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $VenvPath
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creation du venv avec Python $PythonVersion..." -ForegroundColor Green
    & py "-$PythonVersion" -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Echec de la creation du venv." }
} else {
    Write-Host "Environnement existant reutilise (utilisez -Force pour le recreer)." -ForegroundColor Yellow
}

$Python = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Interpreteur introuvable : $Python" }

& $Python -m pip install --upgrade pip --quiet
Write-Host ("Python du venv : " + (& $Python --version)) -ForegroundColor Green

# --- 3. Detection du GPU ---------------------------------------------------
$UseCuda = $Cuda -ne "cpu"
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($UseCuda) {
    if (-not $smi) {
        Write-Host "nvidia-smi introuvable : bascule sur l'installation CPU." -ForegroundColor Yellow
        $UseCuda = $false
    } else {
        Write-Host "--- GPU detecte ---" -ForegroundColor Cyan
        & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    }
}

# --- 4. Installation de PyTorch -------------------------------------------
if ($UseCuda) {
    $IndexUrl = "https://download.pytorch.org/whl/$Cuda"
    Write-Host "Installation de PyTorch ($Cuda)... (telechargement volumineux)" -ForegroundColor Green
    & $Python -m pip install torch torchvision --index-url $IndexUrl
} else {
    Write-Host "Installation de PyTorch (CPU)..." -ForegroundColor Green
    & $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { throw "Echec de l'installation de PyTorch." }

# --- 5. Installation du projet --------------------------------------------
Write-Host "Installation du projet et de ses dependances..." -ForegroundColor Green
& $Python -m pip install -e ".[api,ui,export,audit,dev]"
if ($LASTEXITCODE -ne 0) { throw "Echec de l'installation des dependances du projet." }

# --- 6. Verification -------------------------------------------------------
Write-Host "`n=== Verification ===" -ForegroundColor Cyan
& $Python -c @"
import torch, ultralytics, cv2
print(f'torch         : {torch.__version__}')
print(f'ultralytics   : {ultralytics.__version__}')
print(f'opencv        : {cv2.__version__}')
print(f'CUDA dispo    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f'GPU           : {torch.cuda.get_device_name(0)}')
    print(f'Capacite      : sm_{cap[0]}{cap[1]}')
    print(f'Archs compilees: {torch.cuda.get_arch_list()}')
    if f'sm_{cap[0]}{cap[1]}' not in torch.cuda.get_arch_list():
        print()
        print('ATTENTION : cette build de PyTorch ne contient pas de noyaux pour')
        print('votre GPU. Reinstallez avec une variante CUDA plus recente,')
        print('par exemple : .\\scripts\\setup.ps1 -Cuda cu128 -Force')
"@

Write-Host "`n=== Environnement pret ===" -ForegroundColor Green
Write-Host "Activez-le avec :  .\.venv\Scripts\Activate.ps1"
Write-Host "Etape suivante  :  .\scripts\audit_dataset.ps1"
