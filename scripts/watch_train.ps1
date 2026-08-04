<#
.SYNOPSIS
    Suit l'evolution d'un entrainement en cours, epoque par epoque.

.DESCRIPTION
    Lit le fichier results.csv qu'Ultralytics complete a chaque fin d'epoque et
    affiche les metriques, la meilleure epoque atteinte et une estimation du
    temps restant.

    Preferable a un `tail` du log brut : celui-ci contient des barres de
    progression ANSI qui se relisent mal.

    Ctrl+C interrompt l'affichage sans affecter l'entrainement.

.PARAMETER Name
    Nom du run a suivre (defaut : le run modifie le plus recemment).

.PARAMETER Rows
    Nombre d'epoques affichees (defaut : 12).

.PARAMETER Interval
    Delai de rafraichissement en secondes (defaut : 30).

.PARAMETER Once
    Affiche l'etat une seule fois puis rend la main.

.EXAMPLE
    .\scripts\watch_train.ps1
    .\scripts\watch_train.ps1 -Name ppe_yolo26s_960 -Rows 20
    .\scripts\watch_train.ps1 -Once
#>
[CmdletBinding()]
param(
    [string]$Name = "",
    [int]$Rows = 12,
    [int]$Interval = 30,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$RunsDir = Join-Path $ProjectRoot "artifacts\runs"
if (-not (Test-Path $RunsDir)) {
    throw "Aucun repertoire de runs : $RunsDir. Lancez d'abord un entrainement."
}

# Sans nom explicite, on suit le run dont results.csv a bouge le plus recemment.
if ([string]::IsNullOrWhiteSpace($Name)) {
    $latest = Get-ChildItem $RunsDir -Directory |
        ForEach-Object { Join-Path $_.FullName "results.csv" } |
        Where-Object { Test-Path $_ } |
        Get-Item | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) { throw "Aucun results.csv trouve dans $RunsDir." }
    $Csv = $latest.FullName
    $Name = Split-Path (Split-Path $Csv -Parent) -Leaf
} else {
    $Csv = Join-Path $RunsDir "$Name\results.csv"
    if (-not (Test-Path $Csv)) {
        throw "Fichier introuvable : $Csv`nRuns disponibles : $((Get-ChildItem $RunsDir -Directory).Name -join ', ')"
    }
}

function Show-Progress {
    # Copie prealable : Ultralytics peut ecrire pendant la lecture.
    $tmp = [System.IO.Path]::GetTempFileName()
    try {
        Copy-Item $Csv $tmp -Force
        $data = Import-Csv $tmp
    } catch {
        Write-Host "Lecture impossible (fichier en cours d'ecriture), nouvelle tentative..." -ForegroundColor DarkGray
        return
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }

    if (-not $data) { Write-Host "Aucune epoque terminee pour l'instant." -ForegroundColor Yellow; return }

    # @(...) est indispensable : sans lui, un Where-Object qui ne renvoie qu'un
    # seul nom produit une CHAINE, et [0] en extrait le premier caractere.
    $names = $data[0].PSObject.Properties.Name
    $epochCol = @($names | Where-Object { $_ -match 'epoch' })[0]
    $mapCol   = @($names | Where-Object { $_ -match 'mAP50-95' })[0]
    $map50Col = @($names | Where-Object { $_ -match 'mAP50\(' })[0]
    $pCol     = @($names | Where-Object { $_ -match 'precision' })[0]
    $rCol     = @($names | Where-Object { $_ -match 'recall' })[0]
    $timeCol  = @($names | Where-Object { $_ -match '^\s*time\s*$' })[0]

    if (-not $epochCol -or -not $mapCol) {
        Write-Host "Colonnes attendues absentes de $Csv." -ForegroundColor Red
        return
    }

    $done = $data.Count
    $best = $data | Sort-Object { [double]$_.$mapCol } -Descending | Select-Object -First 1
    $bestEpoch = [int]$best.$epochCol
    $elapsed = [double]$data[-1].$timeCol
    $perEpoch = $elapsed / $done

    # Le total d'epoques vient de args.yaml : results.csv ne le contient pas.
    $argsFile = Join-Path (Split-Path $Csv -Parent) "args.yaml"
    $total = 0
    if (Test-Path $argsFile) {
        $line = Get-Content $argsFile | Select-String -Pattern '^epochs:\s*(\d+)'
        if ($line) { $total = [int]$line.Matches[0].Groups[1].Value }
    }

    Clear-Host
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    $pct = if ($total -gt 0) { " ({0:N0} %)" -f (100 * $done / $total) } else { "" }
    Write-Host ("Epoque {0}/{1}{2}   |   {3:N0} s/epoque   |   ecoule {4}" -f `
        $done, $(if ($total) { $total } else { "?" }), $pct, $perEpoch, `
        ([TimeSpan]::FromSeconds($elapsed).ToString('hh\:mm\:ss')))

    if ($total -gt $done) {
        $remaining = [TimeSpan]::FromSeconds($perEpoch * ($total - $done))
        Write-Host ("Fin estimee dans {0} (vers {1:HH:mm})   |   early stopping possible avant" -f `
            $remaining.ToString('hh\:mm\:ss'), (Get-Date).Add($remaining)) -ForegroundColor DarkGray
    }
    Write-Host ""

    $data | Select-Object -Last $Rows | ForEach-Object {
        $isBest = ([int]$_.$epochCol -eq $bestEpoch)
        $line = "  {0,4}   mAP50 {1:N4}   mAP50-95 {2:N4}   P {3:N4}   R {4:N4}" -f `
            [int]$_.$epochCol, [double]$_.$map50Col, [double]$_.$mapCol, `
            [double]$_.$pCol, [double]$_.$rCol
        if ($isBest) { Write-Host "$line   <= meilleure" -ForegroundColor Green }
        else { Write-Host $line }
    }

    Write-Host ""
    Write-Host ("Meilleure epoque : {0}   mAP50-95 = {1:N4}   mAP50 = {2:N4}" -f `
        $bestEpoch, [double]$best.$mapCol, [double]$best.$map50Col) -ForegroundColor Green
    $sinceBest = $done - $bestEpoch
    if ($sinceBest -gt 0) {
        Write-Host ("Aucun progres depuis {0} epoque(s)." -f $sinceBest) -ForegroundColor DarkGray
    }
}

if ($Once) {
    Show-Progress
    return
}

Write-Host "Suivi de l'entrainement — Ctrl+C pour quitter (l'entrainement continue)." -ForegroundColor Yellow
Start-Sleep -Seconds 1
while ($true) {
    Show-Progress
    Write-Host ""
    Write-Host "Rafraichissement toutes les $Interval s — Ctrl+C pour quitter." -ForegroundColor DarkGray
    Start-Sleep -Seconds $Interval
}
