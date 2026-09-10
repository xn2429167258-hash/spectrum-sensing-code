param(
    [string]$Python = "python",
    [string]$DateTag = "20260730",
    [int]$Seed = 42,
    [string]$RunSuffix = "pc_rician_seed42",
    [int]$BatchSize = 32,
    [string]$SourcePath = "data/source_gaussian_qpsk_rxshift_100mhz.pkl",
    [string]$TargetPath = "data/target_rician_qpsk_rxshift_100mhz.pkl",
    [string]$SaveRoot = "",
    [switch]$SkipEvaluation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($SaveRoot)) {
    $SaveRoot = Join-Path $ProjectRoot ("runs\\rician_" + $DateTag + "_" + $RunSuffix)
}
New-Item -ItemType Directory -Force -Path $SaveRoot | Out-Null

if (-not (Test-Path $SourcePath)) { throw "Source data file not found: $SourcePath" }
if (-not (Test-Path $TargetPath)) { throw "Rician target data file not found: $TargetPath" }

function Invoke-Step {
    param([string]$Name, [string[]]$Args)
    Write-Host ""
    Write-Host ("===== " + $Name + " =====")
    & $Python @Args
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Name" }
}

function Get-LatestRunDir {
    param([string]$Pattern)
    $dirs = Get-ChildItem -Path $SaveRoot -Directory | Where-Object { $_.Name -like $Pattern } | Sort-Object LastWriteTime -Descending
    if (-not $dirs) { throw "No run directory found for pattern: $Pattern under $SaveRoot" }
    return $dirs[0].FullName
}

function Get-Phase2Checkpoint {
    param([string]$Pattern)
    $dir = Get-LatestRunDir $Pattern
    $ckpt = Join-Path $dir "checkpoints\\phase2_finetuned.pth"
    if (-not (Test-Path $ckpt)) { throw "Checkpoint not found: $ckpt" }
    return $ckpt
}

function Get-FinalCheckpoint {
    param([string]$Pattern)
    $dir = Get-LatestRunDir $Pattern
    $ckpt = Join-Path $dir "checkpoints\\final_occupancy_semantic_adapt_model.pth"
    if (-not (Test-Path $ckpt)) { throw "Checkpoint not found: $ckpt" }
    return $ckpt
}

$CommonSourceArgs = @(
    "-m", "osada.train", "--mode", "basic",
    "--source-family", "gaussian", "--source-path", $SourcePath,
    "--save-root", $SaveRoot, "--seed", [string]$Seed, "--batch-size", [string]$BatchSize,
    "--fft-norm-mode", "log_power", "--encoder-norm", "bn", "--encoder-dilations", "1,1,1", "--encoder-use-se",
    "--source-subband-reweight-mode", "soft", "--source-aux-weight", "0.3", "--source-pos-weight", "2.5",
    "--early-stop-patience", "5", "--early-stop-min-delta", "0.0005", "--early-stop-metric", "combo"
)

$CommonDaArgs = @(
    "-m", "osada.train", "--mode", "da",
    "--source-family", "gaussian", "--source-path", $SourcePath,
    "--target-family", "rician", "--target-domain", "hard", "--target-path", $TargetPath,
    "--save-root", $SaveRoot, "--seed", [string]$Seed, "--batch-size", [string]$BatchSize,
    "--fft-norm-mode", "log_power", "--encoder-norm", "bn", "--encoder-dilations", "1,1,1", "--encoder-use-se",
    "--source-subband-reweight-mode", "soft", "--da-subband-reweight-mode", "soft",
    "--source-aux-weight", "0.3", "--source-pos-weight", "2.5",
    "--early-stop-patience", "5", "--early-stop-min-delta", "0.0005", "--early-stop-metric", "combo",
    "--phase3-init-domain-weight", "0.005", "--phase3-max-domain-weight", "0.05", "--source-anchor-weight", "0.1"
)

Invoke-Step "01 source-only" ($CommonSourceArgs + @("--model-tag", ("source_only_" + $DateTag)))
$SourceCkpt = Get-Phase2Checkpoint ("SourceOnly_source_only_" + $DateTag + "*")

Invoke-Step "02 Full OSADA main" ($CommonDaArgs + @("--model-tag", ("main_full_osada_" + $DateTag), "--base-checkpoint", $SourceCkpt, "--da-pseudo-weight", "0.02", "--da-pseudo-gate-mode", "snr_quantile", "--prototype-weight", "0.05"))
$FullCkpt = Get-FinalCheckpoint ("Adapt_Mode_main_full_osada_" + $DateTag + "*")

Invoke-Step "03 DA-Occ ablation" ($CommonDaArgs + @("--model-tag", ("ablation_02_occupancy_da_" + $DateTag), "--base-checkpoint", $SourceCkpt, "--da-pseudo-weight", "0.0", "--da-pseudo-gate-mode", "fixed", "--prototype-weight", "0.0"))
Invoke-Step "04 Occ+Pseudo ablation" ($CommonDaArgs + @("--model-tag", ("ablation_03_occupancy_pseudo_" + $DateTag), "--base-checkpoint", $SourceCkpt, "--da-pseudo-weight", "0.02", "--da-pseudo-gate-mode", "snr_quantile", "--prototype-weight", "0.0"))

$OccCkpt = Get-FinalCheckpoint ("Adapt_Mode_ablation_02_occupancy_da_" + $DateTag + "*")
$OccPseudoCkpt = Get-FinalCheckpoint ("Adapt_Mode_ablation_03_occupancy_pseudo_" + $DateTag + "*")

if (-not $SkipEvaluation) {
    Invoke-Step "05 print main evaluation" @("evaluate_main.py", "--eval-domains", "rician", "--source-checkpoint", $SourceCkpt, "--osada-checkpoint", $FullCkpt, "--rician-path", $TargetPath, "--batch-size", [string]$BatchSize, "--no-print-per-snr")
    Invoke-Step "06 print ablation evaluation" @("evaluate_ablation.py", "--target-path", $TargetPath, "--domain-name", "Rician", "--da-occ-checkpoint", $OccCkpt, "--occ-pseudo-checkpoint", $OccPseudoCkpt, "--full-osada-checkpoint", $FullCkpt, "--batch-size", [string]$BatchSize, "--no-print-per-snr")
}

Write-Host ""
Write-Host "Done. Run folder: $SaveRoot"
