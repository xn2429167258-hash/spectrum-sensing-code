# OSADA Wideband Spectrum Sensing

OSADA is a wideband multi-subband spectrum sensing method for cross-channel domain adaptation. This repository provides the cleaned code, prepared datasets, and final checkpoints needed to train and test our method on Gaussian-to-Rayleigh and Gaussian-to-Rician settings.

## Repository structure

```text
.
|-- data/                 # prepared Gaussian, Rayleigh, and Rician datasets
|-- weights/              # final checkpoints for direct testing
|-- osada/                # model, losses, training, data loading, and metrics
|-- scripts/              # Windows PowerShell and Linux/Git Bash scripts
|-- evaluate_main.py      # print main Rayleigh/Rician results
|-- evaluate_ablation.py  # print ablation results
|-- check_pc_setup.py     # quick PC setup check
|-- requirements.txt
`-- docs/                 # validation notes and file manifest
```

## Quick PC check

```powershell
python check_pc_setup.py
```

If PowerShell blocks local scripts, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Train first, then test

Windows PowerShell:

```powershell
.\scripts\train_rayleigh.ps1
.\scripts\train_rician.ps1
```

Linux or Git Bash:

```bash
bash scripts/train_rayleigh.sh
bash scripts/train_rician.sh
```

The training scripts run source training first, then the main OSADA stage, followed by the ablation stages. They print the main evaluation before the ablation evaluation. Training outputs are written under `runs/`.

## Convenient direct testing

Use the included final checkpoints when you only want to test without retraining.

```powershell
python evaluate_main.py --no-print-per-snr
python evaluate_ablation.py --no-print-per-snr
```

Run the main evaluation before the ablation evaluation when reporting results.

## Included files

The package includes prepared datasets, final checkpoints, core OSADA source code, Windows/Linux training scripts, print-only evaluation scripts, a PC setup checker, and file/hash manifests.
