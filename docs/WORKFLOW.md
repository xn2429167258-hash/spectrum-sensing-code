# Workflow

1. Check the PC setup:

```powershell
python check_pc_setup.py
```

2. Train first:

```powershell
.\scripts	rain_rayleigh.ps1
.\scripts	rain_rician.ps1
```

Linux/Git Bash alternatives:

```bash
bash scripts/train_rayleigh.sh
bash scripts/train_rician.sh
```

3. Test with included final weights when retraining is not needed:

```powershell
python evaluate_main.py --no-print-per-snr
python evaluate_ablation.py --no-print-per-snr
```

Public evaluation scripts print to terminal only and do not save CSV/MD outputs.
