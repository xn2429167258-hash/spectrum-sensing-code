# Validation

Validation date: 2026-09-10.

Checked:

```text
python check_pc_setup.py
python -m py_compile check_pc_setup.py evaluate_main.py evaluate_ablation.py osada/*.py
PowerShell AST parse: scripts/*.ps1
python evaluate_main.py --batch-size 512 --no-print-per-snr
python evaluate_ablation.py --batch-size 512 --no-print-per-snr
```

Main output:

```text
Source-only    Rayleigh  F1=0.946189  Pd=0.897941  Subset=0.794706
OSADA-Rayleigh Rayleigh  F1=0.976978  Pd=0.956059  Subset=0.908088
OSADA-Rician   Rician    F1=0.987723  Pd=0.976955  Subset=0.948676
```

Ablation output:

```text
DA-Occ       F1=0.973071  Subset=0.894118
Occ+Pseudo   F1=0.975888  Subset=0.904412
Full OSADA   F1=0.976978  Subset=0.908088
```

No full training was run during packaging validation.
