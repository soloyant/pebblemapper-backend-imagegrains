"""Fetch the ImageGrains models into models/ (CC BY 4.0, Zenodo 10.5281/zenodo.15309323).

    conda run -n pm-imagegrains1 python scripts/download_models.py

ImageGrains downloads its models and demo data into ~/imagegrains; the model
files are then copied here (set PM_IG_WEIGHTS to put them elsewhere). The
Cellpose-2 models are 26 MB each; the Cellpose-SAM one is 1.2 GB and only
loads in the 2.0 environment.
"""
# Copyright (c) 2026 Antoine Soloy
# SPDX-License-Identifier: MIT
import os
import shutil
import subprocess
import sys
from pathlib import Path

home = Path.home() / "imagegrains" / "models"
dest = Path(os.environ.get("PM_IG_WEIGHTS") or Path(__file__).resolve().parents[1] / "models")
subprocess.run([sys.executable, "-m", "imagegrains", "--download_data", "True"], check=True)
dest.mkdir(parents=True, exist_ok=True)
for p in sorted(home.iterdir()):
    if p.is_file() and not (dest / p.name).exists():
        shutil.copy2(p, dest / p.name)
        print(f"{p.name} -> {dest}")
print("models:", sorted(q.name for q in dest.iterdir() if q.is_file()))
