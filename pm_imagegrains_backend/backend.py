"""ImageGrains (Mair et al.) as a PebbleMapper detection backend.

Declared through ``user_detectors.json`` (``{"module":
"pm_imagegrains_backend.backend", "factory": "make_backend", "path": "<this
repository>"}``); no PebbleMapper source is touched. The driver is
PebbleMapper's own :class:`detectors.instance_backend.InstanceSubprocessBackend`:
this file only says what the model is, where its weights are, and which
options its ``run.py`` takes.

``run.py`` runs in the ImageGrains conda env, segments the image with a
Cellpose model fine-tuned on sediment images and leaves a label image; every
instance is then measured by PebbleMapper's own measurement step, so the table
means exactly what Mask R-CNN's does.

Two generations exist. ImageGrains 1.x drives Cellpose 2 (a CNN, 26 MB
models, seconds on a CPU); 2.0 drives Cellpose-SAM (a transformer, a 1.2 GB
model, a GPU with several GB is needed in practice). ``PM_IG_ENV`` and
``PM_IG_WEIGHTS`` pick which installation this backend uses; the default is
the 1.x environment.
"""
# Copyright (c) 2026 Antoine Soloy
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
from pathlib import Path

from detectors.base import BackendInfo
from detectors.instance_backend import InstanceSubprocessBackend

PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent
ENV_NAME = os.environ.get("PM_IG_ENV", "pm-imagegrains1")
_SCRIPT = PKG_DIR / "run.py"
WEIGHTS_DIR = Path(os.environ.get("PM_IG_WEIGHTS") or (REPO_DIR / "models"))

# The citations the ImageGrains authors ask for (their README): the 2023 ESPL
# paper plus Cellpose 1.0 for the 1.x models; the 2026 ESurf paper plus
# Cellpose-SAM for the 2.0 model.
CITATION = ("Mair, D., Witz, G., Do Prado, A. H., Garefalakis, P., & "
            "Schlunegger, F. (2023). Automated detecting, segmenting and "
            "measuring of grains in images of fluvial sediments: The potential "
            "for large and precise data from specialist deep learning models "
            "and transfer learning. Earth Surface Processes and Landforms, "
            "1-18. https://doi.org/10.1002/esp.5755; and Stringer, C. A., & "
            "Pachitariu, M. (2021). Cellpose: a generalist algorithm for "
            "cellular segmentation. Nature Methods, 18, 100-106. With the 2.0 "
            "model cite instead Mair et al. (2026), Earth Surf. Dyn. 14, "
            "527-551, https://doi.org/10.5194/esurf-14-527-2026, and "
            "Pachitariu, Rariden & Stringer (2025), Cellpose-SAM, bioRxiv "
            "10.1101/2025.04.28.651001.")

INFO = BackendInfo(
    name="imagegrains",
    display_name="ImageGrains (Mair et al.)",
    framework="pytorch (cellpose)",
    license=("BSD-3-Clause (imagegrains, Copyright (c) 2023 David Mair); "
             "BSD-3-Clause (cellpose, Copyright 2020 Howard Hughes Medical "
             "Institute); the pretrained sediment models are distributed by "
             "the ImageGrains authors on Zenodo under CC BY 4.0 "
             "(https://doi.org/10.5281/zenodo.15309323). None of it is "
             "redistributed with this adapter (MIT)."),
    output_type="instance",
    env=ENV_NAME,
    in_process=False,
    weights=str(WEIGHTS_DIR),
    install_hint=("Create the conda env (environment.yml), run "
                  "scripts/download_models.py, and declare the backend in "
                  "user_detectors.json."),
    description=("A Cellpose model fine-tuned on sediment images; instances "
                 "are measured by PebbleMapper's shared measurement step. "
                 "Cellpose reports no per-instance confidence, so every clast "
                 "is scored 1.0. Cite: " + CITATION),
)


def make_backend():
    """Zero-argument factory named in user_detectors.json."""
    return ImageGrainsBackend()


class ImageGrainsBackend(InstanceSubprocessBackend):
    info = INFO
    env_name = ENV_NAME
    required_modules = ("torch", "cellpose", "imagegrains")
    script = _SCRIPT
    model_version = "imagegrains (cellpose 2.x or 4.x, see the manifest params)"
    log_prefix = "imagegrains"

    def is_available(self) -> bool:
        return (super().is_available()
                and WEIGHTS_DIR.is_dir()
                and any(p.is_file() for p in WEIGHTS_DIR.rglob("*")))

    def spec_params(self, mode, kwargs, resolution):
        log_fn = kwargs.get("log_fn") or (lambda s: None)
        if kwargs.get("min_confidence") is not None:
            log_fn("[imagegrains] note: min_confidence is a Mask R-CNN "
                   "threshold; Cellpose reports no per-instance confidence, "
                   "so it is recorded in the manifest but not applied.")
        opts = dict(kwargs.get("imagegrains_options")
                    or json.loads(os.environ.get("PM_IG_OPTIONS", "{}") or "{}"))
        return {"weights_dir": str(WEIGHTS_DIR), "ig": opts}
