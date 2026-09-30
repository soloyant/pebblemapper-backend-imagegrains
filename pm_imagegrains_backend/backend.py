"""ImageGrains (Mair et al.) as PebbleMapper detection backends.

Declared through ``user_detectors.json``, one entry per generation; no
PebbleMapper source is touched::

    {"module": "pm_imagegrains_backend.backend", "factory": "make_backend",
     "path": "<this repository>"}        ImageGrains 2.0, name "imagegrains"
    {"module": "pm_imagegrains_backend.backend", "factory": "make_backend_v1",
     "path": "<this repository>"}        ImageGrains 1.2, name "imagegrains1"

The driver is PebbleMapper's own
:class:`detectors.instance_backend.InstanceSubprocessBackend`: this file only
says what each model is, where its weights are, and which options its
``run.py`` takes. ``run.py`` runs in the generation's own conda env, segments
the image with a Cellpose model fine-tuned on sediment images and leaves a
label image; every instance is then measured by PebbleMapper's own measurement
step, so the table means exactly what Mask R-CNN's does.

ImageGrains 2.0 drives Cellpose-SAM (a transformer, the 1.2 GB
``IG2_full_set_cp_SAM`` model, a GPU with at least 3 GB in practice).
ImageGrains 1.2 drives Cellpose 2 (a CNN, 26 MB models that run on a CPU).
Their environments cannot be merged: Cellpose 2 and Cellpose 4 do not install
together, and neither loads the other's models.
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
_SCRIPT = PKG_DIR / "run.py"
WEIGHTS_DIR = Path(os.environ.get("PM_IG_WEIGHTS") or (REPO_DIR / "models"))

LICENSE = ("BSD-3-Clause (imagegrains, Copyright (c) 2023 David Mair); "
           "BSD-3-Clause (cellpose, Copyright 2020 Howard Hughes Medical "
           "Institute); the pretrained sediment models are distributed by the "
           "ImageGrains authors on Zenodo under CC BY 4.0 "
           "(https://doi.org/10.5281/zenodo.15309323). None of it is "
           "redistributed with this adapter (MIT).")

# The citations the ImageGrains authors ask for (their README).
CITATION_V2 = ("Mair, D., Witz, G., Do Prado, A., Garefalakis, P., Wild, A., "
               "Ville, F., Schuster, B., Horn, M., Österle, J., Fabbri, S. C., "
               "Litty, C., Achleitner, S., Leistner, S., Hiller, C., & "
               "Schlunegger, F. (2026). ImageGrains 2.0: Improved precision and "
               "generalization for grain segmentation. Earth Surface Dynamics, "
               "14, 527-551. https://doi.org/10.5194/esurf-14-527-2026; and "
               "Pachitariu, M., Rariden, M., & Stringer, C. (2025). Cellpose-SAM: "
               "superhuman generalization for cellular segmentation. bioRxiv. "
               "https://doi.org/10.1101/2025.04.28.651001")
CITATION_V1 = ("Mair, D., Witz, G., Do Prado, A. H., Garefalakis, P., & "
               "Schlunegger, F. (2023). Automated detecting, segmenting and "
               "measuring of grains in images of fluvial sediments: The "
               "potential for large and precise data from specialist deep "
               "learning models and transfer learning. Earth Surface Processes "
               "and Landforms. https://doi.org/10.1002/esp.5755; and Stringer, "
               "C. A., & Pachitariu, M. (2021). Cellpose: a generalist algorithm "
               "for cellular segmentation. Nature Methods, 18, 100-106.")


def _info(name, display, env, model, citation, what):
    return BackendInfo(
        name=name,
        display_name=display,
        framework="pytorch (cellpose)",
        license=LICENSE,
        output_type="instance",
        env=env,
        in_process=False,
        weights=str(WEIGHTS_DIR / model),
        install_hint=("Create the conda env (see README.md), run "
                      "scripts/download_models.py, and declare the backend in "
                      "user_detectors.json."),
        description=(what + "; instances are measured by PebbleMapper's shared "
                     "measurement step. Cellpose reports no per-instance "
                     "confidence, so every clast is scored 1.0. Cite: " + citation),
    )


class _ImageGrainsBase(InstanceSubprocessBackend):
    required_modules = ("torch", "cellpose", "imagegrains")
    script = _SCRIPT
    default_model = ""

    def is_available(self) -> bool:
        return (super().is_available()
                and any(WEIGHTS_DIR.glob(f"{self.default_model}*")))

    def spec_params(self, mode, kwargs, resolution):
        log_fn = kwargs.get("log_fn") or (lambda s: None)
        if kwargs.get("min_confidence") is not None:
            log_fn(f"[{self.log_prefix}] note: min_confidence is a Mask R-CNN "
                   "threshold; Cellpose reports no per-instance confidence, "
                   "so it is recorded in the manifest but not applied.")
        opts = dict(kwargs.get("imagegrains_options")
                    or json.loads(os.environ.get("PM_IG_OPTIONS", "{}") or "{}"))
        opts.setdefault("model", self.default_model)
        return {"weights_dir": str(WEIGHTS_DIR), "ig": opts}


class ImageGrains2Backend(_ImageGrainsBase):
    default_model = "IG2_full_set_cp_SAM"
    env_name = os.environ.get("PM_IG2_ENV", "pm-imagegrains")
    info = _info("imagegrains", "ImageGrains 2.0 (Mair et al., 2026)", env_name,
                 default_model, CITATION_V2,
                 "Cellpose-SAM fine-tuned on the IG2 sediment dataset "
                 "(ImageGrains 2.0.2, model IG2_full_set_cp_SAM)")
    model_version = "imagegrains 2.0.2 + cellpose 4, IG2_full_set_cp_SAM"
    log_prefix = "imagegrains"


class ImageGrains1Backend(_ImageGrainsBase):
    default_model = "IG2_full_set.200525"
    env_name = os.environ.get("PM_IG1_ENV", "pm-imagegrains1")
    info = _info("imagegrains1", "ImageGrains 1.2 (Mair et al., 2023)", env_name,
                 default_model, CITATION_V1,
                 "A Cellpose 2 CNN trained on the IG2 sediment dataset "
                 "(ImageGrains 1.2.1, model IG2_full_set.200525)")
    model_version = "imagegrains 1.2.1 + cellpose 2, IG2_full_set.200525"
    log_prefix = "imagegrains1"


def make_backend():
    """ImageGrains 2.0 (Cellpose-SAM): the factory named in user_detectors.json."""
    return ImageGrains2Backend()


def make_backend_v1():
    """ImageGrains 1.2 (Cellpose 2): declare it as a second entry to run both."""
    return ImageGrains1Backend()
