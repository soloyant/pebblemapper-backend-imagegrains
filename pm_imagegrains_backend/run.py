"""ImageGrains backend: subprocess entry point (runs in the generation's own conda env).

Reads the PebbleMapper job spec (``--spec <json>``), segments each job image
with an ImageGrains model (Mair et al.; a Cellpose model fine-tuned on
sediment images) and writes, for each job, an *instance file* next to the CSV
the core expects::

    <out_csv>.instances.npz   labels  int32 HxW  (0 = background, k = grain k)
                              scores  float32 N  (1.0: Cellpose returns no
                                                  per-instance confidence)
                              shape   (H, W)

It deliberately does NOT write the canonical CSV: measurement happens in the
core (detectors.measure.measure_mask), so the numbers are defined identically
to Mask R-CNN's. This file must never import the PebbleMapper core.

Spec params honoured: ``devicemode`` ("gpu"/"cpu"), ``weights_dir`` (the folder
holding the ImageGrains model files), ``resolution``/``metric_cropsize``
(ortho tiling is left to Cellpose's own tiling), and ``ig`` (a dict of
ImageGrains options: model, diameter, min_size, rescale, max_side).
"""
# Copyright (c) 2026 Antoine Soloy
# SPDX-License-Identifier: MIT
import argparse
import json
import os
import sys
import time

import numpy as np


def log(msg):
    print(f"[imagegrains] {msg}", flush=True)


def _load_rgb(path, mode):
    """HxWx3 uint8 in the stored pixel frame (no EXIF transpose, like the core).

    Ortho GeoTIFFs go through rasterio; photographs through Pillow, with
    pillow-heif registered when present so an iPhone HEIC opens too.
    """
    if mode == "ortho":
        import rasterio
        with rasterio.open(path) as ds:
            n = min(3, ds.count)
            arr = np.moveaxis(ds.read(list(range(1, n + 1))), 0, -1)
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
        if arr.dtype != np.uint8:
            a = arr.astype(np.float64)
            lo, hi = np.nanpercentile(a, 0.5), np.nanpercentile(a, 99.5)
            arr = (np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
        return np.ascontiguousarray(arr)
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _pick_device(devicemode):
    import torch
    forced = os.environ.get("PM_IG_DEVICE", "").strip().lower()
    want_gpu = (forced == "gpu") or (forced != "cpu"
                                     and str(devicemode or "").lower() == "gpu")
    if want_gpu and torch.cuda.is_available():
        return True
    return False


def _find_model(weights_dir, wanted="", cp_major=4):
    """The model file to load: the one named in the options, else the
    ImageGrains default for this Cellpose generation, else the first model
    in the folder. A Cellpose-SAM model cannot be loaded by Cellpose 2, and
    the reverse, so the wrong generation is filtered out."""
    from pathlib import Path
    d = Path(weights_dir)
    if wanted:
        p = Path(wanted)
        if p.is_file():
            return p
        hit = sorted(d.glob(f"*{wanted}*"))
        if hit:
            return hit[0]
    files = [p for p in sorted(d.rglob("*"))
             if p.is_file() and p.suffix.lower() not in (".txt", ".md", ".json")]
    is_sam = [p for p in files if "cp_sam" in p.name.lower()]
    files = is_sam if cp_major >= 4 else [p for p in files if p not in is_sam]
    if not files:
        raise RuntimeError(
            f"no ImageGrains model for cellpose {cp_major}.x under {d}")
    for key in ("default", "full_set", "imagegrains"):
        hit = [p for p in files if key.lower() in p.name.lower()]
        if hit:
            return hit[0]
    return files[0]


def _downscale(image, max_side):
    """(image, factor): Cellpose-SAM is a transformer, and a 3000 px phone
    photograph on the CPU is minutes per image. ``max_side`` caps the long
    side; the labels are put back at full size by the caller."""
    if not max_side:
        return image, 1.0
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image, 1.0
    import cv2
    f = float(max_side) / float(long_side)
    small = cv2.resize(image, (max(1, int(round(w * f))), max(1, int(round(h * f)))),
                       interpolation=cv2.INTER_AREA)
    return small, f


def _upscale_labels(labels, shape):
    import cv2
    h, w = shape
    if labels.shape == (h, w):
        return labels
    return cv2.resize(labels.astype(np.int32), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(np.int32)


def main(argv=None):
    ap = argparse.ArgumentParser(description="ImageGrains backend for PebbleMapper.")
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    with open(args.spec, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    mode = str(spec.get("mode", "quadrat")).lower()
    params = spec.get("params", {})
    opts = params.get("ig", {}) or {}
    jobs = spec.get("jobs", [])
    weights_dir = params.get("weights_dir") or os.environ.get("PM_IG_WEIGHTS") or ""
    use_gpu = _pick_device(params.get("devicemode"))

    t_load = time.time()
    from cellpose import models, version as cp_version
    _cp_major = int(str(cp_version).split(".")[0] or 0)
    model_path = _find_model(weights_dir, str(opts.get("model", "")),
                             cp_major=_cp_major)
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))
    log(f"cellpose {cp_version} on {'gpu' if use_gpu else 'cpu'}; model "
        f"{model_path.name} ({time.time() - t_load:.1f}s)")

    diameter = opts.get("diameter")
    min_size = int(opts.get("min_size", 15))
    rescale = opts.get("rescale")
    max_side = int(opts.get("max_side", 2048))

    for ji, job in enumerate(jobs):
        path = job["path"]
        out_npz = job.get("instances_path") or (job["out_csv"] + ".instances.npz")
        t0 = time.time()
        log(f"job {ji + 1}/{len(jobs)}: {os.path.basename(path)}")
        image = _load_rgb(path, mode)
        h, w = image.shape[:2]
        small, factor = _downscale(image, max_side)
        if factor != 1.0:
            log(f"  image {w}x{h} px, segmented at {small.shape[1]}x{small.shape[0]} "
                f"(max_side={max_side}); labels are returned at full size")
        else:
            log(f"  image {w}x{h} px")
        eval_kw = dict(diameter=diameter, min_size=min_size, rescale=rescale)
        if _cp_major < 4:
            # Cellpose 2 (ImageGrains 1.x) needs the channel pair; Cellpose-SAM
            # takes the image as it is and rejects the argument.
            eval_kw["channels"] = opts.get("channels", [0, 0])
        masks, _flows, _styles = model.eval([small], **eval_kw)
        labels = np.asarray(masks[0], dtype=np.int32)
        labels = _upscale_labels(labels, (h, w))
        n = int(labels.max())
        # Cellpose gives no per-instance confidence; the core needs a Score
        # column, so every grain is reported at 1.0 and the manifest says so.
        scores = np.ones(n, dtype=np.float32)
        os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
        np.savez_compressed(out_npz, labels=labels, scores=scores,
                            shape=np.array([h, w], dtype=np.int64))
        log(f"  {n} grains -> {os.path.basename(out_npz)} ({time.time() - t0:.1f}s)")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
