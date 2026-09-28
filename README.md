# ImageGrains as a PebbleMapper detection model

This plug-in lets [PebbleMapper](https://github.com/soloyant/pebblemapper) detect clasts
with [ImageGrains](https://github.com/dmair1989/imagegrains) (Mair et al.), a Cellpose
model fine-tuned on images of fluvial sediment. The adapter runs ImageGrains in its own
conda environment, receives the grain outlines as a label image and hands them to
PebbleMapper, which measures every clast with its own measurement step. Sizes are
therefore defined exactly as for PebbleMapper's built-in Mask R-CNN model. Nothing in
PebbleMapper is modified.

The repository holds the adapter (built on PebbleMapper's `InstanceSubprocessBackend`),
the script that runs inside the ImageGrains environment and the environment files. The
models are not included; a script downloads them from the authors' Zenodo record.

## Requirements

- A working PebbleMapper installation.
- Conda (Miniconda or Anaconda).
- For ImageGrains 1.x (the default), a CPU is enough.
- For ImageGrains 2.0, a GPU with several GB of memory and a CUDA build of PyTorch.

## Which ImageGrains

| | ImageGrains 1.x (default) | ImageGrains 2.0 |
|---|---|---|
| Model | Cellpose 2, a CNN | Cellpose-SAM, a transformer |
| Model file | 26 MB | 1.2 GB |
| Hardware | a CPU is enough | a GPU with several GB of memory; slow on a CPU |
| Environment | `environment.yml` (`pm-imagegrains1`) | `environment-2.0.yml` (`pm-imagegrains`), plus a CUDA build of PyTorch |

The 1.x models and the 2.0 model are not interchangeable; the adapter picks the right
file for the Cellpose version it finds. The environment variables `PM_IG_ENV` and
`PM_IG_WEIGHTS` switch between the two installations.

## Installation

The steps below install ImageGrains 1.x.

1. Create the conda environment from this folder:

   ```bat
   conda env create -f environment.yml
   ```

   This creates `pm-imagegrains1` with `imagegrains==1.2.1`. For ImageGrains 2.0, use
   `environment-2.0.yml` instead, which creates `pm-imagegrains`; on a machine with a
   GPU, install a CUDA build of PyTorch in it first.

2. Download the models into `models/`:

   ```bat
   conda run -n pm-imagegrains1 python scripts/download_models.py
   ```

   The script runs ImageGrains' own downloader, which fetches the models from Zenodo into
   `~/imagegrains`, then copies the model files into `models/`. Set `PM_IG_WEIGHTS` to
   store them elsewhere.

3. Declare the backend in PebbleMapper's `user_detectors.json`. Copy
   `user_detectors.example.json` and set `path` to the folder of this repository:

   ```json
   [
    {"module": "pm_imagegrains_backend.backend", "factory": "make_backend",
     "path": "C:/path/to/pebblemapper-backend-imagegrains"}
   ]
   ```

4. Restart PebbleMapper.

## Usage in PebbleMapper

After the restart, **ImageGrains (Mair et al.)** appears in the **Detection model**
selector in the left panel, next to Mask R-CNN. Select it and run detection as usual. The
result is the same per-clast table as with Mask R-CNN.

Points to note:

- **Framed quadrat photographs.** ImageGrains segments every grain-like object, and the
  coloured bars of a quadrat frame count as such. A photograph rectified by
  PebbleMapper's Orthorectify carries the frame's thickness in its sidecar, and the frame
  band is left out automatically. For any other framed photograph, draw an *ROI per
  image* (Detect tab).
- **No confidence score.** Cellpose reports none, so every clast is scored 1.0.
  PebbleMapper's *Filter by confidence* is recorded in the run manifest but not applied.
- **Model options.** Options for `run.py` are passed through the `PM_IG_OPTIONS`
  environment variable as JSON: `model` (a model file name), `diameter`, `min_size`,
  `rescale`, `channels` and `max_side` (the long side the image is resampled to before
  segmentation; labels come back at full size).

PebbleMapper writes the model's name, version and licence into every run's
`.manifest.json`.

## Licences

This repository contains only the adapter. The model code is installed by pip from its
published releases, and the models are downloaded by `scripts/download_models.py`.

| Component | Copyright | Licence |
|---|---|---|
| This adapter | © 2026 Antoine Soloy | MIT (`LICENSE`) |
| imagegrains 1.2.1 / 2.0.2 (code) | © 2023 David Mair | BSD-3-Clause |
| cellpose 2.3.2 / 4.2.1 (code) | © 2020 Howard Hughes Medical Institute | BSD-3-Clause |
| ImageGrains models (`IG1_old_set`, `IG2_*`, `IG2_full_set_cp_SAM`) | the ImageGrains authors | CC BY 4.0, Zenodo [10.5281/zenodo.15309323](https://doi.org/10.5281/zenodo.15309323) |

If you pass the models on, keep the attribution to their authors and the Zenodo link, as
CC BY 4.0 requires. The adapter is an independent project and is not affiliated with or
endorsed by the authors of ImageGrains or Cellpose.

## Citation

Cite the paper that matches the ImageGrains generation you use.

**ImageGrains 1.x / `IG1` models (Cellpose 2, the default):**

- Mair, D., Witz, G., Do Prado, A. H., Garefalakis, P., & Schlunegger, F. (2023).
  Automated detecting, segmenting and measuring of grains in images of fluvial
  sediments: The potential for large and precise data from specialist deep learning
  models and transfer learning. *Earth Surface Processes and Landforms*.
  https://doi.org/10.1002/esp.5755
- Stringer, C. A., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular
  segmentation. *Nature Methods*, 18, 100–106. https://doi.org/10.1038/s41592-020-01018-x

**ImageGrains 2.0 / `IG2_full_set_cp_SAM` (Cellpose-SAM):**

- Mair, D., et al. (2026). ImageGrains 2.0: Improved precision and generalization for
  grain segmentation. *Earth Surface Dynamics*, 14, 527–551.
  https://doi.org/10.5194/esurf-14-527-2026
- Pachitariu, M., Rariden, M., & Stringer, C. (2025). Cellpose-SAM: superhuman
  generalization for cellular segmentation. *bioRxiv*.
  https://doi.org/10.1101/2025.04.28.651001

## The same photograph through every model

The figures below show the same rectified quadrat photograph (`example_03_Etretat`,
IMG_0955: 0.84 m frame, 0.567 mm/px) run through every model PebbleMapper can run, with
the frame band left out and each clast measured by PebbleMapper's own step.

| Model | Detections | Hand-outlined clasts found | Detections that match one | Length RMSE | D50 | D84 | Time per photograph |
|---|---|---|---|---|---|---|---|
| Hand outlines | 1,362 | | | | 17.7 mm | 26.5 mm | |
| Mask R-CNN (PebbleMapper, built in) | 324 | 320 (24 %) | 99 % | 1.6 mm | 20.2 mm | 34.2 mm | 41 s (GPU) |
| Segment Every Grain | 1,822 | 1,350 (99 %) | 74 % | 1.0 mm | 17.7 mm | 26.3 mm | 259 s (GPU) |
| ImageGrains | 2,034 | 1,316 (97 %) | 65 % | 1.7 mm | 17.2 mm | 26.0 mm | 51 s (CPU) |
| PebbleCountsAuto | 605 | 491 (36 %) | 81 % | 4.1 mm | 21.0 mm | 35.3 mm | 17 s (CPU) |

Detections are paired with the 1,362 hand-outlined clasts by position and size, as PebbleMapper's
Validate tab does. The hand outlines started from Segment Every Grain's detections, which favours
that model here. Times are for one photograph once the model is loaded (loading adds 20 to 70 s
once per run), on a 2018 laptop (Intel Core i7-8850H, NVIDIA Quadro P600 with 4 GB).

<p align="center">
  <img src="docs/figures/same-photo-imagegrains.jpg" alt="The example quadrat through this model" width="70%"/>
</p>
<p align="center"><em>This model's detections on the example quadrat: each clast filled and outlined, its long and short axes drawn.</em></p>

<p align="center">
  <img src="docs/figures/same-photo-all-models.jpg" alt="The example quadrat through the four models" width="100%"/>
</p>
<p align="center"><em>The four models side by side on the same photograph.</em></p>

<p align="center">
  <img src="docs/figures/same-photo-cdf.png" alt="Cumulative size distributions of the four models" width="70%"/>
</p>
<p align="center"><em>Cumulative distributions of clast length, D50 (circle) and D84 (square) marked; the grey band is the 8-pixel detection limit of this photograph.</em></p>
