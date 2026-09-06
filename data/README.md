# data/

This directory holds local symlinks or mount points to the real data, which lives on GCS.
Nothing under here is committed except this file — see `.gitignore`.

Expected layout (mirror of the GCS bucket):

```
data/
  dem/                  # 30 m conditioned DEM tiles, same source as the HEC-RAS runs
  anchor/               # d100.tif — RP100 2030 India-wide depth grid (the calibration anchor)
  hecras/               # centerline vectors, per-basin, for reach snapping
  imd/                  # IMD gridded daily precipitation, for GEV fitting
  cmip6/                # NEX-GDDP-CMIP6 annual-max 1-day precip deltas
  sentinel1/             # observed-event inundation extents for validation (Kerala 2018, Assam 2022, Bihar)
  derived/               # hand.tif, reach_id.tif, catchment_id.tif, curves.parquet, calibration.parquet
```

Mount or symlink the GCS paths into these subdirectories locally, e.g.:

```
gcsfuse <bucket> data/
```
