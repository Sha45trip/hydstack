"""One-off, read-only extraction: pull a basin-sized window of DEM, the RP100
2030 anchor depth, and rainfall out of the India-wide GCS-backed catalogs in
data/*.parquet, into local GeoTIFFs under data/derived/<basin>/ for
src/hand.py and friends to consume.

Every raster is opened via GDAL's read-only /vsigs/ driver — this script never
writes to the source bucket, only to local disk.

Storage conventions on the source COGs, confirmed directly:
  - depth:    raw int32 value = depth_m * DEPTH_SCALE       (DEPTH_SCALE=1000)
  - rainfall: raw int32 value = rainfall, unscaled           (RAINFALL_SCALE=1)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window, from_bounds

CHUNK_ROWS = 2000  # row-block size for windowed reads -- keeps peak memory to a
                    # few hundred MB regardless of basin size, rather than
                    # materializing the whole (int32 + float64 + float32) array
                    # at once, which OOM'd on the full Godavari depth raster.

DEM_CATALOG = "data/Copernicus_GLO30DEM_IndiaCOG_dem.parquet"
DEPTH_CATALOG = "data/pluvial_flood_parquet_flood_depth_10.2.parquet"
RAINFALL_CATALOG = "data/pluvial_flood_parquet_rain_10.2.parquet"
GCS_KEY = "data/service-dev.json"

ANCHOR_DEPTH_ID = "india_flood_2030_ssp245_rp100"
ANCHOR_RAINFALL_ID = "rain_ssp245_rp100"
ANCHOR_YEAR = 2030

DEPTH_SCALE = 1000.0
RAINFALL_SCALE = 1.0

# Approximate — refine with a real basin boundary vector when available.
BASIN_BBOXES = {
    "godavari": (73.5, 16.0, 83.5, 23.0),
    "mahanadi": (80.5, 19.0, 86.5, 23.5),
}


def _href_to_vsigs(href):
    assert href.startswith("gs://"), f"expected a gs:// href, got {href!r}"
    return "/vsigs/" + href[len("gs://"):]


def _set_gcs_credentials(key_path=GCS_KEY):
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(Path(key_path).resolve()))


def _find_catalog_row(catalog_df, id_value):
    matches = catalog_df[catalog_df["id"] == id_value]
    if matches.empty:
        raise KeyError(f"id {id_value!r} not found; available: {list(catalog_df['id'])}")
    return matches.iloc[0]


def _write_raster(out_path, array, transform, crs, nodata):
    profile = {
        "driver": "GTiff", "dtype": str(array.dtype), "count": 1, "crs": crs, "transform": transform,
        "height": array.shape[0], "width": array.shape[1], "nodata": nodata,
        "compress": "deflate", "tiled": True,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array, 1)
    return out_path


def extract_dem(bbox, out_path, catalog_parquet=DEM_CATALOG, key_path=GCS_KEY):
    """Mosaic every DEM tile intersecting bbox into one local GeoTIFF."""
    _set_gcs_credentials(key_path)
    catalog_df = pd.read_parquet(catalog_parquet)
    xmin, ymin, xmax, ymax = bbox

    intersecting = catalog_df[catalog_df["bbox"].apply(
        lambda b: b["xmin"] < xmax and b["xmax"] > xmin and b["ymin"] < ymax and b["ymax"] > ymin
    )]
    if intersecting.empty:
        raise ValueError(f"no DEM tiles intersect bbox {bbox}")

    datasets = [rasterio.open(_href_to_vsigs(row["assets"]["data"]["href"])) for _, row in intersecting.iterrows()]
    crs, dtype, nodata = datasets[0].crs, datasets[0].dtypes[0], datasets[0].nodata
    try:
        mosaic, transform = merge(datasets, bounds=bbox)
    finally:
        for ds in datasets:
            ds.close()

    return _write_raster(out_path, mosaic[0].astype(dtype), transform, crs, nodata)


def _windowed_scale_convert(href, bbox, out_path, band, scale, chunk_rows=CHUNK_ROWS):
    """Read one band of a (possibly huge) remote raster in row-chunks, scale
    each chunk to float32, and stream it straight to a local GeoTIFF. Peak
    memory is a couple of chunks' worth, not the whole array — a whole-array
    read + float64 conversion OOM'd on the full Godavari depth raster."""
    with rasterio.open(_href_to_vsigs(href)) as src:
        full_window = from_bounds(*bbox, transform=src.transform).round_offsets().round_lengths()
        transform = src.window_transform(full_window)
        crs = src.crs
        nodata = src.nodata
        width, height = int(full_window.width), int(full_window.height)

        profile = {
            "driver": "GTiff", "dtype": "float32", "count": 1, "crs": crs, "transform": transform,
            "height": height, "width": width, "nodata": np.nan, "compress": "deflate", "tiled": True,
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            for row_start in range(0, height, chunk_rows):
                rows = min(chunk_rows, height - row_start)
                sub_window = Window(full_window.col_off, full_window.row_off + row_start, width, rows)
                raw = src.read(band, window=sub_window)

                chunk = raw.astype("float32")
                if scale != 1.0:
                    chunk /= np.float32(scale)
                if nodata is not None:
                    chunk[raw == nodata] = np.nan

                dst.write(chunk, 1, window=Window(0, row_start, width, rows))

    return out_path


def extract_depth(bbox, out_path, id_value=ANCHOR_DEPTH_ID, catalog_parquet=DEPTH_CATALOG,
                   depth_scale=DEPTH_SCALE, key_path=GCS_KEY):
    """Windowed, chunked read of one depth raster, converted from the on-disk
    int32 storage convention to float32 meters."""
    _set_gcs_credentials(key_path)
    catalog_df = pd.read_parquet(catalog_parquet)
    row = _find_catalog_row(catalog_df, id_value)
    href = row["assets"]["data"]["href"]
    return _windowed_scale_convert(href, bbox, out_path, band=1, scale=depth_scale)


def extract_rainfall_band(bbox, out_path, id_value=ANCHOR_RAINFALL_ID, year=ANCHOR_YEAR,
                           catalog_parquet=RAINFALL_CATALOG, rainfall_scale=RAINFALL_SCALE, key_path=GCS_KEY):
    """Windowed, chunked read of one (RP, SSP) rainfall raster's band for one
    year, converted from the on-disk int32 storage convention to float32."""
    _set_gcs_credentials(key_path)
    catalog_df = pd.read_parquet(catalog_parquet)
    row = _find_catalog_row(catalog_df, id_value)
    asset = row["assets"][str(year)]
    band = asset["band_index"] + 1  # rasterio bands are 1-indexed
    return _windowed_scale_convert(asset["href"], bbox, out_path, band=band, scale=rainfall_scale)


def extract_basin(bbox, out_dir, rainfall_ids=(ANCHOR_RAINFALL_ID,), year=ANCHOR_YEAR, overwrite=False):
    """Pull DEM + anchor depth + one or more rainfall (RP, SSP) rasters for one
    bbox. Returns a dict of output paths.

    Skips any output that already exists on disk (unless overwrite=True) —
    these pulls are large and the connection to GCS can drop mid-run, so a
    retry shouldn't re-download what already succeeded.
    """
    out_dir = Path(out_dir)

    def _extract_if_missing(fn, out_path, **kwargs):
        if out_path.exists() and not overwrite:
            print(f"skipping {out_path} (already exists)")
            return out_path
        return fn(bbox, out_path, **kwargs)

    paths = {
        "dem": _extract_if_missing(extract_dem, out_dir / "dem.tif"),
        "d100": _extract_if_missing(extract_depth, out_dir / "d100.tif"),
    }
    for rainfall_id in rainfall_ids:
        paths[f"rainfall_{rainfall_id}_{year}"] = _extract_if_missing(
            extract_rainfall_band, out_dir / f"{rainfall_id}_{year}.tif", id_value=rainfall_id, year=year,
        )
    return paths


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--basin", choices=sorted(BASIN_BBOXES), help="a preset basin bbox")
    group.add_argument("--bbox", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rainfall-ids", nargs="+", default=[ANCHOR_RAINFALL_ID])
    parser.add_argument("--year", type=int, default=ANCHOR_YEAR)
    parser.add_argument("--overwrite", action="store_true", help="re-download outputs that already exist")
    args = parser.parse_args()

    bbox = BASIN_BBOXES[args.basin] if args.basin else tuple(args.bbox)
    paths = extract_basin(bbox, args.out_dir, rainfall_ids=args.rainfall_ids, year=args.year, overwrite=args.overwrite)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
