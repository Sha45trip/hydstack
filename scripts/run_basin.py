"""Run the full pipeline for one basin bbox, end to end, in one command:

  1. Extract DEM + anchor depth + anchor/scenario rainfall (read-only, GCS)
  2. HAND / reach / catchment delineation (scripts/extract_basin.py + src/hand.py)
  3. Align d100 and rainfall onto the HAND grid
  4. Depression delineation, filtered to real features (src/pluvial.py)
  5. Depression curves + alpha calibration (src/pluvial.py)
  6. Channel-ratio calibration for everything outside depressions (src/render.py)
  7. Render the scenario and combine both domains
  8. (optional) a separate, much coarser catchment layer for a legible
     boundary overlay, and a PNG (scripts/plot_depth.py)

This exists because steps 3-8 were, until now, ad-hoc glue run by hand for
the Godavari sub-basin — see that session's history for the reasoning behind
each default below. Nothing here is more "correct" than what's in src/; this
just wires the existing library functions together for a new basin.

Output layout under --out-dir:
  raw/            extracted DEM, anchor depth, rainfall (Step 1)
  intermediate/   HAND/reach/catchment delineations, unfiltered depressions
  outputs/        depth_<scenario>_<year>.tif + .png, and the fitted
                   calibration tables (channel_ratio_k, depression_alpha,
                   depression_curves) -- everything worth looking at without
                   re-running the pipeline lives here, nowhere else.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

from scripts.extract_basin import (
    ANCHOR_DEPTH_ID,
    ANCHOR_RAINFALL_ID,
    ANCHOR_YEAR,
    extract_depth,
    extract_dem,
    extract_rainfall_band,
)
from scripts.plot_depth import plot_depth_png
from src.curves import anchor_stage_by_reach
from src.hand import build_hand_stack, read_raster, write_raster
from src.pluvial import (
    calibrate_depression_alpha,
    delineate_depressions,
    depression_curves,
    depression_relative_elevation,
    render_pluvial_depth,
)
from src.render import (
    calibrate_channel_ratio,
    combine_channel_and_depression_depth,
    render_channel_ratio_depth,
)

DEFAULT_THRESHOLD = 5000  # flow-accumulation threshold for HAND/calibration reaches
DEFAULT_VIZ_THRESHOLD = 300000  # much coarser -- for a legible boundary overlay only
DEFAULT_MIN_DEPRESSION_CELLS = 100  # raw DEMs produce ~1M+ 1-2 cell noise pits


def _align_to(src_path, target_profile, resampling=Resampling.bilinear):
    """Reproject/resample one band onto target_profile's exact grid. d100 and
    the rainfall rasters are natively a different resolution than the HAND
    grid, so this always runs before anything touches them together."""
    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_transform, src_crs, src_nodata = src.transform, src.crs, src.nodata

    dst = np.full((target_profile["height"], target_profile["width"]), np.nan, dtype="float32")
    reproject(
        source=src_data, destination=dst,
        src_transform=src_transform, src_crs=src_crs, src_nodata=src_nodata,
        dst_transform=target_profile["transform"], dst_crs=target_profile["crs"], dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


def _skip_if_exists(path, fn, *args, **kwargs):
    if Path(path).exists():
        print(f"  skip {path} (already exists)")
        return path
    return fn(*args, **kwargs)


def run_basin(bbox, out_dir, threshold=DEFAULT_THRESHOLD, viz_threshold=DEFAULT_VIZ_THRESHOLD,
              min_depression_cells=DEFAULT_MIN_DEPRESSION_CELLS,
              depth_id=ANCHOR_DEPTH_ID, anchor_rainfall_id=ANCHOR_RAINFALL_ID, anchor_year=ANCHOR_YEAR,
              scenario_rainfall_id=None, scenario_year=None, skip_viz=False):
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    intermediate_dir = out_dir / "intermediate"
    outputs_dir = out_dir / "outputs"
    for d in (raw_dir, intermediate_dir, outputs_dir):
        d.mkdir(parents=True, exist_ok=True)

    scenario_rainfall_id = scenario_rainfall_id or anchor_rainfall_id
    scenario_year = scenario_year or anchor_year

    print("== 1/8: extract DEM, anchor depth, rainfall ==")
    dem_path = raw_dir / "dem.tif"
    d100_path = raw_dir / "d100.tif"
    anchor_rain_path = raw_dir / f"{anchor_rainfall_id}_{anchor_year}.tif"
    scenario_rain_path = raw_dir / f"{scenario_rainfall_id}_{scenario_year}.tif"

    _skip_if_exists(dem_path, extract_dem, bbox, dem_path)
    _skip_if_exists(d100_path, extract_depth, bbox, d100_path, id_value=depth_id)
    _skip_if_exists(anchor_rain_path, extract_rainfall_band, bbox, anchor_rain_path,
                     id_value=anchor_rainfall_id, year=anchor_year)
    _skip_if_exists(scenario_rain_path, extract_rainfall_band, bbox, scenario_rain_path,
                     id_value=scenario_rainfall_id, year=scenario_year)

    print("== 2/8: HAND / reach / catchment delineation ==")
    hand_out = intermediate_dir / "hand_out"
    if not (hand_out / "hand.tif").exists():
        build_hand_stack(dem_path, hand_out, threshold=threshold)
    else:
        print(f"  skip {hand_out} (already exists)")

    hand, hand_profile = read_raster(hand_out / "hand.tif")
    catchment_id, _ = read_raster(hand_out / "catchment_id.tif")
    cell_area = abs(hand_profile["transform"].a * hand_profile["transform"].e)

    print("== 3/8: align d100 and rainfall to the HAND grid ==")
    d100 = np.nan_to_num(_align_to(d100_path, hand_profile), nan=0.0)
    rain_p100 = _align_to(anchor_rain_path, hand_profile)
    rain_p_prime = _align_to(scenario_rain_path, hand_profile)

    print("== 4/8: depression delineation ==")
    depression_id_raw_path = intermediate_dir / "depression_id_raw.tif"
    if not depression_id_raw_path.exists():
        delineate_depressions(dem_path, depression_id_raw_path)
    else:
        print(f"  skip {depression_id_raw_path} (already exists)")
    depression_id_raw, _ = read_raster(depression_id_raw_path)

    ids, counts = np.unique(depression_id_raw[depression_id_raw > 0], return_counts=True)
    keep_ids = ids[counts >= min_depression_cells]
    depression_id = np.where(np.isin(depression_id_raw, keep_ids), depression_id_raw, 0).astype("int32")
    print(f"  kept {len(keep_ids)}/{len(ids)} depressions (>= {min_depression_cells} cells)")

    dem, _ = read_raster(dem_path)

    print("== 5/8: depression curves + alpha ==")
    hd = depression_relative_elevation(dem, depression_id)
    anchor_stage = anchor_stage_by_reach(hd, depression_id, d100)
    depression_curves_df = depression_curves(dem, depression_id, cell_area, anchor_stage=anchor_stage)
    depression_curves_df.to_parquet(outputs_dir / "depression_curves.parquet", index=False)

    dep_valid = depression_id > 0
    dep_ids, dep_counts = np.unique(depression_id[dep_valid], return_counts=True)
    footprint_area_by_id = dict(zip(dep_ids.tolist(), (dep_counts * cell_area).tolist()))

    dep_df = pd.DataFrame({
        "depression_id": depression_id[dep_valid],
        "p100": rain_p100[dep_valid],
        "p_prime": rain_p_prime[dep_valid],
    })
    p100_mean_by_depression = dep_df.groupby("depression_id")["p100"].mean().to_dict()
    p_prime_mean_by_depression = dep_df.groupby("depression_id")["p_prime"].mean().to_dict()

    alpha_df = calibrate_depression_alpha(d100, depression_id, cell_area, p100_mean_by_depression, footprint_area_by_id)
    alpha_df.to_parquet(outputs_dir / "depression_alpha.parquet", index=False)
    alpha_by_depression = alpha_df.set_index("catchment_id")["alpha"].to_dict()

    print("== 6/8: channel-ratio calibration (excluding depression cells) ==")
    k_df = calibrate_channel_ratio(d100, catchment_id, rain_p100, exclude_mask=dep_valid)
    k_df.to_parquet(outputs_dir / "channel_ratio_k.parquet", index=False)

    print("== 7/8: render + combine ==")
    channel_depth = render_channel_ratio_depth(d100, catchment_id, rain_p_prime, k_df)
    depression_depth, _ = render_pluvial_depth(
        hd, depression_id, depression_curves_df, alpha_by_depression,
        p_prime_mean_by_depression, footprint_area_by_id,
    )
    combined = combine_channel_and_depression_depth(channel_depth, depression_depth, depression_id)

    out_depth_path = outputs_dir / f"depth_{scenario_rainfall_id}_{scenario_year}.tif"
    write_raster(out_depth_path, combined.astype("float32"), hand_profile, dtype="float32", nodata=np.nan)
    print(f"  wrote {out_depth_path}")

    if skip_viz:
        return out_depth_path

    print("== 8/8: coarse catchment boundary + PNG ==")
    viz_hand_out = intermediate_dir / "hand_out_viz"
    if not (viz_hand_out / "catchment_id.tif").exists():
        build_hand_stack(dem_path, viz_hand_out, threshold=viz_threshold)
    else:
        print(f"  skip {viz_hand_out} (already exists)")

    out_png = outputs_dir / f"depth_{scenario_rainfall_id}_{scenario_year}.png"
    plot_depth_png(
        out_depth_path, out_png, title=f"{scenario_rainfall_id} {scenario_year} depth",
        basemap=True, boundary_raster=viz_hand_out / "catchment_id.tif",
    )
    print(f"  wrote {out_png}")

    return out_depth_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                         help="start with a ~2x2 degree window, not a full basin -- a full-basin DEM OOM'd "
                              "WhiteboxTools' depression breaching in this session")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                         help="flow-accumulation threshold for HAND/reach delineation (calibration granularity)")
    parser.add_argument("--viz-threshold", type=int, default=DEFAULT_VIZ_THRESHOLD,
                         help="separate, much coarser threshold for a legible boundary overlay")
    parser.add_argument("--min-depression-cells", type=int, default=DEFAULT_MIN_DEPRESSION_CELLS)
    parser.add_argument("--depth-id", default=ANCHOR_DEPTH_ID)
    parser.add_argument("--anchor-rainfall-id", default=ANCHOR_RAINFALL_ID)
    parser.add_argument("--anchor-year", type=int, default=ANCHOR_YEAR)
    parser.add_argument("--scenario-rainfall-id", default=None,
                         help="e.g. rain_ssp245_rp500 -- one of the ids in the rainfall catalog. "
                              "Defaults to the anchor (a reproduction sanity check, not a real scenario).")
    parser.add_argument("--scenario-year", type=int, default=None)
    parser.add_argument("--skip-viz", action="store_true", help="skip the coarse boundary layer and PNG")
    args = parser.parse_args()

    run_basin(
        tuple(args.bbox), args.out_dir, threshold=args.threshold, viz_threshold=args.viz_threshold,
        min_depression_cells=args.min_depression_cells, depth_id=args.depth_id,
        anchor_rainfall_id=args.anchor_rainfall_id, anchor_year=args.anchor_year,
        scenario_rainfall_id=args.scenario_rainfall_id, scenario_year=args.scenario_year,
        skip_viz=args.skip_viz,
    )


if __name__ == "__main__":
    main()
