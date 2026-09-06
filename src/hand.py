"""Step 1 — DEM conditioning, flow routing, HAND, and reach/catchment delineation.

Produces the three static rasters everything downstream reads: hand.tif,
reach_id.tif, catchment_id.tif, aligned and tiled identically to the d100 anchor.

Flow routing is done with WhiteboxTools (breaching, D8, stream links, subbasins,
elevation-above-stream). It is lazy-imported so the rest of this module — and
anything that imports it — stays usable without the whitebox binary installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize

DEFAULT_FLOW_ACC_THRESHOLD_CELLS = 500
DEFAULT_BREACH_MAX_DIST_CELLS = 1000
HECRAS_SNAP_BUFFER_CELLS = 2


def _wbt(working_dir: Path):
    from whitebox import WhiteboxTools

    wbt = WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.set_working_dir(str(Path(working_dir).resolve()))
    return wbt


def _abs(path):
    """WhiteboxTools resolves relative input/output paths against --wd, not
    the calling process's cwd — always pass it absolute paths."""
    return str(Path(path).resolve())


def _run_wbt(tool_fn, output_path, **kwargs):
    """Call a WhiteboxTools tool and verify it actually produced output.

    The Python wrapper can return 0 even when the underlying exe panics (e.g.
    on a bad path) — the only reliable failure signal is the output file's
    existence.
    """
    tool_fn(**kwargs)
    output_path = Path(output_path)
    if not output_path.exists():
        raise RuntimeError(
            f"WhiteboxTools call {tool_fn.__name__!r} did not produce {output_path} "
            "(check stdout above, or call wbt.set_verbose_mode(True) for details)"
        )
    return output_path


def read_raster(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()


def write_raster(path, array, profile, dtype=None, nodata=None):
    profile = profile.copy()
    dtype = dtype or array.dtype
    # A source profile's own blockxsize/blockysize (e.g. from a WhiteboxTools
    # output) isn't guaranteed to divide this array's dimensions -- GDAL then
    # refuses to write ("BLOCKXSIZE must be a multiple of 16"). Let it pick a
    # compatible block size instead of inheriting one that may not fit.
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(dtype=dtype, count=1, compress="deflate", tiled=True)
    if nodata is not None:
        profile["nodata"] = nodata
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def condition_dem(dem_path, out_path, max_dist=DEFAULT_BREACH_MAX_DIST_CELLS):
    """Breach depressions rather than fill, so flow paths survive embankments with culverts."""
    out_path = Path(out_path)
    wbt = _wbt(out_path.parent)
    _run_wbt(wbt.breach_depressions_least_cost, out_path,
              dem=_abs(dem_path), output=_abs(out_path), dist=max_dist, fill=True)
    return out_path


def flow_direction(conditioned_dem_path, out_path):
    out_path = Path(out_path)
    wbt = _wbt(out_path.parent)
    _run_wbt(wbt.d8_pointer, out_path, dem=_abs(conditioned_dem_path), output=_abs(out_path))
    return out_path


def flow_accumulation(conditioned_dem_path, out_path):
    out_path = Path(out_path)
    wbt = _wbt(out_path.parent)
    _run_wbt(wbt.d8_flow_accumulation, out_path,
              i=_abs(conditioned_dem_path), output=_abs(out_path), out_type="cells")
    return out_path


def extract_streams(flow_accum_path, out_path, threshold=DEFAULT_FLOW_ACC_THRESHOLD_CELLS):
    out_path = Path(out_path)
    wbt = _wbt(out_path.parent)
    _run_wbt(wbt.extract_streams, out_path,
              flow_accum=_abs(flow_accum_path), output=_abs(out_path), threshold=threshold)
    return out_path


def compute_hand(conditioned_dem_path, streams_path, out_path):
    out_path = Path(out_path)
    wbt = _wbt(out_path.parent)
    _run_wbt(wbt.elevation_above_stream, out_path,
              dem=_abs(conditioned_dem_path), streams=_abs(streams_path), output=_abs(out_path))
    return out_path


def assign_reach_ids(streams_path, pointer_path, out_path):
    """Unique ID per stream link — this becomes reach_id before any HEC-RAS snapping."""
    out_path = Path(out_path)
    wbt = _wbt(out_path.parent)
    _run_wbt(wbt.stream_link_identifier, out_path,
              d8_pntr=_abs(pointer_path), streams=_abs(streams_path), output=_abs(out_path))
    return out_path


def assign_catchment_ids(pointer_path, reach_id_path, out_path):
    """Per-reach contributing catchment. Subbasin IDs match the link ID they drain to."""
    out_path = Path(out_path)
    wbt = _wbt(out_path.parent)
    _run_wbt(wbt.subbasins, out_path,
              d8_pntr=_abs(pointer_path), streams=_abs(reach_id_path), output=_abs(out_path))
    return out_path


def snap_reach_ids_to_hecras(reach_id_path, hecras_lines_gdf, id_field, out_path,
                              buffer_cells=HECRAS_SNAP_BUFFER_CELLS):
    """Relabel delineated reaches to HEC-RAS centerline IDs where they agree.

    For each local reach, the HEC-RAS line whose buffered footprint covers the most
    of that reach's cells wins. Reaches with no HEC-RAS line within the buffer keep
    their local ID, offset to guarantee no collision with real HEC-RAS IDs.
    """
    reach_id, profile = read_raster(reach_id_path)
    transform = profile["transform"]
    cell_size = abs(transform.a)
    buffer_dist = buffer_cells * cell_size

    hecras_ids = hecras_lines_gdf[id_field].to_numpy()
    shapes = [
        (geom.buffer(buffer_dist), hid)
        for geom, hid in zip(hecras_lines_gdf.geometry, hecras_ids)
    ]
    hecras_grid = rasterize(
        shapes, out_shape=reach_id.shape, transform=transform, fill=0, dtype="int64",
    )

    local_offset = int(max(hecras_ids.max(), 0)) + 1
    out = reach_id.astype("int64").copy()
    for local_id in np.unique(reach_id):
        if local_id <= 0:
            continue
        mask = reach_id == local_id
        overlap = hecras_grid[mask]
        overlap = overlap[overlap > 0]
        if overlap.size == 0:
            out[mask] = local_id + local_offset
            continue
        values, counts = np.unique(overlap, return_counts=True)
        out[mask] = values[np.argmax(counts)]

    write_raster(out_path, out, profile, dtype="int32", nodata=0)
    return out_path


def build_hand_stack(dem_path, out_dir, threshold=DEFAULT_FLOW_ACC_THRESHOLD_CELLS,
                      hecras_lines_gdf=None, hecras_id_field=None):
    """Run Step 1 end to end for one basin. Returns paths to the three deliverables."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditioned = condition_dem(dem_path, out_dir / "dem_conditioned.tif")
    pointer = flow_direction(conditioned, out_dir / "d8_pointer.tif")
    accum = flow_accumulation(conditioned, out_dir / "flow_accum.tif")
    streams = extract_streams(accum, out_dir / "streams.tif", threshold=threshold)

    hand_path = compute_hand(conditioned, streams, out_dir / "hand.tif")
    reach_path = assign_reach_ids(streams, pointer, out_dir / "reach_id_raw.tif")

    if hecras_lines_gdf is not None:
        reach_path = snap_reach_ids_to_hecras(
            reach_path, hecras_lines_gdf, hecras_id_field, out_dir / "reach_id.tif"
        )
    else:
        reach_path = Path(reach_path).rename(out_dir / "reach_id.tif")

    catchment_path = assign_catchment_ids(pointer, reach_path, out_dir / "catchment_id.tif")

    return {"hand": hand_path, "reach_id": reach_path, "catchment_id": catchment_path}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dem", help="path to the 30 m conditioned-input DEM for one basin")
    parser.add_argument("out_dir", help="directory to write hand.tif, reach_id.tif, catchment_id.tif")
    parser.add_argument("--threshold", type=int, default=DEFAULT_FLOW_ACC_THRESHOLD_CELLS,
                         help="flow-accumulation cell threshold for stream extraction")
    parser.add_argument("--hecras-centerlines", default=None,
                         help="vector file of HEC-RAS centerlines to snap reach IDs to")
    parser.add_argument("--hecras-id-field", default="reach_id",
                         help="attribute field on --hecras-centerlines holding the reach ID")
    args = parser.parse_args()

    hecras_gdf = None
    if args.hecras_centerlines:
        import geopandas as gpd

        hecras_gdf = gpd.read_file(args.hecras_centerlines)

    paths = build_hand_stack(
        args.dem, args.out_dir, threshold=args.threshold,
        hecras_lines_gdf=hecras_gdf, hecras_id_field=args.hecras_id_field,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
