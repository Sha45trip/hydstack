"""Render a depth raster (or any single-band float raster) to a PNG for a
quick look — not a mapping product, just a fast sanity-check visualization.

Optionally overlays an OSM basemap (via contextily) and a catchment/reach
boundary raster (drawn as the edges between distinct IDs)."""
from __future__ import annotations

import numpy as np
import rasterio


def _boundary_mask(id_raster, dilate_px=2):
    """True on any cell adjacent to a cell with a different ID (including 0/
    background) — a cheap raster-native way to draw zone boundaries without
    polygonizing. Dilated a couple pixels so 1-px-wide lines survive the
    ~5x downsampling from a multi-thousand-pixel raster to a ~1500px PNG."""
    b = np.zeros(id_raster.shape, dtype=bool)
    b[:-1, :] |= id_raster[:-1, :] != id_raster[1:, :]
    b[1:, :] |= id_raster[:-1, :] != id_raster[1:, :]
    b[:, :-1] |= id_raster[:, :-1] != id_raster[:, 1:]
    b[:, 1:] |= id_raster[:, :-1] != id_raster[:, 1:]
    if dilate_px:
        from scipy.ndimage import binary_dilation
        b = binary_dilation(b, iterations=dilate_px)
    return b


def plot_depth_png(raster_path, out_png, title=None, cmap="Blues", vmax_percentile=98, dpi=150,
                    basemap=False, boundary_raster=None, crs="EPSG:4326"):
    import matplotlib.pyplot as plt

    with rasterio.open(raster_path) as src:
        depth = src.read(1, masked=True)
        bounds = src.bounds

    valid = depth.compressed()  # plain ndarray of unmasked values -- np.percentile doesn't respect masks
    wet = valid[valid > 0]
    vmax = float(np.percentile(wet, vmax_percentile)) if wet.size else 1.0

    fig, ax = plt.subplots(figsize=(10, 10 * depth.shape[0] / depth.shape[1]))

    if basemap:
        import contextily as cx

        # OSM's own tile server (Mapnik) 403s automated/app traffic that doesn't
        # follow its volunteer-run usage policy (osm.wiki/Blocked), CartoDB now
        # requires an API key, and OpenTopoMap's contour/road detail competes
        # with the data overlay. Esri's gray canvas is deliberately minimal --
        # faint place names and boundaries only -- so the depth layer stays
        # legible.
        ax.set_xlim(bounds.left, bounds.right)
        ax.set_ylim(bounds.bottom, bounds.top)
        cx.add_basemap(ax, crs=crs, source=cx.providers.Esri.WorldGrayCanvas)

    im = ax.imshow(
        depth, cmap=cmap, vmin=0, vmax=vmax, alpha=0.85 if basemap else 1.0,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
    )

    if boundary_raster is not None:
        with rasterio.open(boundary_raster) as src:
            ids = src.read(1)
        boundary = _boundary_mask(ids)
        overlay = np.zeros((*boundary.shape, 4))
        overlay[boundary] = (0.1, 0.1, 0.1, 0.6)
        ax.imshow(overlay, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))

    if not basemap:
        ax.set_facecolor("#f2f2f0")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="depth (m)", shrink=0.8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raster")
    parser.add_argument("out_png")
    parser.add_argument("--title", default=None)
    parser.add_argument("--cmap", default="Blues")
    parser.add_argument("--vmax-percentile", type=float, default=98)
    parser.add_argument("--basemap", action="store_true", help="overlay an OSM basemap (needs network access)")
    parser.add_argument("--boundary-raster", default=None, help="an ID raster (e.g. catchment_id.tif) to draw zone boundaries from")
    args = parser.parse_args()

    plot_depth_png(args.raster, args.out_png, title=args.title, cmap=args.cmap,
                    vmax_percentile=args.vmax_percentile, basemap=args.basemap,
                    boundary_raster=args.boundary_raster)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
