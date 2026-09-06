"""Render a depth raster (or any single-band float raster) to a PNG for a
quick look — not a mapping product, just a fast sanity-check visualization."""
from __future__ import annotations

import numpy as np
import rasterio


def plot_depth_png(raster_path, out_png, title=None, cmap="Blues", vmax_percentile=98, dpi=150):
    import matplotlib.pyplot as plt

    with rasterio.open(raster_path) as src:
        depth = src.read(1, masked=True)
        bounds = src.bounds

    valid = depth.compressed()  # plain ndarray of unmasked values -- np.percentile doesn't respect masks
    wet = valid[valid > 0]
    vmax = float(np.percentile(wet, vmax_percentile)) if wet.size else 1.0

    fig, ax = plt.subplots(figsize=(10, 10 * depth.shape[0] / depth.shape[1]))
    im = ax.imshow(
        depth, cmap=cmap, vmin=0, vmax=vmax,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
    )
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
    args = parser.parse_args()

    plot_depth_png(args.raster, args.out_png, title=args.title, cmap=args.cmap, vmax_percentile=args.vmax_percentile)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
