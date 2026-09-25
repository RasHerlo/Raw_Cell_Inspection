"""Write a synthetic movie stack and a reference image for trying out the app.

    python tools/make_demo_stack.py [out_dir] [--frames N] [--size PX]

Cells 0-3 respond during frames 200-260 and 500-560; cells 4-7 only during 350-410.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", nargs="?", default="demo_data")
    parser.add_argument("--frames", type=int, default=800)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    n, s = args.frames, args.size
    yy, xx = np.mgrid[0:s, 0:s]
    centres = rng.uniform(0.15 * s, 0.85 * s, size=(8, 2))
    radius = s / 40
    cells = [np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2))) for cx, cy in centres]

    t = np.arange(n)
    response_a = np.zeros(n)
    response_b = np.zeros(n)
    for a, b in ((200, 260), (500, 560)):
        response_a[a : b + 1] = 1.0
    response_b[350:411] = 1.0
    bleach = 1.0 - 0.15 * t / n

    base = 300 + 60 * np.sin(xx / s * 3) * np.cos(yy / s * 2)
    stack = np.empty((n, s, s), dtype=np.uint16)
    for i in range(n):
        frame = base.copy()
        for k, cell in enumerate(cells):
            resp = response_a[i] if k < 4 else response_b[i]
            frame += cell * (400 + 600 * resp)
        frame *= bleach[i]
        frame += rng.normal(0, 25, size=frame.shape)
        stack[i] = np.clip(frame, 0, 65535).astype(np.uint16)
    tifffile.imwrite(out / "demo_stack.tif", stack)

    reference = base * 0.5 + sum(cells) * 900 + rng.normal(0, 10, size=base.shape)
    tifffile.imwrite(out / "demo_reference.tif", reference.astype(np.float32))
    print(f"Wrote {out / 'demo_stack.tif'} ({n} x {s} x {s}) and {out / 'demo_reference.tif'}")
    print("Cell centres (x, y):", np.round(centres, 1).tolist())


if __name__ == "__main__":
    main()
