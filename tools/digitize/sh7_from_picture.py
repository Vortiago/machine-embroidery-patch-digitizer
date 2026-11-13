#!/usr/bin/env python3
"""Turn single-colour line art into a .sh7 decorative stitch (running stitch).

A .sh7 file is one thread with no trims, so the whole drawing must be sewn as
one continuous path.  The script:

1. takes the dark (or opaque) pixels of the picture as the line;
2. thins them to a 1 px skeleton;
3. walks the skeleton from the top of the drawing to its bottom in one go: the
   route to the bottom is sewn once, side branches out and back;
4. scales it so the drawing is ``--length-mm`` long in the feed direction
   (picture top to bottom), simplifies it and cuts it into stitches of at
   most ``--stitch-mm``;
5. writes the .sh7 with ``sh7_writer`` and, if asked, a PNG of the stitch path.

Only the largest connected part of the drawing is kept; the others are
reported, because joining them would sew a visible thread across the fabric.

Usage:
    sh7_from_picture.py <picture> <out.sh7> [--length-mm 30] [--stitch-mm 1.5]
                        [--foot S|B|hidden] [--tension 4.0] [--preview out.png]

    tools/digitize/sh7_from_picture.py input/wave.png output/wave.sh7 --length-mm 25
"""
import argparse
import math
import sys

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure, morphology

import sh7_writer as sw


def line_mask(img: Image.Image, threshold: int = 128) -> np.ndarray:
    """True where the line is: opaque pixels of a transparent picture, else dark ones."""
    rgba = np.asarray(img.convert("RGBA"))
    alpha = rgba[..., 3]
    if alpha.min() < 128:
        return alpha >= 128
    lum = np.asarray(img.convert("L"))
    return lum < threshold


def largest_part(mask: np.ndarray):
    labels = measure.label(mask, connectivity=2)
    if labels.max() == 0:
        raise sw.Sh7Error("the picture has no line pixels")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    keep = int(sizes.argmax())
    return labels == keep, int(labels.max()) - 1


def skeleton_tree(mask: np.ndarray) -> nx.Graph:
    """The line's 1 px skeleton as a tree of pixels (x, y).

    A spanning tree drops one pixel step from each closed loop (a gap far
    below a stitch length) and lets the walk below exist for any drawing.
    Spurs shorter than about one and a half line widths are skeleton noise
    from the line's edges and are pruned."""
    skel = morphology.skeletonize(mask)
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        raise sw.Sh7Error("the line is too thin to trace")
    pixels = set(zip(xs.tolist(), ys.tolist()))
    g = nx.Graph()
    g.add_nodes_from(pixels)
    for x, y in pixels:
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            q = (x + dx, y + dy)
            if q in pixels:
                g.add_edge((x, y), q, weight=math.hypot(dx, dy))
    tree = nx.minimum_spanning_tree(g)
    width = mask.sum() / max(1, len(pixels))
    _prune_spurs(tree, 1.5 * width)
    return tree


def _prune_spurs(tree: nx.Graph, max_len: float) -> None:
    for _ in range(3):
        removed = False
        for leaf in [n for n in tree if tree.degree(n) == 1]:
            if leaf not in tree or tree.degree(leaf) != 1:
                continue
            chain, cur, prev, length = [leaf], leaf, None, 0.0
            while True:
                nxt = [n for n in tree[cur] if n != prev]
                if len(nxt) != 1:
                    break
                length += tree[cur][nxt[0]]["weight"]
                prev, cur = cur, nxt[0]
                if tree.degree(cur) != 2:
                    break
                chain.append(cur)
            if tree.degree(cur) >= 3 and length < max_len:
                tree.remove_nodes_from(chain)
                removed = True
        if not removed:
            return


def sewing_walk(tree: nx.Graph):
    """One continuous walk over the tree, from its topmost pixel to its
    bottommost.  The branch that leads to the bottom is sewn once; every other
    branch is sewn out and back, so the stitch never has to jump."""
    top = min(tree, key=lambda p: (p[1], p[0]))
    bottom = max(tree, key=lambda p: (p[1], -p[0]))
    on_main = set(nx.shortest_path(tree, top, bottom))
    walk = [top]
    # Iterative DFS: visit side branches first, the main-path child last.
    stack = [(top, None, iter(_children(tree, top, None, on_main)))]
    while stack:
        node, parent, it = stack[-1]
        child = next(it, None)
        if child is None:
            stack.pop()
            if stack and node not in on_main:
                walk.append(stack[-1][0])     # come back up a side branch
            continue
        walk.append(child)
        stack.append((child, node, iter(_children(tree, child, node, on_main))))
    return walk


def _children(tree, node, parent, on_main):
    kids = [n for n in tree[node] if n != parent]
    return sorted(kids, key=lambda n: n in on_main)   # main-path child last


def to_stitches(walk_px, px_per_mm: float, stitch_mm: float, tolerance_mm: float = 0.15):
    """Pixel walk to design points in mm: start at (0, 0), Y down the picture,
    simplified and cut into stitches no longer than ``stitch_mm``."""
    x0, y0 = walk_px[0]
    pts = np.array([((x - x0) / px_per_mm, (y - y0) / px_per_mm) for x, y in walk_px])
    pts = measure.approximate_polygon(pts, tolerance=tolerance_mm)
    out = [(0.0, 0.0)]
    for a, b in zip(pts, pts[1:]):
        n = max(1, math.ceil(math.dist(a, b) / stitch_mm))
        for i in range(1, n + 1):
            out.append((round(float(a[0] + (b[0] - a[0]) * i / n), 3),
                        round(float(a[1] + (b[1] - a[1]) * i / n), 3)))
    return out


def draw_preview(points, path, scale=20, margin=10):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    w = int((max(xs) - min(xs)) * scale) + 2 * margin
    h = int((max(ys) - min(ys)) * scale) + 2 * margin
    img = Image.new("RGB", (max(w, 1), max(h, 1)), "white")
    d = ImageDraw.Draw(img)
    xy = [((x - min(xs)) * scale + margin, (y - min(ys)) * scale + margin) for x, y in points]
    d.line(xy, fill=(30, 30, 30), width=2)
    for x, y in xy:
        d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(200, 40, 40))
    img.save(path)


def picture_to_design(img: Image.Image, length_mm: float, stitch_mm: float,
                      foot: str = "S", tension: float = 4.0, threshold: int = 128):
    """Return (Sh7Design, notes) for a line-art picture."""
    if not 0 < length_mm <= sw.SH7_MAX_Y_MM:
        raise sw.Sh7Error(f"--length-mm must be between 0 and {sw.SH7_MAX_Y_MM}")
    notes = []
    mask, dropped = largest_part(line_mask(img, threshold))
    if dropped:
        notes.append(f"kept the largest part of the drawing; {dropped} separate "
                     f"part(s) dropped (.sh7 has no trims)")
    rows = np.nonzero(mask.any(axis=1))[0]
    px_per_mm = (rows[-1] - rows[0] + 1) / length_mm
    walk = sewing_walk(skeleton_tree(mask))
    points = to_stitches(walk, px_per_mm, stitch_mm)
    width = max(p[0] for p in points) - min(p[0] for p in points)
    notes.append(f"{len(points) - 1} stitches, {width:.1f} x {length_mm:.1f} mm")
    return sw.Sh7Design(points=points, foot=foot, tension=tension), notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("picture")
    ap.add_argument("out")
    ap.add_argument("--length-mm", type=float, default=30.0,
                    help="length along the feed direction (picture height), max 43.6")
    ap.add_argument("--stitch-mm", type=float, default=1.5, help="longest stitch, mm")
    ap.add_argument("--foot", choices=sorted(sw.FEET), default="S")
    ap.add_argument("--tension", type=float, default=4.0)
    ap.add_argument("--threshold", type=int, default=128,
                    help="grey level below which an opaque pixel is line (0-255)")
    ap.add_argument("--preview", help="also write a PNG of the stitch path")
    args = ap.parse_args(argv)
    try:
        design, notes = picture_to_design(Image.open(args.picture), args.length_mm,
                                          args.stitch_mm, args.foot, args.tension,
                                          args.threshold)
        data = sw.write_sh7(design)
    except sw.Sh7Error as exc:
        print(f"sh7_from_picture: {exc}", file=sys.stderr)
        return 1
    with open(args.out, "wb") as fh:
        fh.write(data)
    if args.preview:
        draw_preview(design.points, args.preview)
    for note in notes:
        print(note)
    print(f"wrote {args.out} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
