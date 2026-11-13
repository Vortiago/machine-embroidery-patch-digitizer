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

import numpy as np
from PIL import Image, ImageDraw
from scipy import sparse
from scipy.sparse import csgraph
from skimage import measure, morphology

import sh7_writer as sw
from gradient import regions


def line_mask(img: Image.Image, threshold: int = 128) -> np.ndarray:
    """True where the line is: opaque pixels of a transparent picture, else dark ones."""
    rgba = np.asarray(img.convert("RGBA"))
    alpha = rgba[..., 3]
    if alpha.min() < 128:
        return alpha >= 128
    lum = np.asarray(img.convert("L"))
    return lum < threshold


def largest_part(mask: np.ndarray):
    """The largest 8-connected part of the mask, and how many others there are."""
    parts = regions(mask, 1)
    if not parts:
        raise sw.Sh7Error("the picture has no line pixels")
    return parts[0], len(parts) - 1


class SkeletonTree:
    """The line's 1 px skeleton as a spanning tree over its pixels.

    ``xy[i]`` is pixel i as (x, y); ``adj[i]`` lists its tree neighbours, and
    a pruned pixel has an empty list.  A spanning tree drops one pixel step
    from each closed loop (a gap far below a stitch length) so the walk below
    exists for any drawing."""

    def __init__(self, mask: np.ndarray):
        skel = morphology.skeletonize(mask)
        ys, xs = np.nonzero(skel)
        n = len(xs)
        if n == 0:
            raise sw.Sh7Error("the line is too thin to trace")
        self.xy = np.column_stack([xs, ys])
        index = np.full(skel.shape, -1, dtype=np.int64)
        index[ys, xs] = np.arange(n)
        # 8-neighbour edges, each found once: right, down, down-right, up-right.
        h, w = skel.shape
        rows, cols, weights = [], [], []
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            nx_, ny_ = xs + dx, ys + dy
            ok = (nx_ < w) & (ny_ >= 0) & (ny_ < h)
            other = np.full(n, -1, dtype=np.int64)
            other[ok] = index[ny_[ok], nx_[ok]]
            hit = other >= 0
            rows.append(np.nonzero(hit)[0])
            cols.append(other[hit])
            weights.append(np.full(hit.sum(), math.hypot(dx, dy)))
        graph = sparse.coo_matrix((np.concatenate(weights),
                                   (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        tree = csgraph.minimum_spanning_tree(graph).tocoo()
        self.adj = [[] for _ in range(n)]
        for i, j in zip(tree.row.tolist(), tree.col.tolist()):
            self.adj[i].append(j)
            self.adj[j].append(i)
        # Spurs shorter than about one and a half line widths are skeleton
        # noise from the line's edges.
        self.prune_spurs(1.5 * mask.sum() / n)

    def prune_spurs(self, max_len: float) -> None:
        adj, xy = self.adj, self.xy
        for _ in range(3):
            removed = False
            for leaf in [i for i, nb in enumerate(adj) if len(nb) == 1]:
                if len(adj[leaf]) != 1:
                    continue
                chain, prev, cur, length = [leaf], None, leaf, 0.0
                while length < max_len:
                    nxt = [j for j in adj[cur] if j != prev]
                    if len(nxt) != 1:
                        break
                    length += math.dist(xy[cur], xy[nxt[0]])
                    prev, cur = cur, nxt[0]
                    if len(adj[cur]) != 2:
                        break
                    chain.append(cur)
                if len(adj[cur]) >= 3 and length < max_len:
                    adj[cur].remove(chain[-1])
                    for i in chain:
                        adj[i] = []
                    removed = True
            if not removed:
                return

    def nodes(self):
        return [i for i, nb in enumerate(self.adj) if nb]


def sewing_walk(tree: SkeletonTree):
    """One continuous walk over the tree as (x, y) pixels, from its topmost
    pixel to its bottommost.  The branch that leads to the bottom is sewn once;
    every other branch is sewn out and back, so the stitch never has to jump."""
    xy, adj = tree.xy, tree.adj
    nodes = tree.nodes() or [0]
    top = min(nodes, key=lambda i: (xy[i][1], xy[i][0]))
    bottom = max(nodes, key=lambda i: (xy[i][1], -xy[i][0]))
    # Parents from a walk out of the top; the main path is bottom's ancestry.
    parent = {top: None}
    order = [top]
    for node in order:
        for j in adj[node]:
            if j not in parent:
                parent[j] = node
                order.append(j)
    main_child = {}
    node = bottom
    while parent[node] is not None:
        main_child[parent[node]] = node
        node = parent[node]
    walk = [top]
    # Iterative DFS: side branches first, the main-path child last.
    def children(node):
        main = main_child.get(node)
        kids = [j for j in adj[node] if j != parent[node] and j != main]
        return iter(kids + [main] if main is not None else kids)
    stack = [(top, children(top))]
    while stack:
        node, it = stack[-1]
        child = next(it, None)
        if child is None:
            stack.pop()
            if stack and main_child.get(stack[-1][0]) != node:
                walk.append(stack[-1][0])     # come back up a side branch
            continue
        walk.append(child)
        stack.append((child, children(child)))
    return [tuple(xy[i]) for i in walk]


def to_stitches(walk_px, px_per_mm: float, stitch_mm: float, tolerance_mm: float = 0.15):
    """Pixel walk to design points in mm: start at (0, 0), Y down the picture,
    simplified and cut into stitches no longer than ``stitch_mm``."""
    walk = np.asarray(walk_px, dtype=float)
    pts = (walk - walk[0]) / px_per_mm
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
    x0, y0 = min(xs), min(ys)
    w = int((max(xs) - x0) * scale) + 2 * margin
    h = int((max(ys) - y0) * scale) + 2 * margin
    img = Image.new("RGB", (max(w, 1), max(h, 1)), "white")
    d = ImageDraw.Draw(img)
    xy = [((x - x0) * scale + margin, (y - y0) * scale + margin) for x, y in points]
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
    walk = sewing_walk(SkeletonTree(mask))
    # Scale the sewn path itself (the skeleton, not the inked area) to length.
    span = max(p[1] for p in walk) - min(p[1] for p in walk)
    if span == 0:
        raise sw.Sh7Error("the line has no length along the feed (picture height)")
    px_per_mm = span / length_mm
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
