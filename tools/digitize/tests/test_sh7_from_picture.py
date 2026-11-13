"""Tests for sh7_from_picture.py on small drawn pictures.

Run from the repo root:
    .venv/bin/python -m unittest tools/digitize/tests/test_sh7_from_picture.py -v
"""
import math
import os
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import sh7_from_picture as sfp  # noqa: E402
import sh7_writer as sw  # noqa: E402


def wave(transparent=False):
    mode, bg, ink = ("RGBA", (0, 0, 0, 0), (0, 0, 0, 255)) if transparent \
        else ("RGB", "white", "black")
    im = Image.new(mode, (200, 600), bg)
    pts = [(100 + 60 * math.sin(y / 40), y) for y in range(20, 580, 4)]
    ImageDraw.Draw(im).line(pts, fill=ink, width=8)
    return im


def y_with_loop_and_dot():
    im = Image.new("RGB", (300, 500), "white")
    d = ImageDraw.Draw(im)
    d.line([(50, 30), (150, 200), (250, 30)], fill="black", width=10)
    d.line([(150, 200), (150, 470)], fill="black", width=10)
    d.ellipse((100, 280, 200, 380), outline="black", width=10)
    d.ellipse((260, 450, 280, 470), fill="black")      # a separate dot
    return im


class PictureToDesign(unittest.TestCase):
    def check(self, design, length_mm, stitch_mm):
        pts = design.points
        self.assertEqual(pts[0], (0.0, 0.0))
        ys = [p[1] for p in pts]
        self.assertAlmostEqual(max(ys) - min(ys), length_mm, delta=0.5)
        for a, b in zip(pts, pts[1:]):
            self.assertLessEqual(math.dist(a, b), stitch_mm + 1e-6)
        data = sw.write_sh7(design)
        self.assertEqual(data[:5], b"%spx%")

    def test_wave(self):
        design, notes = sfp.picture_to_design(wave(), length_mm=30, stitch_mm=1.5)
        self.check(design, 30, 1.5)
        self.assertNotIn("dropped", " ".join(notes))

    def test_transparent_picture_uses_alpha(self):
        design, _ = sfp.picture_to_design(wave(transparent=True), length_mm=20, stitch_mm=1.0)
        self.check(design, 20, 1.0)

    def test_branches_are_sewn_out_and_back_and_parts_dropped(self):
        design, notes = sfp.picture_to_design(y_with_loop_and_dot(), length_mm=30, stitch_mm=2)
        self.check(design, 30, 2)
        self.assertIn("1 separate part(s) dropped", notes[0])
        # Starts at the top-left arm, ends at the bottom of the stem.
        self.assertAlmostEqual(design.points[-1][1], 30, delta=0.5)

    def test_length_limit(self):
        with self.assertRaises(sw.Sh7Error):
            sfp.picture_to_design(wave(), length_mm=50, stitch_mm=1.5)

    def test_cli_writes_file_and_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "wave.png")
            wave().save(src)
            out, prev = os.path.join(tmp, "wave.sh7"), os.path.join(tmp, "wave_preview.png")
            self.assertEqual(sfp.main([src, out, "--length-mm", "25", "--preview", prev]), 0)
            self.assertTrue(os.path.getsize(out) > 0)
            self.assertTrue(os.path.getsize(prev) > 0)

    def test_cli_reports_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "wide.png")
            im = Image.new("RGB", (600, 100), "white")
            ImageDraw.Draw(im).line([(10, 10), (590, 90)], fill="black", width=6)
            im.save(src)
            # 600 px wide at 100 px = 40 mm long is 240 mm wide: past any foot.
            self.assertEqual(sfp.main([src, os.path.join(tmp, "x.sh7"), "--length-mm", "40"]), 1)


if __name__ == "__main__":
    unittest.main()
