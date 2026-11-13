"""Tests for sh7_writer.py.

Run from the repo root:
    .venv/bin/python -m unittest tools/digitize/tests/test_sh7_writer.py -v

The ``sh7_fixtures`` folder holds designs (``designs.json``) and the .sh7
files sh7pad exported for them (sh7pad commit cf1fcf2, ``exportProjectBinary``
with its envelope and byte validator on).  The writer must reproduce each file
byte for byte.  Coordinates are mm; X is across the stitch, Y along it.
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import sh7_writer as sw  # noqa: E402

FIXTURES = os.path.join(HERE, "sh7_fixtures")


def design_from_fixture(spec):
    segments = []
    for seg in spec["segments"]:
        if seg == "straight":
            segments.append("straight")
        else:
            segments.append(sw.Satin(seg["widthStart"], seg["widthEnd"], seg["density"],
                                     seg.get("endAt", "right")))
    return sw.Sh7Design(points=spec["points"], foot=spec["foot"], tension=spec["tension"],
                        start_x_mm=spec.get("start_x_mm", 0),
                        uniform=spec.get("encoder_mode") == "uniform", segments=segments)


class MatchesSh7pad(unittest.TestCase):
    def test_every_fixture_is_byte_identical(self):
        with open(os.path.join(FIXTURES, "designs.json")) as fh:
            specs = json.load(fh)
        self.assertGreaterEqual(len(specs), 13)
        for spec in specs:
            with self.subTest(spec["name"]):
                with open(os.path.join(FIXTURES, spec["name"] + ".sh7"), "rb") as fh:
                    want = fh.read()
                self.assertEqual(sw.write_sh7(design_from_fixture(spec)).hex(), want.hex())


class FileShape(unittest.TestCase):
    def test_header_and_declared_size(self):
        data = sw.write_sh7(sw.Sh7Design(points=[(0, 0), (2, 1), (-2, 2)]))
        self.assertEqual(data[:8], b"%spx%\x01\x02\x01")
        self.assertEqual(int.from_bytes(data[8:12], "big"), len(data) - 12)

    def test_satin_selects_the_multi_element_class(self):
        straight = sw.write_sh7(sw.Sh7Design(points=[(0, 0), (0, 5)]))
        satin = sw.write_sh7(sw.Sh7Design(points=[(0, 0), (0, 5)],
                                          segments=[sw.Satin(3, 3)]))
        off = 14 + 2 * len(sw.PRODUCER)
        self.assertEqual(straight[off:off + 3], b"\x07\x01\x01")
        self.assertEqual(satin[off:off + 3], b"\x07\x05\x01")

    def test_long_move_becomes_carriage_jumps(self):
        records = sw.sew_records(sw.Sh7Design(points=[(0, 0), (10, 2)]))
        kinds = [r[0] for r in records]
        self.assertIn("jump", kinds)
        for kind, dx, dy, _x, _y in records:
            if kind == "jump":
                self.assertLessEqual(abs(dx), sw.X_UNITS_PER_MM)
            self.assertLessEqual(abs(dy), sw.STITCH_DY_MAX_RAW)
        self.assertAlmostEqual(records[-1][3], 10)
        self.assertAlmostEqual(records[-1][4], 2)


class Refusals(unittest.TestCase):
    def assertRefused(self, design, text):
        with self.assertRaises(sw.Sh7Error) as ctx:
            sw.write_sh7(design)
        self.assertIn(text, str(ctx.exception))

    def test_too_long(self):
        self.assertRefused(sw.Sh7Design(points=[(0, 0), (0, 20), (0, 44)]), "mm long")

    def test_too_wide_for_foot_b(self):
        self.assertRefused(sw.Sh7Design(points=[(0, 0), (6, 2), (-6, 4)], foot="B"), "wide")

    def test_satin_wider_than_the_needle_window(self):
        self.assertRefused(sw.Sh7Design(points=[(0, 0), (0, 5)], segments=[sw.Satin(8, 4)]),
                           "satin width")

    def test_start_stitch_must_be_on_y0(self):
        self.assertRefused(sw.Sh7Design(points=[(0, 1), (0, 5)]), "Start Stitch")

    def test_unknown_foot(self):
        self.assertRefused(sw.Sh7Design(points=[(0, 0), (0, 5)], foot="Q"), "unknown foot")


if __name__ == "__main__":
    unittest.main()
