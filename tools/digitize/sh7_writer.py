#!/usr/bin/env python3
"""Write Husqvarna Viking ``.sh7`` decorative-stitch files.

``.sh7`` is the format the machine's decorative-stitch menu loads: one thread,
sewn by the presser foot's needle swing and a carriage that slides sideways,
not by an embroidery hoop.  No public spec exists.  This module is a Python
port of the encoder in sh7pad (https://github.com/Vortiago/sh7pad, MIT), whose
``FORMAT.md`` documents the reverse-engineered layout.  The tests in
``tests/test_sh7_writer.py`` compare its output byte for byte with files that
sh7pad exports.

A design is a chain of points joined by straight segments:

    design = Sh7Design(points=[(0, 0), (2, 1), (-2, 2), (0, 3)])
    data = write_sh7(design)

``points[0]`` is the Start Stitch, the first needle drop; its Y must be 0.
X runs across the stitch (mm, + is right), Y runs along it (mm, + is the
direction the fabric feeds).  The writer decides which records are needle
stitches and which are carriage jumps, under the foot's limits.

Format limits the writer enforces (sh7pad ``sh7Limits.ts`` and ``foot.ts``):

- design height (Y) at most 43.69 mm;
- needle window 7 mm (3.5 mm either side of the carriage);
- a carriage jump moves at most 1 mm per record, and the carriage stays
  within the foot's reach (Foot S 27.25 mm either side, Foot B 4.5 mm).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import sh7_templates as tpl

X_UNITS_PER_MM = 8    # stitch deltas: 1/8 mm per raw X unit
Y_UNITS_PER_MM = 12   # stitch deltas: 1/12 mm per raw Y unit

SH7_MAX_Y_MM = 43.6      # sh7pad's encoder envelope (format cap is 43.69 mm)
SATIN_WIDTH_MIN_MM = 0.25
SATIN_WIDTH_MAX_MM = 7.0  # wider cones load blank or are rejected on the machine
STITCH_DY_MAX_RAW = 4 * Y_UNITS_PER_MM   # |dy| <= 4 mm per record
NEEDLE_SLOT_HALF_MM = 3.5
SHORT_DX_INT8_MAX = 127
EPSILON = 1e-6

PRODUCER = "sh7pad"   # the firmware ignores the producer string's contents


@dataclass(frozen=True)
class Foot:
    id: str
    byte: int
    reach_half_mm: float
    slot_half_mm: float = NEEDLE_SLOT_HALF_MM


FEET = {
    "S": Foot("S", 0x07, 27.25),        # side-motion foot
    "B": Foot("B", 0x02, 4.5),          # decorative foot
    "hidden": Foot("hidden", 0xFF, 27.25),  # no foot suggestion
}


class Sh7Error(ValueError):
    """The design cannot be written as a valid .sh7 file."""


@dataclass
class Sh7Design:
    """A design-mode chain: ``points[i]`` to ``points[i + 1]`` is one segment."""
    points: Sequence[Tuple[float, float]]
    foot: str = "S"
    tension: float = 4.0          # upper-thread tension as the machine shows it
    start_x_mm: float = 0.0       # Carriage Start: carriage X when the file loads
    uniform: bool = False         # cap every record at 1 mm (even needle spacing)
    # One entry per segment: "straight" or a Satin.  None means all straight.
    segments: Optional[Sequence[Union[str, "Satin"]]] = None


@dataclass(frozen=True)
class Satin:
    """A satin cone along the segment's spine: a zigzag the needle swing sews
    while the carriage rides the spine.  Widths are across the spine, in mm."""
    width_start: float
    width_end: float
    density: float = 0.4          # zigzag drop spacing along the spine, mm
    end_at: str = "right"         # leave the cone at its "left", "center" or "right"


# ---------------------------------------------------------------------------
# JavaScript number semantics.  The tests compare bytes with sh7pad, so every
# rounding step has to match the TypeScript encoder exactly.

def _js_round(x: float) -> int:
    """Math.round: halves round toward +infinity (Python's round() is banker's)."""
    return math.floor(x + 0.5)


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _trunc(x: float) -> int:
    return int(x)


# ---------------------------------------------------------------------------
# Carriage planner (sh7pad ``carriagePlanner.ts``)
#
# A short record moves only the needle, inside the slot around the planted
# carriage.  A jump record moves needle and carriage together, at most 1 mm.

@dataclass
class _Record:
    kind: str        # "short" or "jump"
    dx: int
    dy: int
    end_x_mm: float
    end_y_mm: float
    carriage_x_mm: float


def _plan(foot: Foot, deltas: Sequence[Tuple[int, int]], uniform: bool,
          cursor_x: int, carriage_x: int) -> List[List[_Record]]:
    """Return one list of records per input (dx, dy) segment delta."""
    slot_half = foot.slot_half_mm * X_UNITS_PER_MM
    reach_half = foot.reach_half_mm * X_UNITS_PER_MM
    max_needle_mm = 1.0 if uniform else math.inf
    max_dx = max_needle_mm * X_UNITS_PER_MM
    max_dy = max_needle_mm * Y_UNITS_PER_MM
    piece_dx_cap = min(X_UNITS_PER_MM, max_dx)
    piece_dy_cap = min(STITCH_DY_MAX_RAW, max_dy)
    cursor_y = 0
    buckets: List[List[_Record]] = []

    def push(bucket, kind, dx, dy):
        bucket.append(_Record(kind, int(dx), int(dy), cursor_x / X_UNITS_PER_MM,
                              cursor_y / Y_UNITS_PER_MM, carriage_x / X_UNITS_PER_MM))

    def refuse(seg_idx):
        raise Sh7Error(
            f"segment {seg_idx} cannot be sewn with foot {foot.id}: the carriage "
            f"would pass its reach of {foot.reach_half_mm} mm either side")

    for seg_idx, (seg_dx, seg_dy) in enumerate(deltas):
        bucket: List[_Record] = []
        buckets.append(bucket)
        target_x = cursor_x + seg_dx

        # Fast path: the whole segment is one needle swing.
        if (abs(target_x - carriage_x) <= slot_half + EPSILON
                and abs(seg_dy) <= STITCH_DY_MAX_RAW
                and abs(seg_dx) <= max_dx and abs(seg_dy) <= max_dy):
            cursor_x = target_x
            cursor_y += seg_dy
            push(bucket, "short", seg_dx, seg_dy)
            continue

        if max_dx <= piece_dx_cap:
            # Uniform mode: split evenly into <= 1 mm pieces.
            n = max(1, math.ceil(abs(seg_dx) / piece_dx_cap),
                    math.ceil(abs(seg_dy) / piece_dy_cap))
            sx = _sign(seg_dx)
            step_x = sx * math.floor(abs(seg_dx) / n)
            x_rem = abs(seg_dx) - abs(step_x) * n
            step_y = _trunc(seg_dy / n)
            y_rem = seg_dy - step_y * n
            for _ in range(n):
                extra_x = sx if x_rem > 0 else 0
                if x_rem > 0:
                    x_rem -= 1
                sy = _sign(y_rem)
                if y_rem != 0:
                    y_rem -= sy
                dx_piece = step_x + extra_x
                dy_piece = step_y + sy
                next_x = cursor_x + dx_piece
                in_slot = abs(next_x - carriage_x) <= slot_half + EPSILON
                cursor_x = next_x
                cursor_y += dy_piece
                if not in_slot:
                    next_carriage = carriage_x + dx_piece
                    if abs(next_carriage) > reach_half + EPSILON:
                        refuse(seg_idx)
                    carriage_x = next_carriage
                push(bucket, "short" if in_slot else "jump", dx_piece, dy_piece)
            continue

        # Compact mode.  Phase A: needle stitches out to the slot edge.
        rem_dx, rem_dy = seg_dx, seg_dy
        sign_x = _sign(seg_dx) or 1
        sign_y = _sign(seg_dy) or 1
        while abs(rem_dx) > EPSILON:
            slot_edge = carriage_x + sign_x * slot_half
            reach_along = (slot_edge - cursor_x) * sign_x
            if reach_along <= EPSILON:
                break
            a_dx = sign_x * min(abs(rem_dx), reach_along, SHORT_DX_INT8_MAX, max_dx)
            a_dy = _js_round(rem_dy * a_dx / rem_dx) if rem_dx != 0 else 0
            if abs(a_dy) > piece_dy_cap:
                a_dy = sign_y * piece_dy_cap
                if rem_dy != 0:
                    scaled = _js_round(rem_dx * a_dy / rem_dy)
                    a_dx = sign_x * max(1, min(abs(scaled), reach_along, SHORT_DX_INT8_MAX,
                                               max_dx, abs(rem_dx)))
            cursor_x += a_dx
            cursor_y += a_dy
            rem_dx -= a_dx
            rem_dy -= a_dy
            push(bucket, "short", a_dx, a_dy)
            if abs(cursor_x - carriage_x) >= slot_half - EPSILON:
                break

        # Phase B: walk the carriage in 1 mm jumps, dy along the segment line.
        if abs(rem_dx) > EPSILON:
            start_dx_abs = abs(rem_dx)
            start_dy = rem_dy
            n = max(1, math.ceil(start_dx_abs / piece_dx_cap),
                    math.ceil(abs(start_dy) / piece_dy_cap))
            used_dx = 0
            used_dy = 0
            for i in range(n):
                left = start_dx_abs - used_dx
                piece_abs = min(piece_dx_cap, left) if i < n - 1 else left
                dx_piece = sign_x * piece_abs
                used_dx += piece_abs
                target_dy = _js_round(start_dy * used_dx / start_dx_abs)
                dy_piece = target_dy - used_dy
                used_dy = target_dy
                next_carriage = carriage_x + dx_piece
                if abs(next_carriage) > reach_half + EPSILON:
                    refuse(seg_idx)
                cursor_x += dx_piece
                cursor_y += dy_piece
                carriage_x = next_carriage
                rem_dx -= dx_piece
                rem_dy -= dy_piece
                push(bucket, "jump", dx_piece, dy_piece)

        # Whatever Y is left goes in straight-down needle stitches.
        if abs(rem_dy) > EPSILON:
            n = max(1, math.ceil(abs(rem_dy) / piece_dy_cap))
            step_y = _trunc(rem_dy / n)
            y_rem = rem_dy - step_y * n
            for _ in range(n):
                sy = _sign(y_rem)
                dy_piece = step_y + sy
                if y_rem != 0:
                    y_rem -= sy
                cursor_y += dy_piece
                push(bucket, "short", 0, dy_piece)
    return buckets


# ---------------------------------------------------------------------------
# Byte encoders (sh7pad ``sh7Codec.ts`` and ``recordCodec.ts``)

def _be16(v: int) -> bytes:
    if not 0 <= v <= 0xFFFF:
        raise Sh7Error(f"BE16 out of range: {v}")
    return v.to_bytes(2, "big")


def _be32(v: int) -> bytes:
    if not 0 <= v <= 0xFFFFFFFF:
        raise Sh7Error(f"BE32 out of range: {v}")
    return v.to_bytes(4, "big")


def _i8(v: int) -> int:
    if not -128 <= v <= 127:
        raise Sh7Error(f"int8 out of range: {v}")
    return v & 0xFF


def _encode_record(kind: str, dx: int, dy: int) -> bytes:
    if kind == "short":
        if dx == -128:
            raise Sh7Error("short stitch dx = -128 collides with the jump prefix 0x80")
        return bytes([_i8(dx), _i8(dy)])
    dx_hi = 0 if dx == 0 else _sign(dx) * math.ceil(abs(dx) / X_UNITS_PER_MM)
    dx_low = dx - dx_hi * X_UNITS_PER_MM
    return bytes([0x80, 0x23, _i8(dx_low), _i8(dy), _i8(dx_hi), 0x80, 0x03])


def _chunk(prefix: bytes, payload: bytes) -> bytes:
    return prefix + _be32(len(payload)) + payload


SLOT_PATTERN = (0x3C, 0x3C, 0x3C, 0x3C, 0x2D, 0x1E, 0x1E, 0x2D, 0x2D)
TENSION_BUMP_SLOT, TENSION_BUMP = 3, 6
O6_STRIDE = {"singleton": 118, "multi": 113}
O6_XUM_OFFSETS = {"singleton": (0x57, 0x5B), "multi": (0x4F, 0x53)}


def _slot_tension(slot: int, tension_byte: int) -> int:
    return tension_byte + TENSION_BUMP if slot == TENSION_BUMP_SLOT else tension_byte


def _o6_block(cls: str, foot_byte: int, tension_byte: int, x_um: int, y_um: int) -> bytes:
    out = bytearray(tpl.SINGLETON_O6_BLOCK if cls == "singleton" else tpl.MULTI_O6_BLOCK)
    val0, val1, val2 = x_um, y_um, _js_round(y_um * 1.5)
    if max(val0, val1, val2) > 0xFFFF:
        raise Sh7Error(f"design too large for the 0x06 chunk: {x_um} x {y_um} um")
    xa, xb = O6_XUM_OFFSETS[cls]
    for slot in range(9):
        base = slot * O6_STRIDE[cls]
        out[base + 0x0C] = foot_byte
        out[base + 0x16] = _slot_tension(slot, tension_byte)
        out[base + 0x24:base + 0x26] = _be16(val0)
        out[base + 0x28:base + 0x2A] = _be16(val1)
        out[base + 0x2C:base + 0x2E] = _be16(val2)
        out[base + 0x2F:base + 0x33] = _be32(val0)
        out[base + xa:base + xa + 4] = _be32(x_um)
        out[base + xb:base + xb + 4] = _be32(x_um)
    return bytes(out)


def _o5_singleton(slot: int, tension_byte: int, x_um: int, y_um: int, x_elem: int) -> bytes:
    payload = bytearray(32)
    payload[0x04:0x08] = _be32(x_elem)
    payload[0x10] = _slot_tension(slot, tension_byte)
    payload[0x11:0x15] = _be32(y_um)
    payload[0x15:0x19] = _be32(x_um)
    payload[0x1E] = SLOT_PATTERN[slot]
    return _chunk(b"\x05\x01\x02", bytes(payload))


def _header(file_size: int) -> bytes:
    producer = PRODUCER.encode("utf-16-be")
    return (b"%spx%" + b"\x01\x02\x01" + _be32(file_size - 12)
            + _be16(len(producer)) + producer)


def _finalize(outer_prefix: bytes, o6: bytes, o5: bytes, geometry: bytes) -> bytes:
    payload = tpl.METADATA_TABLE + b"\x09" + o6 + b"\x09" + o5 + geometry
    outer = _chunk(outer_prefix, payload)
    header_len = 14 + 2 * len(PRODUCER)
    return _header(header_len + len(outer)) + outer


def _singleton_file(foot_byte: int, tension_byte: int, x_um: int, y_um: int,
                    x_elem: int, records: Sequence[Tuple[str, int, int]]) -> bytes:
    o6 = _o6_block("singleton", foot_byte, tension_byte, x_um, y_um)
    o5 = b"".join(_o5_singleton(s, tension_byte, x_um, y_um, x_elem) for s in range(9))
    stitch_chunk = _chunk(b"\x02\x01\x01", b"".join(_encode_record(*r) for r in records))
    geometry = _chunk(b"\x01\x03\x01\x01",
                      _be32(125) + _be32(125) + _be32(1000) + _be32(x_elem) + stitch_chunk)
    return _finalize(b"\x07\x01\x01", o6, o5, geometry)


def _o5_multi(slot: int, tension_byte: int, x_um: int, y_um: int) -> bytes:
    out = bytearray(tpl.MULTI_O5_CHUNK)
    p = 7  # payload starts after the 7-byte chunk header
    out[p + 0x10] = _slot_tension(slot, tension_byte)
    out[p + 0x11:p + 0x15] = _be32(y_um)
    out[p + 0x15:p + 0x19] = _be32(x_um)
    out[p + 0x1F] = SLOT_PATTERN[slot]
    return bytes(out)


def _satin_payload(left: Sequence[Tuple[float, float]], right: Sequence[Tuple[float, float]]) -> bytes:
    """Cone edges in a local frame whose minimum X and Y sit at 0 (the machine
    reads these BE32 values as unsigned).  Y uses the X scale: 1500 um per mm."""
    pts = list(left) + list(right)
    min_x = min(p[0] for p in pts)
    min_y = min(p[1] for p in pts)

    def edge(points):
        out = _be16(len(points))
        for x, y in points:
            out += _be32(_js_round((x - min_x) * 1000)) + _be32(_js_round((y - min_y) * 1500))
        return out
    return _be16(1) + edge(left) + edge(right) + _be16(0)


def _multi_file(foot_byte: int, tension_byte: int, x_um: int, y_um: int, blocks) -> bytes:
    o6 = _o6_block("multi", foot_byte, tension_byte, x_um, y_um)
    o5 = b"".join(_o5_multi(s, tension_byte, x_um, y_um) for s in range(9))
    parts, prev = [], None
    for b in blocks:
        if b["kind"] == "element":
            parts.append(_be32(125) + _be32(125) + _be32(1000)
                         + _be32(b["x_elem"]) + _be32(b["y_pos"]))
            parts.append(_chunk(b"\x02\x01\x01",
                                b"".join(_encode_record(*r) for r in b["records"])))
        else:
            if prev == "element":
                parts.append(_be32(1) + _be32(1) + _be32(1)
                             + _be32(b["interstitial"][0]) + _be32(b["interstitial"][1]))
            parts.append(_chunk(b"\x02\x03\x01", _satin_payload(b["left"], b["right"])))
        prev = b["kind"]
    preamble = _be32(0) + _be16(len(blocks))
    geometry = _chunk(b"\x01\x03\x03\x01", preamble + b"".join(parts))
    return _finalize(b"\x07\x05\x01", o6, o5, geometry)


# ---------------------------------------------------------------------------
# Design to bytes (sh7pad ``encodeSegments.ts``, ``designSource.ts``,
# ``multiBlockEmit.ts`` and ``satinShape.ts``)

def _segment_kinds(design: Sh7Design) -> List[Union[str, Satin]]:
    return list(design.segments) if design.segments is not None \
        else ["straight"] * (len(design.points) - 1)


def _cone(a: Tuple[float, float], b: Tuple[float, float], satin: Satin):
    """Left and right cone edges (top to bottom).  Left is the +perp side of
    the spine, so for a spine pointing +Y it is at smaller X."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1
    nx, ny = -dy / length, dx / length
    hs, he = satin.width_start / 2, satin.width_end / 2
    left = [(a[0] + nx * hs, a[1] + ny * hs), (b[0] + nx * he, b[1] + ny * he)]
    right = [(a[0] - nx * hs, a[1] - ny * hs), (b[0] - nx * he, b[1] - ny * he)]
    return left, right


def _check_design(design: Sh7Design) -> Foot:
    if design.foot not in FEET:
        raise Sh7Error(f"unknown foot {design.foot!r}; use one of {sorted(FEET)}")
    foot = FEET[design.foot]
    if len(design.points) < 2:
        raise Sh7Error("a design needs at least two points")
    if design.points[0][1] != 0:
        raise Sh7Error("points[0] is the Start Stitch and must have y = 0")
    kinds = _segment_kinds(design)
    if len(kinds) != len(design.points) - 1:
        raise Sh7Error("segments must have one entry per pair of points")
    for i, k in enumerate(kinds):
        if isinstance(k, Satin):
            a, b = design.points[i], design.points[i + 1]
            if tuple(a) == tuple(b):
                raise Sh7Error(f"segment {i}: a satin spine needs a length")
            for w in (k.width_start, k.width_end):
                if not SATIN_WIDTH_MIN_MM <= w <= SATIN_WIDTH_MAX_MM:
                    raise Sh7Error(f"segment {i}: satin width {w} mm is outside "
                                   f"{SATIN_WIDTH_MIN_MM}..{SATIN_WIDTH_MAX_MM} mm")
            if k.density <= 0:
                raise Sh7Error(f"segment {i}: satin density must be positive")
            if k.end_at not in ("left", "center", "right"):
                raise Sh7Error(f"segment {i}: end_at must be left, center or right")
        elif k != "straight":
            raise Sh7Error(f"segment {i}: unknown segment type {k!r}")
    xs = [p[0] for p in design.points]
    ys = [p[1] for p in design.points]
    if max(ys) - min(ys) > SH7_MAX_Y_MM:
        raise Sh7Error(f"design is {max(ys) - min(ys):.2f} mm long; .sh7 allows "
                       f"{SH7_MAX_Y_MM} mm")
    if max(xs) - min(xs) > foot.reach_half_mm * 2:
        raise Sh7Error(f"design is {max(xs) - min(xs):.2f} mm wide; foot {foot.id} "
                       f"reaches {foot.reach_half_mm * 2} mm")
    if abs(design.points[0][0] - design.start_x_mm) > foot.slot_half_mm + EPSILON:
        raise Sh7Error("the Start Stitch must sit inside the needle slot around the "
                       "Carriage Start")
    if design.start_x_mm > 0:
        # xElem = -start_x * 1000 is written as an unsigned BE32.
        raise Sh7Error("start_x_mm must be 0 or negative")
    return foot


def _straight_records(design: Sh7Design, foot: Foot):
    """Records of a satin-free design: (kind, dx_raw, dy_raw, x_mm, y_mm)."""
    pts = [(float(x), float(y)) for x, y in design.points]
    start_stitch_x = pts[0][0]
    deltas = [(_js_round((b[0] - a[0]) * X_UNITS_PER_MM),
               _js_round((b[1] - a[1]) * Y_UNITS_PER_MM))
              for a, b in zip(pts, pts[1:])]
    buckets = _plan(foot, deltas, design.uniform,
                    cursor_x=_js_round(start_stitch_x * X_UNITS_PER_MM),
                    carriage_x=_js_round(design.start_x_mm * X_UNITS_PER_MM))
    out = [("short", _js_round(start_stitch_x * X_UNITS_PER_MM), 0, start_stitch_x, 0.0)]
    for (a, b), bucket in zip(zip(pts, pts[1:]), buckets):
        if not bucket:
            continue
        last = bucket[-1]
        base_x = a[0] - (last.end_x_mm - (b[0] - a[0]))
        base_y = a[1] - (last.end_y_mm - (b[1] - a[1]))
        for r in bucket:
            out.append((r.kind, r.dx, r.dy, base_x + r.end_x_mm, base_y + r.end_y_mm))
    return out


def _interpolate_at_y(points, y: float) -> Tuple[float, float]:
    """Point on an edge polyline at height y; past either end, that endpoint."""
    for a, b in zip(points, points[1:]):
        if min(a[1], b[1]) <= y <= max(a[1], b[1]):
            t = 0 if b[1] - a[1] == 0 else (y - a[1]) / (b[1] - a[1])
            return (a[0] + (b[0] - a[0]) * t, y)
    return tuple(points[0]) if y <= points[0][1] else tuple(points[-1])


def _spine_x_at_y(left, right, y: float) -> float:
    return (_interpolate_at_y(left, y)[0] + _interpolate_at_y(right, y)[0]) / 2


def _satin_blocks(design: Sh7Design, foot: Foot):
    """Block list of a satin-bearing design: element stitch runs between cones.

    The machine sews each cone itself from its edges, so a cone contributes
    no needle records; the chain enters at the cone's top-left corner and
    leaves at its bottom-right (or the corner ``end_at`` asks for)."""
    pts = [(float(x), float(y)) for x, y in design.points]
    kinds = _segment_kinds(design)
    cones = {i: _cone(pts[i], pts[i + 1], k) for i, k in enumerate(kinds) if isinstance(k, Satin)}

    blocks, starts = [], []
    cur: list = []
    cur_start = None
    chain = pts[0]
    carriage = pts[0][0]

    def flush():
        nonlocal cur, cur_start
        if cur:
            blocks.append({"kind": "element", "records": cur})
            starts.append(cur_start or chain)
        cur, cur_start = [], None

    def move_to(target, seg_idx):
        nonlocal chain, carriage, cur_start
        if cur_start is None:
            cur_start = chain
        dx = _js_round((target[0] - chain[0]) * X_UNITS_PER_MM)
        dy = _js_round((target[1] - chain[1]) * Y_UNITS_PER_MM)
        if dx == 0 and dy == 0:
            chain = target
            return 0
        try:
            [bucket] = _plan(foot, [(dx, dy)], design.uniform,
                             cursor_x=_js_round(chain[0] * X_UNITS_PER_MM),
                             carriage_x=_js_round(carriage * X_UNITS_PER_MM))
        except Sh7Error:
            raise Sh7Error(f"segment {seg_idx} cannot be sewn with foot {foot.id}: the "
                           f"carriage would pass its reach of {foot.reach_half_mm} mm") from None
        for r in bucket:
            cur.append((r.kind, r.dx, r.dy))
            carriage = r.carriage_x_mm
        chain = target
        return len(bucket)

    for i, k in enumerate(kinds):
        if not isinstance(k, Satin):
            move_to(pts[i + 1], i)
            continue
        left, right = cones[i]
        tl, br = left[0], right[-1]
        if move_to(tl, i):
            carriage = _spine_x_at_y(left, right, tl[1])
        flush()
        blocks.append({"kind": "satin", "left": left, "right": right})
        starts.append(None)
        chain = br
        drops = _satin_drops(left, right, k.density)
        prev = tl
        for d in drops:
            # sh7pad refuses a cone whose zigzag steps more than 4 mm in Y.
            if abs(_js_round((d[1] - prev[1]) * Y_UNITS_PER_MM)) > STITCH_DY_MAX_RAW:
                raise Sh7Error(f"segment {i}: the satin zigzag steps more than 4 mm "
                               "along the stitch; make the cone narrower or less slanted")
            prev = d
        carriage = _spine_x_at_y(left, right, drops[-1][1])
        trailer = None
        if k.end_at == "left":
            trailer = left[-1]
        elif k.end_at == "center":
            trailer = ((left[-1][0] + right[-1][0]) / 2, (left[-1][1] + right[-1][1]) / 2)
        if trailer is not None:
            if move_to(trailer, i):
                carriage = _spine_x_at_y(left, right, trailer[1])
    flush()
    if blocks and blocks[-1]["kind"] == "satin":
        blocks.append({"kind": "element", "records": [("short", 0, 0)]})
        starts.append(chain)

    # Offsets: element starts and cone positions relative to the design bbox.
    bbox_pts = list(pts) + [p for left, right in cones.values() for p in left + right]
    min_x = min(p[0] for p in bbox_pts)
    min_y = min(p[1] for p in bbox_pts)
    start_dx = _js_round(pts[0][0] * X_UNITS_PER_MM)
    injected = pts[0][0] == 0
    running_y = pts[0][1]
    out = []
    for b, start in zip(blocks, starts):
        if b["kind"] == "element":
            start = start or pts[0]
            records = b["records"]
            if not injected:
                records = [("short", start_dx, 0)] + records
                injected = True
            out.append({"kind": "element", "records": records,
                        "x_elem": _js_round((start[0] - min_x) * 1000),
                        "y_pos": _js_round((start[1] - min_y) * 1500)})
        else:
            cone_min_x = min(p[0] for p in b["left"] + b["right"])
            out.append({**b, "interstitial": (_js_round((cone_min_x - min_x) * 1000),
                                              _js_round((running_y - min_y) * 1500))})
            running_y = b["right"][-1][1]
    max_x = max(p[0] for p in bbox_pts)
    max_y = max(p[1] for p in bbox_pts)
    size = (_js_round((max_x - min_x) * 1000), _js_round((max_y - min_y) * 1000))
    return out, size


def _satin_drops(left, right, density: float) -> List[Tuple[float, float]]:
    """The zigzag drops the machine sews inside a cone, as sh7pad's
    ``satinStitches`` computes them.  The step count is odd, so the last drop
    is on the right edge.  Heights come from the left edge's span."""
    y_start, y_end = left[0][1], left[-1][1]
    steps = max(1, _js_round(abs(y_end - y_start) / density))
    if steps % 2 == 0:
        steps += 1
    drops = []
    for i in range(1, steps + 1):
        y = y_start + ((y_end - y_start) * i) / steps
        drops.append(_interpolate_at_y(right if i % 2 == 1 else left, y))
    return drops


def sew_records(design: Sh7Design) -> List[Tuple[str, int, int, float, float]]:
    """The machine records of a satin-free design: (kind, dx_raw, dy_raw, x_mm, y_mm).

    The first record is the Start Stitch.  ``x_mm`` and ``y_mm`` are where the
    needle lands after the record, in design coordinates.
    """
    foot = _check_design(design)
    if any(isinstance(k, Satin) for k in _segment_kinds(design)):
        raise Sh7Error("sew_records covers straight segments only")
    return _straight_records(design, foot)


def _check_records(records) -> None:
    for n, (kind, _dx, dy, *_rest) in enumerate(records):
        if abs(dy) > STITCH_DY_MAX_RAW:
            raise Sh7Error(f"{kind} record {n} moves {dy / Y_UNITS_PER_MM:.2f} mm along the "
                           "stitch; the machine allows 4 mm per record")


def write_sh7(design: Sh7Design) -> bytes:
    """Encode a design as the bytes of a .sh7 file."""
    foot = _check_design(design)
    tension_byte = _js_round(design.tension * 10)
    if any(isinstance(k, Satin) for k in _segment_kinds(design)):
        blocks, (x_um, y_um) = _satin_blocks(design, foot)
        for b in blocks:
            if b["kind"] == "element":
                _check_records(b["records"])
        return _multi_file(foot.byte, tension_byte, x_um, y_um, blocks)
    records = _straight_records(design, foot)
    _check_records(records)
    # The displayed size is the bbox of every needle position, as sh7pad
    # computes it for satin-free designs.
    xs = [r[3] for r in records]
    ys = [r[4] for r in records]
    x_um = _js_round((max(xs) - min(xs)) * 1000)
    y_um = _js_round((max(ys) - min(ys)) * 1000)
    x_elem = _js_round(-design.start_x_mm * 1000)
    return _singleton_file(foot.byte, tension_byte, x_um, y_um, x_elem,
                           [(k, dx, dy) for k, dx, dy, _, _ in records])
