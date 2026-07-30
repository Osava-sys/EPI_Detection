"""Parsing, validation and normalisation of YOLO annotation lines.

This module is deliberately dependency-free (standard library only) so that the
annotation logic — the most correctness-critical part of the project — can be
unit tested without PyTorch, OpenCV or Ultralytics being installed.

Two line shapes coexist in the source Roboflow export:

* **detection** — exactly 5 fields: ``class_id cx cy w h`` (normalised).
* **polygon** — ``class_id x1 y1 x2 y2 ...`` with an even number of coordinates
  and at least 3 points, i.e. an odd total field count >= 7.

A polygon line is *not* an error: it is a segmentation annotation that must be
converted to its axis-aligned bounding box for an object-detection task.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "BBox",
    "IssueCode",
    "LineKind",
    "ParsedLine",
    "DEFAULT_CLAMP_TOLERANCE",
    "DEFAULT_MIN_BOX_SIDE",
    "parse_line",
    "parse_label_text",
    "polygon_to_bbox",
    "normalise_line",
    "format_detection_line",
]

# Coordinates further outside [0, 1] than this are considered a real annotation
# error rather than floating-point drift from the exporter.
DEFAULT_CLAMP_TOLERANCE: float = 1e-3

# Normalised side length below which a box is flagged as degenerate/too small.
DEFAULT_MIN_BOX_SIDE: float = 1e-4


class LineKind(str, Enum):
    """Shape of a raw annotation line."""

    DETECTION = "detection"
    POLYGON = "polygon"
    MALFORMED = "malformed"


class IssueCode(str, Enum):
    """Machine-readable issue identifiers attached to a parsed line.

    Codes are split between *warnings* (recoverable, the box is kept after a
    documented correction) and *errors* (the line cannot yield a trustworthy
    box and is dropped by the cleaner).
    """

    # --- errors -----------------------------------------------------------
    EMPTY_LINE = "empty_line"
    NON_NUMERIC = "non_numeric"
    CLASS_ID_NOT_INTEGER = "class_id_not_integer"
    CLASS_ID_OUT_OF_RANGE = "class_id_out_of_range"
    BAD_FIELD_COUNT = "bad_field_count"
    POLYGON_ODD_COORDS = "polygon_odd_coords"
    POLYGON_TOO_FEW_POINTS = "polygon_too_few_points"
    NON_FINITE = "non_finite"
    CENTER_OUT_OF_RANGE = "center_out_of_range"
    NON_POSITIVE_SIZE = "non_positive_size"
    SIZE_OUT_OF_RANGE = "size_out_of_range"
    DEGENERATE_AFTER_CLIP = "degenerate_after_clip"

    # --- warnings ---------------------------------------------------------
    CLAMPED_MINOR = "clamped_minor"
    CLIPPED_TO_BOUNDS = "clipped_to_bounds"
    TINY_BOX = "tiny_box"
    CONVERTED_FROM_POLYGON = "converted_from_polygon"


ERROR_CODES: frozenset[IssueCode] = frozenset(
    {
        IssueCode.EMPTY_LINE,
        IssueCode.NON_NUMERIC,
        IssueCode.CLASS_ID_NOT_INTEGER,
        IssueCode.CLASS_ID_OUT_OF_RANGE,
        IssueCode.BAD_FIELD_COUNT,
        IssueCode.POLYGON_ODD_COORDS,
        IssueCode.POLYGON_TOO_FEW_POINTS,
        IssueCode.NON_FINITE,
        IssueCode.CENTER_OUT_OF_RANGE,
        IssueCode.NON_POSITIVE_SIZE,
        IssueCode.SIZE_OUT_OF_RANGE,
        IssueCode.DEGENERATE_AFTER_CLIP,
    }
)


@dataclass(frozen=True)
class BBox:
    """A normalised YOLO detection box (centre form), values nominally in [0, 1]."""

    cx: float
    cy: float
    w: float
    h: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Return the box as ``(x1, y1, x2, y2)`` corners."""
        return (
            self.cx - self.w / 2.0,
            self.cy - self.h / 2.0,
            self.cx + self.w / 2.0,
            self.cy + self.h / 2.0,
        )

    @property
    def area(self) -> float:
        """Normalised area of the box."""
        return max(self.w, 0.0) * max(self.h, 0.0)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> BBox:
        """Build a centre-form box from corner coordinates."""
        return cls(cx=(x1 + x2) / 2.0, cy=(y1 + y2) / 2.0, w=x2 - x1, h=y2 - y1)


@dataclass
class ParsedLine:
    """Result of parsing (and optionally normalising) a single annotation line."""

    kind: LineKind
    class_id: int | None = None
    box: BBox | None = None
    n_points: int = 0
    n_fields: int = 0
    issues: list[IssueCode] = field(default_factory=list)
    detail: str = ""

    @property
    def is_error(self) -> bool:
        """True when at least one issue prevents emitting a trustworthy box."""
        return any(code in ERROR_CODES for code in self.issues)

    @property
    def warnings(self) -> list[IssueCode]:
        """Issues that were corrected rather than fatal."""
        return [code for code in self.issues if code not in ERROR_CODES]

    @property
    def errors(self) -> list[IssueCode]:
        """Issues that make the line unusable."""
        return [code for code in self.issues if code in ERROR_CODES]


def polygon_to_bbox(coords: Sequence[float]) -> BBox:
    """Convert a flat ``[x1, y1, x2, y2, ...]`` polygon to its bounding box.

    Args:
        coords: Flat sequence of normalised coordinates, even length, >= 6 items.

    Returns:
        The axis-aligned bounding box enclosing every polygon vertex.

    Raises:
        ValueError: If the coordinate list is empty or has an odd length.
    """
    if not coords or len(coords) % 2 != 0:
        raise ValueError(f"polygon needs an even, non-empty coordinate count, got {len(coords)}")
    xs = coords[0::2]
    ys = coords[1::2]
    return BBox.from_xyxy(min(xs), min(ys), max(xs), max(ys))


def _classify(n_fields: int) -> LineKind:
    """Decide the shape of a line from its field count alone."""
    if n_fields == 5:
        return LineKind.DETECTION
    if n_fields >= 7 and n_fields % 2 == 1:
        return LineKind.POLYGON
    return LineKind.MALFORMED


def parse_line(raw: str) -> ParsedLine:
    """Parse one raw annotation line into its structural form.

    No range validation is performed here beyond what is needed to build the
    geometry; use :func:`normalise_line` for the full validate-and-correct pass.

    Args:
        raw: The raw text of a single line.

    Returns:
        A :class:`ParsedLine`. Structural problems are reported through
        ``issues`` rather than raised.
    """
    stripped = raw.strip()
    if not stripped:
        return ParsedLine(kind=LineKind.MALFORMED, issues=[IssueCode.EMPTY_LINE])

    parts = stripped.split()
    n_fields = len(parts)
    kind = _classify(n_fields)

    if kind is LineKind.MALFORMED:
        return ParsedLine(
            kind=kind,
            n_fields=n_fields,
            issues=[_field_count_issue(n_fields)],
            detail=f"{n_fields} fields",
        )

    try:
        values = [float(token) for token in parts]
    except ValueError:
        return ParsedLine(
            kind=kind,
            n_fields=n_fields,
            issues=[IssueCode.NON_NUMERIC],
            detail=stripped[:120],
        )

    if any(not math.isfinite(value) for value in values):
        return ParsedLine(
            kind=kind,
            n_fields=n_fields,
            issues=[IssueCode.NON_FINITE],
            detail=stripped[:120],
        )

    class_raw = values[0]
    if not float(class_raw).is_integer():
        return ParsedLine(
            kind=kind,
            n_fields=n_fields,
            issues=[IssueCode.CLASS_ID_NOT_INTEGER],
            detail=f"class_id={class_raw!r}",
        )
    class_id = int(class_raw)

    coords = values[1:]
    if kind is LineKind.DETECTION:
        box = BBox(cx=coords[0], cy=coords[1], w=coords[2], h=coords[3])
        return ParsedLine(kind=kind, class_id=class_id, box=box, n_fields=n_fields)

    # Polygon
    n_points = len(coords) // 2
    if n_points < 3:
        return ParsedLine(
            kind=kind,
            class_id=class_id,
            n_fields=n_fields,
            n_points=n_points,
            issues=[IssueCode.POLYGON_TOO_FEW_POINTS],
            detail=f"{n_points} points",
        )
    return ParsedLine(
        kind=kind,
        class_id=class_id,
        box=polygon_to_bbox(coords),
        n_points=n_points,
        n_fields=n_fields,
    )


def _field_count_issue(n_fields: int) -> IssueCode:
    """Pick the most descriptive issue code for an unusable field count."""
    if n_fields >= 7 and n_fields % 2 == 0:
        return IssueCode.POLYGON_ODD_COORDS
    return IssueCode.BAD_FIELD_COUNT


def _clamp_unit(value: float, tolerance: float) -> tuple[float, bool, bool]:
    """Clamp ``value`` into [0, 1].

    Returns:
        ``(clamped_value, was_minor_drift, was_major_excursion)``.
    """
    if 0.0 <= value <= 1.0:
        return value, False, False
    excess = -value if value < 0.0 else value - 1.0
    clamped = 0.0 if value < 0.0 else 1.0
    if excess <= tolerance:
        return clamped, True, False
    return clamped, False, True


def normalise_line(
    raw: str,
    *,
    num_classes: int,
    clamp_tolerance: float = DEFAULT_CLAMP_TOLERANCE,
    min_box_side: float = DEFAULT_MIN_BOX_SIDE,
) -> ParsedLine:
    """Parse, validate and correct one annotation line for a detection dataset.

    Correction policy, in order of severity:

    1. Coordinates outside [0, 1] by at most ``clamp_tolerance`` are silently
       clamped (exporter floating-point drift) and flagged ``clamped_minor``.
    2. A box whose *corners* fall outside the image while its centre and size
       remain valid is clipped to the image bounds and flagged
       ``clipped_to_bounds`` — this is the correct handling of an object that
       is only partially visible.
    3. Anything else (non-numeric fields, out-of-range class id, non-positive
       size, centre outside the image by more than the tolerance) is reported
       as an error; the caller is expected to drop the line rather than invent
       a plausible box.

    Args:
        raw: Raw line text.
        num_classes: Number of classes declared in ``data.yaml``.
        clamp_tolerance: Maximum excursion treated as floating-point drift.
        min_box_side: Normalised side below which a box is flagged as tiny.

    Returns:
        A :class:`ParsedLine` whose ``box`` is a valid detection box when
        ``is_error`` is False.
    """
    parsed = parse_line(raw)
    if parsed.kind is LineKind.MALFORMED or parsed.box is None:
        return parsed

    assert parsed.class_id is not None  # guaranteed once box is set
    if not 0 <= parsed.class_id < num_classes:
        parsed.issues.append(IssueCode.CLASS_ID_OUT_OF_RANGE)
        parsed.detail = f"class_id={parsed.class_id} not in [0, {num_classes - 1}]"
        return parsed

    if parsed.kind is LineKind.POLYGON:
        parsed.issues.append(IssueCode.CONVERTED_FROM_POLYGON)

    box = parsed.box
    x1, y1, x2, y2 = box.xyxy

    # Step 1: clamp tiny drift on the corners; record major excursions.
    minor = False
    major = False
    clipped: list[float] = []
    for value in (x1, y1, x2, y2):
        new_value, was_minor, was_major = _clamp_unit(value, clamp_tolerance)
        clipped.append(new_value)
        minor = minor or was_minor
        major = major or was_major
    nx1, ny1, nx2, ny2 = clipped

    # The centre is checked first: a box lying entirely outside the frame has
    # both its edges collapse onto the same bound, which would surface only as a
    # generic "degenerate box". Reporting the centre is far more actionable for
    # whoever has to fix the annotation.
    centre_ok = (
        -clamp_tolerance <= box.cx <= 1.0 + clamp_tolerance
        and -clamp_tolerance <= box.cy <= 1.0 + clamp_tolerance
    )
    if not centre_ok:
        parsed.issues.append(IssueCode.CENTER_OUT_OF_RANGE)
        parsed.detail = f"centre outside image: cx={box.cx:.6f} cy={box.cy:.6f}"
        return parsed

    if nx2 <= nx1 or ny2 <= ny1:
        parsed.issues.append(IssueCode.NON_POSITIVE_SIZE)
        parsed.detail = f"degenerate box cx={box.cx:.6f} cy={box.cy:.6f} w={box.w:.6f} h={box.h:.6f}"
        return parsed

    if minor:
        parsed.issues.append(IssueCode.CLAMPED_MINOR)
    if major:
        # The box genuinely extended past the image border. Clipping is the
        # standard, documented handling for partially visible objects.
        if box.w <= 0.0 or box.h <= 0.0:
            parsed.issues.append(IssueCode.NON_POSITIVE_SIZE)
            parsed.detail = f"w={box.w:.6f} h={box.h:.6f}"
            return parsed
        if box.w > 1.0 + clamp_tolerance or box.h > 1.0 + clamp_tolerance:
            parsed.issues.append(IssueCode.SIZE_OUT_OF_RANGE)
            parsed.detail = f"w={box.w:.6f} h={box.h:.6f}"
            return parsed
        parsed.issues.append(IssueCode.CLIPPED_TO_BOUNDS)

    new_box = BBox.from_xyxy(nx1, ny1, nx2, ny2)
    if new_box.w < min_box_side or new_box.h < min_box_side:
        parsed.issues.append(IssueCode.DEGENERATE_AFTER_CLIP)
        parsed.detail = f"w={new_box.w:.8f} h={new_box.h:.8f}"
        return parsed

    if new_box.w * new_box.h < min_box_side:
        parsed.issues.append(IssueCode.TINY_BOX)

    parsed.box = new_box
    return parsed


def parse_label_text(
    text: str,
    *,
    num_classes: int,
    clamp_tolerance: float = DEFAULT_CLAMP_TOLERANCE,
    min_box_side: float = DEFAULT_MIN_BOX_SIDE,
) -> list[tuple[int, ParsedLine]]:
    """Normalise every non-blank line of a label file.

    Args:
        text: Full text content of a ``.txt`` label file.
        num_classes: Number of classes declared in ``data.yaml``.
        clamp_tolerance: Maximum excursion treated as floating-point drift.
        min_box_side: Normalised side below which a box is flagged as tiny.

    Returns:
        A list of ``(line_number, parsed_line)`` pairs, 1-indexed, skipping
        blank lines.
    """
    results: list[tuple[int, ParsedLine]] = []
    for index, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        results.append(
            (
                index,
                normalise_line(
                    raw,
                    num_classes=num_classes,
                    clamp_tolerance=clamp_tolerance,
                    min_box_side=min_box_side,
                ),
            )
        )
    return results


def format_detection_line(class_id: int, box: BBox, *, precision: int = 6) -> str:
    """Render a class id and box as a 5-field YOLO detection line."""
    return (
        f"{class_id} "
        f"{box.cx:.{precision}f} {box.cy:.{precision}f} "
        f"{box.w:.{precision}f} {box.h:.{precision}f}"
    )


def summarise_issues(lines: Iterable[ParsedLine]) -> dict[str, int]:
    """Count issue codes across many parsed lines."""
    counts: dict[str, int] = {}
    for line in lines:
        for code in line.issues:
            counts[code.value] = counts.get(code.value, 0) + 1
    return counts
