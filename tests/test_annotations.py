"""Tests du parsing, de la validation et de la conversion des annotations."""

from __future__ import annotations

import math

import pytest

from ppe_detection.annotations import (
    BBox,
    IssueCode,
    LineKind,
    format_detection_line,
    normalise_line,
    parse_label_text,
    parse_line,
    polygon_to_bbox,
)

NUM_CLASSES = 7


# --------------------------------------------------------------------------- #
# Classification structurelle
# --------------------------------------------------------------------------- #
def test_five_fields_is_detection() -> None:
    parsed = parse_line("1 0.5 0.5 0.2 0.4")
    assert parsed.kind is LineKind.DETECTION
    assert parsed.class_id == 1
    assert parsed.box == BBox(0.5, 0.5, 0.2, 0.4)
    assert not parsed.issues


def test_odd_field_count_at_least_seven_is_polygon() -> None:
    parsed = parse_line("4 0.1 0.1 0.3 0.1 0.3 0.3")
    assert parsed.kind is LineKind.POLYGON
    assert parsed.n_points == 3
    assert parsed.class_id == 4


@pytest.mark.parametrize("raw", ["", "   ", "\t"])
def test_blank_line_is_flagged_empty(raw: str) -> None:
    parsed = parse_line(raw)
    assert IssueCode.EMPTY_LINE in parsed.issues


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1 0.5 0.5", IssueCode.BAD_FIELD_COUNT),                    # 3 champs
        ("1 0.5 0.5 0.2", IssueCode.BAD_FIELD_COUNT),                # 4 champs
        ("1 0.1 0.1 0.2 0.2 0.3", IssueCode.BAD_FIELD_COUNT),        # 6 champs
        # 8 champs = 1 identifiant + 7 coordonnees : nombre impair de
        # coordonnees, donc pas de decoupage possible en paires (x, y).
        ("1 0.1 0.1 0.2 0.2 0.3 0.3 0.4", IssueCode.POLYGON_ODD_COORDS),
    ],
)
def test_unusable_field_counts(raw: str, expected: IssueCode) -> None:
    parsed = parse_line(raw)
    assert parsed.kind is LineKind.MALFORMED
    assert expected in parsed.issues


def test_seven_fields_is_a_valid_triangle_polygon() -> None:
    """7 champs = 1 identifiant + 3 points : c'est un polygone valide."""
    parsed = parse_line("1 0.1 0.1 0.2 0.2 0.3 0.3")
    assert parsed.kind is LineKind.POLYGON
    assert parsed.n_points == 3
    assert not parsed.issues


def test_non_numeric_field_is_detected() -> None:
    parsed = parse_line("2 abc 0.5 0.1 0.1")
    assert IssueCode.NON_NUMERIC in parsed.issues


def test_non_integer_class_id_is_rejected() -> None:
    parsed = parse_line("1.7 0.5 0.5 0.2 0.2")
    assert IssueCode.CLASS_ID_NOT_INTEGER in parsed.issues


def test_integral_float_class_id_is_accepted() -> None:
    parsed = parse_line("3.0 0.5 0.5 0.2 0.2")
    assert parsed.class_id == 3
    assert not parsed.issues


@pytest.mark.parametrize("token", ["nan", "inf", "-inf"])
def test_non_finite_values_are_rejected(token: str) -> None:
    parsed = parse_line(f"1 {token} 0.5 0.2 0.2")
    assert IssueCode.NON_FINITE in parsed.issues


def test_polygon_with_two_points_is_rejected() -> None:
    # 5 champs seraient interpretes comme une detection : on teste 1 + 2 points
    # sur une ligne a 5 champs -> detection. On force donc via parse direct.
    parsed = parse_line("4 0.1 0.1 0.3 0.1 0.3")  # 6 champs -> malforme
    assert parsed.kind is LineKind.MALFORMED


# --------------------------------------------------------------------------- #
# Conversion polygone -> boite
# --------------------------------------------------------------------------- #
def test_polygon_to_bbox_uses_min_max() -> None:
    box = polygon_to_bbox([0.10, 0.20, 0.30, 0.25, 0.20, 0.50])
    assert box.xyxy == pytest.approx((0.10, 0.20, 0.30, 0.50))
    assert box.cx == pytest.approx(0.20)
    assert box.cy == pytest.approx(0.35)
    assert box.w == pytest.approx(0.20)
    assert box.h == pytest.approx(0.30)


def test_polygon_to_bbox_rejects_odd_length() -> None:
    with pytest.raises(ValueError, match="even"):
        polygon_to_bbox([0.1, 0.2, 0.3])


def test_normalise_marks_polygon_conversion() -> None:
    parsed = normalise_line(
        "4 0.10 0.10 0.30 0.12 0.32 0.28 0.09 0.26", num_classes=NUM_CLASSES
    )
    assert not parsed.is_error
    assert IssueCode.CONVERTED_FROM_POLYGON in parsed.issues
    assert parsed.box is not None
    assert parsed.box.xyxy == pytest.approx((0.09, 0.10, 0.32, 0.28))


def test_converted_polygon_round_trips_to_five_fields() -> None:
    parsed = normalise_line("0 0.2 0.2 0.6 0.2 0.6 0.8 0.2 0.8", num_classes=NUM_CLASSES)
    assert parsed.box is not None
    line = format_detection_line(parsed.class_id, parsed.box)
    assert len(line.split()) == 5
    reparsed = parse_line(line)
    assert reparsed.kind is LineKind.DETECTION


# --------------------------------------------------------------------------- #
# Validation et correction
# --------------------------------------------------------------------------- #
def test_class_id_out_of_range_is_error() -> None:
    parsed = normalise_line("99 0.5 0.5 0.2 0.2", num_classes=NUM_CLASSES)
    assert parsed.is_error
    assert IssueCode.CLASS_ID_OUT_OF_RANGE in parsed.issues


def test_minor_drift_is_clamped_silently() -> None:
    # x1 = 0.5 - 0.5/2 = 0.25 ; on force un depassement infime en haut
    parsed = normalise_line("1 0.5 0.5 1.0005 0.2", num_classes=NUM_CLASSES)
    assert not parsed.is_error
    assert IssueCode.CLAMPED_MINOR in parsed.issues
    assert parsed.box is not None
    x1, y1, x2, y2 = parsed.box.xyxy
    assert 0.0 <= x1 <= x2 <= 1.0
    assert 0.0 <= y1 <= y2 <= 1.0


def test_box_crossing_border_is_clipped_not_dropped() -> None:
    # Centre a 0.95, largeur 0.2 -> x2 = 1.05, nettement au-dela de la tolerance
    parsed = normalise_line("5 0.95 0.5 0.2 0.3", num_classes=NUM_CLASSES)
    assert not parsed.is_error
    assert IssueCode.CLIPPED_TO_BOUNDS in parsed.issues
    assert parsed.box is not None
    assert parsed.box.xyxy[2] == pytest.approx(1.0)
    assert parsed.box.xyxy[0] == pytest.approx(0.85)


def test_center_outside_image_is_error() -> None:
    parsed = normalise_line("1 1.4 0.5 0.2 0.2", num_classes=NUM_CLASSES)
    assert parsed.is_error
    assert IssueCode.CENTER_OUT_OF_RANGE in parsed.issues


@pytest.mark.parametrize("raw", ["1 0.5 0.5 0.0 0.2", "1 0.5 0.5 0.2 -0.1"])
def test_non_positive_size_is_error(raw: str) -> None:
    parsed = normalise_line(raw, num_classes=NUM_CLASSES)
    assert parsed.is_error
    assert IssueCode.NON_POSITIVE_SIZE in parsed.issues


def test_valid_box_produces_no_issue() -> None:
    parsed = normalise_line("1 0.5 0.5 0.2 0.4", num_classes=NUM_CLASSES)
    assert not parsed.issues
    assert not parsed.is_error


def test_bbox_geometry_helpers() -> None:
    box = BBox(0.5, 0.5, 0.2, 0.4)
    assert box.xyxy == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert box.area == pytest.approx(0.08)
    rebuilt = BBox.from_xyxy(*box.xyxy)
    assert rebuilt.cx == pytest.approx(box.cx)
    assert rebuilt.h == pytest.approx(box.h)


# --------------------------------------------------------------------------- #
# Fichiers complets
# --------------------------------------------------------------------------- #
def test_parse_label_text_skips_blank_lines_and_numbers_correctly() -> None:
    text = "1 0.5 0.5 0.2 0.2\n\n  \n4 0.1 0.1 0.3 0.1 0.3 0.3\n"
    results = parse_label_text(text, num_classes=NUM_CLASSES)
    assert [line_no for line_no, _ in results] == [1, 4]
    assert results[1][1].kind is LineKind.POLYGON


def test_parse_label_text_handles_crlf() -> None:
    results = parse_label_text("1 0.5 0.5 0.2 0.2\r\n2 0.3 0.3 0.1 0.1\r\n", num_classes=NUM_CLASSES)
    assert len(results) == 2
    assert all(not parsed.is_error for _, parsed in results)


def test_format_detection_line_precision() -> None:
    line = format_detection_line(4, BBox(1 / 3, 2 / 3, 0.1, 0.2), precision=6)
    parts = line.split()
    assert parts[0] == "4"
    assert len(parts) == 5
    assert math.isclose(float(parts[1]), 1 / 3, abs_tol=1e-6)
