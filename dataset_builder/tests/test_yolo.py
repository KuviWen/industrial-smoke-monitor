from smoke_dataset_builder.yolo import (
    flip_polygons_horizontally,
    polygon_to_yolo_line,
    validate_yolo_line,
    validate_yolo_lines,
)


def test_polygon_is_normalized_for_yolo_segmentation() -> None:
    line = polygon_to_yolo_line([(0, 0), (50, 0), (50, 100)], 100, 200)
    assert line == "0 0.000000 0.000000 0.500000 0.000000 0.500000 0.500000"


def test_invalid_polygon_is_ignored() -> None:
    assert polygon_to_yolo_line([(1, 1), (1, 1), (1, 1)], 100, 100) is None


def test_label_validator_accepts_empty_negative_file_and_valid_line() -> None:
    assert validate_yolo_lines([], 1) == []
    assert validate_yolo_line("0 0.1 0.1 0.2 0.1 0.2 0.2", 1) is None


def test_label_validator_rejects_out_of_range_coordinate() -> None:
    assert validate_yolo_line("0 1.1 0.1 0.2 0.1 0.2 0.2", 1) is not None


def test_horizontal_flip_mirrors_x_and_preserves_y() -> None:
    polygons = [[(0, 10), (9, 10), (4, 20)]]
    assert flip_polygons_horizontally(polygons, width=10) == [
        [(9.0, 10.0), (0.0, 10.0), (5.0, 20.0)]
    ]
