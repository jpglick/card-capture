import numpy as np

from card_capture.stages.refine.cropper import CardCropper, _orient_for_target_canvas, order_points_clockwise


def test_order_points_clockwise_returns_top_left_first():
    points = ((100.0, 200.0), (10.0, 20.0), (105.0, 25.0), (15.0, 210.0))

    ordered = order_points_clockwise(points)

    assert ordered[0] == (10.0, 20.0)
    assert ordered[1] == (105.0, 25.0)
    assert ordered[2] == (100.0, 200.0)
    assert ordered[3] == (15.0, 210.0)


def test_cropper_returns_deskewed_crop_with_expected_dimensions():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[20:100, 40:120] = 255
    polygon = ((40.0, 20.0), (120.0, 20.0), (120.0, 100.0), (40.0, 100.0))

    crop = CardCropper().crop(image, polygon)

    assert crop.image.shape == (80, 80, 3)
    assert crop.width == 80
    assert crop.height == 80
    assert int(crop.image.mean()) == 255


def test_orient_for_target_canvas_rotates_landscape_to_portrait_mapping():
    ordered = (
        (20.0, 40.0),   # TL
        (180.0, 40.0),  # TR
        (180.0, 100.0), # BR
        (20.0, 100.0),  # BL
    )
    oriented = _orient_for_target_canvas(ordered, target_width=750, target_height=1050)

    pts = np.array(oriented, dtype=np.float32)
    top = np.linalg.norm(pts[1] - pts[0])
    right = np.linalg.norm(pts[2] - pts[1])
    assert top < right
