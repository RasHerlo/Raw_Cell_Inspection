import numpy as np

from raw_cell_inspection.analysis import frames_in_ranges, range_ratio_heatmap, stack_summary
from raw_cell_inspection.masks import polygon_mask, roi_trace, simplify_stroke
from raw_cell_inspection.store import load_document, new_document, save_document, signature_mismatch


def test_square_mask_covers_pixel_centres():
    square = np.array([[2, 3], [6, 3], [6, 5], [2, 5]], dtype=float)
    info = polygon_mask(square, 10, 10)
    assert info["bbox"] == (3, 5, 2, 6)
    assert info["mask"].all() and info["mask"].shape == (2, 4)


def test_tiny_polygon_is_empty():
    tri = np.array([[1.1, 1.1], [1.3, 1.1], [1.2, 1.3]])
    assert polygon_mask(tri, 10, 10) is None


def test_roi_trace_is_mean_inside_mask():
    stack = np.zeros((3, 8, 8), dtype=np.uint16)
    stack[:, 2:4, 2:4] = np.array([1, 2, 3])[:, None, None]
    info = polygon_mask(np.array([[2, 2], [4, 2], [4, 4], [2, 4]], dtype=float), 8, 8)
    np.testing.assert_allclose(roi_trace(stack, info), [1, 2, 3])


def test_simplify_keeps_shape_and_caps_vertices():
    theta = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    circle = np.c_[50 + 20 * np.cos(theta), 50 + 20 * np.sin(theta)]
    simple = simplify_stroke(circle)
    assert 8 <= len(simple) <= 60


def test_ratio_heatmap():
    stack = np.full((10, 4, 4), 100, dtype=np.uint16)
    stack[2:5, 0, 0] = 300
    ratio = range_ratio_heatmap(stack, [(2, 4)])
    assert np.isclose(ratio[0, 0], 3.0)
    assert np.isclose(ratio[1, 1], 1.0)
    assert frames_in_ranges(10, [(8, 20)]).sum() == 2


def test_summary():
    stack = np.arange(4 * 2 * 2, dtype=np.uint16).reshape(4, 2, 2)
    s = stack_summary(stack)
    np.testing.assert_allclose(s["mean_image"], stack.mean(0))
    np.testing.assert_allclose(s["fov_mean_trace"], stack.reshape(4, -1).mean(1))


def test_document_roundtrip_and_backup(tmp_path):
    sig = {"filename": "a.tif", "shape": (5, 4, 4), "dtype": "uint16", "file_size": 10}
    doc = new_document(sig)
    path = tmp_path / "a_rci.pkl"
    save_document(path, doc)
    save_document(path, doc)
    assert (tmp_path / "a_rci.pkl.bak").exists()
    loaded = load_document(path)
    assert loaded["stack"]["shape"] == (5, 4, 4)
    assert signature_mismatch(loaded["stack"], sig) == []
    assert signature_mismatch(loaded["stack"], {**sig, "shape": (6, 4, 4)})
