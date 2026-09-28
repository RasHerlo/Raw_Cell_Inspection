import pickle

import numpy as np
import pytest

from raw_cell_inspection.store import load_document, new_document
from raw_cell_inspection.trace_processing import (
    METRIC_EUCLIDEAN,
    METRIC_RUZICKA,
    ORDER_POOLED,
    ORDER_SEPARATE,
    compare_trace_groups,
    epoch_x,
    dendrogram_branches,
    event_onsets,
    hac_sort_key,
    hierarchical_view,
    recut_view,
    leaf_order,
    mean_and_sem,
    minmax_trace,
    ordered_rois,
    pairwise_pearson,
    similarity_conclusion,
    trial_zscores,
    upper_mean,
)


def test_minmax_and_pearson():
    assert np.allclose(minmax_trace([2, 4, 6]), [0, 0.5, 1])
    assert np.allclose(minmax_trace([3, 3, 3]), 0)
    same = np.vstack([np.arange(6), np.arange(6)])
    corr = pairwise_pearson(same)
    assert corr[0, 1] == pytest.approx(1)
    mean, n_pairs = upper_mean(corr)
    assert mean == pytest.approx(1) and n_pairs == 1
    flat = pairwise_pearson(np.vstack([np.arange(6), np.ones(6)]))
    assert np.isnan(flat[0, 1])


def test_leaf_order_keeps_similar_traces_together():
    similarity = np.array(
        [
            [1.0, 0.95, 0.0],
            [0.95, 1.0, 0.05],
            [0.0, 0.05, 1.0],
        ]
    )
    order = leaf_order(similarity)
    assert set(order) == {0, 1, 2}
    assert abs(int(np.where(order == 0)[0][0]) - int(np.where(order == 1)[0][0])) == 1


def test_specific_group_is_tighter():
    t = np.arange(64)
    shared = np.sin(2 * np.pi * t / 64)
    specific = np.vstack([shared, shared + 0.02, 0.98 * shared])
    nonspecific = np.vstack(
        [
            np.cos(2 * np.pi * t / 64),
            np.sin(2 * np.pi * 3 * t / 64),
            np.cos(2 * np.pi * 5 * t / 64),
        ]
    )
    result = compare_trace_groups(
        specific,
        nonspecific,
        ["S1", "S2", "S3"],
        ["N1", "N2", "N3"],
    )
    assert result["specific"]["mean_r"] > 0.95
    assert result["specific"]["mean_r"] > result["nonspecific"]["mean_r"]
    assert result["exhaustive"]
    assert result["p_specific_greater"] <= 0.05
    assert similarity_conclusion(result).startswith("Specific traces")
    assert result["specific"]["names"][0] in {"S1", "S2", "S3"}
    assert result["specific"]["matrix"].shape == (3, 3)


def test_flat_traces_are_left_out_of_the_comparison():
    varying = np.vstack([np.arange(8), np.arange(8) + 1])
    flat = np.ones((1, 8))
    result = compare_trace_groups(varying, flat, ["S1", "S2"], ["N1"])
    assert result["dropped"] == ["N1"]
    assert result["p_specific_greater"] is None
    assert "at least two" in similarity_conclusion(result)


def test_event_onsets_and_trial_zscores():
    assert event_onsets([[5, 1], [5, 7], [8, 9]]) == [1, 5, 8]
    trace = np.array([8, 10, 12, 14, 16], dtype=float)
    trials, skipped = trial_zscores(trace, [3], baseline=3, post=2)
    assert skipped == 0
    np.testing.assert_allclose(trials[0], [-1, 0, 1, 2, 3])
    assert np.allclose(epoch_x(3, 2), [-3, -2, -1, 0, 1])
    off, skipped_off = trial_zscores(trace, [1], baseline=3, post=2)
    assert off.shape == (0, 5) and skipped_off == 1
    flat, skipped_flat = trial_zscores(np.ones(8), [3], baseline=3, post=2)
    assert flat.shape[0] == 0 and skipped_flat == 1


def test_mean_and_sem():
    samples = np.array([[0.0, 2.0], [2.0, 4.0]])
    mean, sem = mean_and_sem(samples)
    np.testing.assert_allclose(mean, [1, 3])
    np.testing.assert_allclose(sem, [1, 1])


def test_pooled_cut_separates_two_shapes():
    wave = np.array([0.0, 1, 0, 1, 0, 1])
    other = np.array([1.0, 0, 1, 0, 1, 0])
    traces = np.vstack([wave, wave + 0.01, other, other + 0.01])
    names = ["S1", "N1", "S2", "N2"]
    kinds = ["specific", "nonspecific", "specific", "nonspecific"]
    for metric in (METRIC_RUZICKA, METRIC_EUCLIDEAN):
        view = hierarchical_view(traces, names, kinds, metric, ORDER_POOLED)
        grouped = [set(group) for group in _names_in_spans(view)]
        assert grouped == [{"S1", "N1"}, {"S2", "N2"}] or grouped == [{"S2", "N2"}, {"S1", "N1"}]
        separate = hierarchical_view(traces, names, kinds, metric, ORDER_SEPARATE, cut=0.1)
        assert separate["clusters"] is None and separate["split"] == 2
        assert set(separate["names"][:2]) == {"S1", "S2"}


def _names_in_spans(view) -> list[list[str]]:
    return [[view["names"][i] for i in span] for span in view["clusters"]]


def test_cut_follows_the_tree_and_counts_singletons():
    traces = np.vstack([
        [0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0],
    ])
    names = ["A", "B", "C", "D"]
    kinds = ["specific"] * 4
    view = hierarchical_view(traces, names, kinds, METRIC_EUCLIDEAN, ORDER_POOLED)
    assert len(view["trees"]) == 1
    heights = np.sort(view["linkage"][:, 2])
    assert heights[-1] > heights[-2]
    mid = recut_view(view, float((heights[-1] + heights[-2]) / 2))
    assert 1 < len(mid["clusters"]) < 4
    assert max(len(span) for span in mid["clusters"]) < 4
    assert len(recut_view(view, float(heights[-1]))["clusters"]) == 1
    assert len(recut_view(view, 0.0)["clusters"]) == 4
    branches = dendrogram_branches(view["linkage"])
    assert len(branches) == 3
    lowest = min(branches, key=lambda item: float(item[0][1]))
    assert abs(float(lowest[1][0]) - float(lowest[1][2])) == pytest.approx(1.0)


def test_raster_sort_follows_each_tree():
    wave = np.array([0.0, 1, 0, 1, 0, 1])
    other = np.array([1.0, 0, 1, 0, 1, 0])
    rows = [
        {"name": "S1", "kind": "specific", "trace": wave},
        {"name": "N1", "kind": "nonspecific", "trace": other},
        {"name": "S2", "kind": "specific", "trace": wave + 0.02},
        {"name": "N2", "kind": "nonspecific", "trace": other + 0.02},
    ]
    pooled = ordered_rois(rows, hac_sort_key(METRIC_RUZICKA, ORDER_POOLED), len(wave))
    # identical shapes are neighbours somewhere in the order
    names = [roi["name"] for roi in pooled]
    assert abs(names.index("S1") - names.index("S2")) == 1
    assert abs(names.index("N1") - names.index("N2")) == 1
    separate = ordered_rois(rows, hac_sort_key(METRIC_EUCLIDEAN, ORDER_SEPARATE), len(wave))
    separate_names = [roi["name"] for roi in separate]
    assert set(separate_names[:2]) == {"S1", "S2"}
    assert set(separate_names[2:]) == {"N1", "N2"}


def test_old_pickle_gains_an_annotation_list(tmp_path):
    doc = new_document({"filename": "a.tif", "shape": (4, 2, 2), "dtype": "uint16", "file_size": 1})
    del doc["annotations"]
    path = tmp_path / "a_rci.pkl"
    with open(path, "wb") as fh:
        pickle.dump(doc, fh)
    loaded = load_document(path)
    assert loaded["annotations"] == []
