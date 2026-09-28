"""Trace comparisons for the trace-processing window.

Similarity uses Pearson correlation of the raw traces: it compares shape and
ignores amplitude, which is what "do these cells move together?" asks of
unnormalised fluorescence. Ordering within a group is average-linkage
clustering on distance ``1 - r`` (the same idea as the HAC view in
s2p Trace Curation, without an extra dependency).

Z-scores are per trial. Each annotation range is one event; its first frame
is the onset. A trial is the baseline frames before that onset plus the
post-stim frames from the onset, and it is divided by the sample standard
deviation of its own baseline.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from raw_cell_inspection.store import KIND_NONSPECIFIC, KIND_SPECIFIC

SORT_DOCUMENT = "document"
SORT_KIND = "kind"
SORT_SIMILARITY = "similarity"  # older saves: Ružička tree within each kind
SORT_MODES = (SORT_DOCUMENT, SORT_KIND, SORT_SIMILARITY)

METRIC_RUZICKA = "ruzicka"
METRIC_EUCLIDEAN = "euclidean"
ORDER_POOLED = "pooled"
ORDER_SEPARATE = "separate"
HAC_METRICS = (METRIC_RUZICKA, METRIC_EUCLIDEAN)
HAC_ORDERS = (ORDER_POOLED, ORDER_SEPARATE)


def fit_trace(trace, n_frames: int) -> np.ndarray:
    """Copy a trace into a length-n vector, padding with NaN if it is short."""
    out = np.full(int(n_frames), np.nan, dtype=np.float64)
    values = np.asarray(trace, dtype=np.float64).ravel()
    n = min(out.shape[0], values.shape[0])
    out[:n] = values[:n]
    return out


def minmax_trace(trace) -> np.ndarray:
    """Scale one trace to [0, 1]. A flat trace becomes zeros."""
    values = np.asarray(trace, dtype=np.float64)
    out = np.zeros(values.shape, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    lo = float(np.min(values[finite]))
    hi = float(np.max(values[finite]))
    if hi <= lo:
        return out
    out[finite] = (values[finite] - lo) / (hi - lo)
    return out


def varying_mask(traces: np.ndarray) -> np.ndarray:
    """Rows that are finite and not constant. ``traces`` is (n_roi, n_frames)."""
    if traces.size == 0 or traces.shape[0] == 0:
        return np.zeros(traces.shape[0], dtype=bool)
    finite = np.isfinite(traces).all(axis=1)
    spread = np.zeros(traces.shape[0], dtype=bool)
    spread[finite] = np.std(traces[finite], axis=1) > 1e-8
    return finite & spread


def pairwise_pearson(traces: np.ndarray) -> np.ndarray:
    """Pearson r for each pair of rows. Constant rows are NaN off the diagonal."""
    X = np.asarray(traces, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("traces must have shape (n_roi, n_frames)")
    n, t = X.shape
    out = np.full((n, n), np.nan, dtype=np.float64)
    if n == 0 or t < 2:
        return out
    centered = X - X.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.sum(centered * centered, axis=1))
    valid = scale > 1e-8
    denom = np.outer(scale, scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (centered @ centered.T) / denom
    corr = np.clip(corr, -1.0, 1.0)
    corr[~valid, :] = np.nan
    corr[:, ~valid] = np.nan
    idx = np.arange(n)
    corr[idx, idx] = np.where(valid, 1.0, np.nan)
    return corr


def upper_mean(similarity: np.ndarray) -> tuple[float, int]:
    """Mean of the upper triangle, ignoring NaN. Returns (mean, n_pairs)."""
    n = similarity.shape[0]
    if n < 2:
        return float("nan"), 0
    values = similarity[np.triu_indices(n, k=1)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), 0
    return float(values.mean()), int(values.size)


def leaf_order(similarity: np.ndarray) -> np.ndarray:
    """Average-linkage leaf order. Distance is ``max(0, 1 - r)``; NaN counts as 1."""
    n = int(similarity.shape[0])
    if n <= 2:
        return np.arange(n)
    distance = 1.0 - np.asarray(similarity, dtype=np.float64)
    distance[~np.isfinite(distance)] = 1.0
    np.maximum(distance, 0.0, out=distance)
    np.fill_diagonal(distance, 0.0)

    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    sizes = {i: 1 for i in range(n)}
    gaps: dict[int, dict[int, float]] = {
        i: {j: float(distance[i, j]) for j in range(n) if j != i} for i in range(n)
    }
    next_id = n
    while len(members) > 1:
        best: tuple[int, int] | None = None
        best_pair: tuple[int, int] | None = None
        best_d = np.inf
        ids = list(members)
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                gap = gaps[left][right]
                pair = (left, right) if left < right else (right, left)
                closer = gap < best_d - 1e-15
                tie = abs(gap - best_d) <= 1e-15 and best_pair is not None and pair < best_pair
                if best is None or closer or tie:
                    best = (left, right)
                    best_pair = pair
                    best_d = gap
        assert best is not None
        left, right = best
        others = [k for k in members if k != left and k != right]
        n_left, n_right = sizes[left], sizes[right]
        merged = members[left] + members[right]
        new_id = next_id
        next_id += 1
        gaps[new_id] = {}
        for other in others:
            combined = (n_left * gaps[left][other] + n_right * gaps[right][other]) / (n_left + n_right)
            gaps[new_id][other] = combined
            gaps[other][new_id] = combined
            del gaps[other][left]
            del gaps[other][right]
        del gaps[left], gaps[right]
        del members[left], members[right]
        members[new_id] = merged
        sizes[new_id] = n_left + n_right
    return np.asarray(next(iter(members.values())), dtype=int)


def _as_matrix(traces) -> np.ndarray:
    values = np.asarray(traces, dtype=np.float64)
    if values.ndim == 1:
        if values.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("expected one trace per row")
    return values


def _keep_varying(traces: np.ndarray, names: list[str] | None) -> tuple[np.ndarray, list[str], list[str]]:
    mask = varying_mask(traces)
    labels = list(names) if names is not None else [str(i) for i in range(traces.shape[0])]
    if len(labels) != traces.shape[0]:
        raise ValueError("names must match the number of traces")
    kept = [labels[i] for i in np.flatnonzero(mask)]
    dropped = [labels[i] for i in np.flatnonzero(~mask)]
    return traces[mask], kept, dropped


def _diff_for_subset(similarity: np.ndarray, subset, n: int) -> float:
    mask = np.zeros(n, dtype=bool)
    mask[list(subset)] = True
    specific, _ = upper_mean(similarity[np.ix_(mask, mask)])
    nonspecific, _ = upper_mean(similarity[np.ix_(~mask, ~mask)])
    return float(specific - nonspecific)


def _ordered_block(similarity: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    if similarity.shape[0] == 0:
        return similarity, []
    order = leaf_order(similarity)
    ordered = similarity[np.ix_(order, order)]
    return ordered, [names[int(i)] for i in order]


def compare_trace_groups(
    specific,
    nonspecific,
    specific_names: list[str] | None = None,
    nonspecific_names: list[str] | None = None,
    *,
    max_exact: int = 20000,
    n_perm: int = 4000,
    seed: int = 0,
) -> dict:
    """Compare mean within-group Pearson r for specific vs non-specific traces.

    The p-values reassign the kept ROIs to two groups of the same sizes.
    When there are few enough assignments they are all enumerated; otherwise
    ``n_perm`` random assignments are used. ``p_specific_greater`` is the
    one-sided probability of a difference at least as large as the one observed.
    """
    specific_m = _as_matrix(specific)
    nonspecific_m = _as_matrix(nonspecific)
    if (
        specific_m.shape[0]
        and nonspecific_m.shape[0]
        and specific_m.shape[1] != nonspecific_m.shape[1]
    ):
        raise ValueError("specific and non-specific traces have different lengths")
    kept_s, names_s, dropped_s = _keep_varying(specific_m, specific_names)
    kept_n, names_n, dropped_n = _keep_varying(nonspecific_m, nonspecific_names)
    n_s, n_n = kept_s.shape[0], kept_n.shape[0]
    if n_s and n_n:
        pooled = np.vstack([kept_s, kept_n])
    elif n_s:
        pooled = kept_s
    else:
        pooled = kept_n
    similarity = pairwise_pearson(pooled) if pooled.size else np.zeros((0, 0))
    mean_s, pairs_s = upper_mean(similarity[:n_s, :n_s]) if n_s else (float("nan"), 0)
    mean_n, pairs_n = upper_mean(similarity[n_s:, n_s:]) if n_n else (float("nan"), 0)
    if n_s and n_n:
        cross = similarity[:n_s, n_s:]
        cross = cross[np.isfinite(cross)]
        between = float(cross.mean()) if cross.size else float("nan")
        n_between = int(cross.size)
    else:
        between, n_between = float("nan"), 0
    difference = (
        float(mean_s - mean_n) if np.isfinite(mean_s) and np.isfinite(mean_n) else float("nan")
    )
    p_specific, p_nonspecific, n_assigned, exhaustive = _group_p_values(
        similarity, n_s, difference, max_exact=max_exact, n_perm=n_perm, seed=seed
    )
    matrix_s, ordered_s = _ordered_block(similarity[:n_s, :n_s], names_s)
    matrix_n, ordered_n = _ordered_block(similarity[n_s:, n_s:], names_n)
    return {
        "specific": {
            "mean_r": mean_s,
            "n_pairs": pairs_s,
            "n_rois": n_s,
            "names": ordered_s,
            "matrix": matrix_s,
        },
        "nonspecific": {
            "mean_r": mean_n,
            "n_pairs": pairs_n,
            "n_rois": n_n,
            "names": ordered_n,
            "matrix": matrix_n,
        },
        "difference": difference,
        "between_r": between,
        "n_between": n_between,
        "p_specific_greater": p_specific,
        "p_nonspecific_greater": p_nonspecific,
        "n_assignments": n_assigned,
        "exhaustive": exhaustive,
        "dropped": dropped_s + dropped_n,
    }


def _group_p_values(
    similarity: np.ndarray,
    n_specific: int,
    observed: float,
    *,
    max_exact: int,
    n_perm: int,
    seed: int,
) -> tuple[float | None, float | None, int, bool]:
    n = similarity.shape[0]
    n_other = n - n_specific
    if n_specific < 2 or n_other < 2 or not np.isfinite(observed):
        return None, None, 0, False
    n_comb = math.comb(n, n_specific)
    if n_comb <= max_exact:
        diffs = np.array(
            [_diff_for_subset(similarity, subset, n) for subset in combinations(range(n), n_specific)],
            dtype=np.float64,
        )
        return (
            float(np.mean(diffs >= observed - 1e-12)),
            float(np.mean(diffs <= observed + 1e-12)),
            int(diffs.size),
            True,
        )
    rng = np.random.default_rng(seed)
    greater = 0
    lesser = 0
    for _ in range(n_perm):
        subset = rng.choice(n, size=n_specific, replace=False)
        diff = _diff_for_subset(similarity, subset, n)
        greater += diff >= observed - 1e-12
        lesser += diff <= observed + 1e-12
    return (greater + 1) / (n_perm + 1), (lesser + 1) / (n_perm + 1), n_perm, False


def similarity_conclusion(result: dict) -> str:
    difference = result.get("difference")
    p_specific = result.get("p_specific_greater")
    p_nonspecific = result.get("p_nonspecific_greater")
    if p_specific is None or difference is None or not np.isfinite(difference):
        return "Need at least two varying traces in each group to compare them."
    if difference > 0 and p_specific <= 0.05:
        return "Specific traces are more similar to each other than non-specific traces are."
    if difference < 0 and p_nonspecific is not None and p_nonspecific <= 0.05:
        return "Non-specific traces are more similar to each other than specific traces are."
    return "The two groups are not clearly different in how similar their traces are."


def parse_hac_sort(mode: str) -> tuple[str, str] | None:
    """``hac:<metric>:<pooled|separate>``. The old similarity sort is Ružička within kind."""
    if mode == SORT_SIMILARITY:
        return METRIC_RUZICKA, ORDER_SEPARATE
    parts = str(mode).split(":")
    if len(parts) == 3 and parts[0] == "hac" and parts[1] in HAC_METRICS and parts[2] in HAC_ORDERS:
        return parts[1], parts[2]
    return None


def hac_sort_key(metric: str, order_mode: str) -> str:
    return f"hac:{metric}:{order_mode}"


def minmax_matrix(traces: np.ndarray) -> np.ndarray:
    if traces.size == 0:
        return np.zeros_like(traces, dtype=np.float64)
    return np.vstack([minmax_trace(row) for row in traces])


def ruzicka_distance_matrix(traces: np.ndarray) -> np.ndarray:
    """Distance ``1 - sum(min) / sum(max)`` between non-negative rows. Identical rows are 0."""
    X = np.asarray(traces, dtype=np.float64)
    n = X.shape[0]
    distance = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        rest = X[i + 1 :]
        shared = np.minimum(rest, X[i]).sum(axis=1)
        total = np.maximum(rest, X[i]).sum(axis=1)
        similarity = np.divide(shared, total, out=np.ones_like(shared), where=total > 0)
        distance[i, i + 1 :] = 1.0 - similarity
        distance[i + 1 :, i] = distance[i, i + 1 :]
    return distance


def euclidean_distance_matrix(traces: np.ndarray) -> np.ndarray:
    X = np.asarray(traces, dtype=np.float64)
    square = np.sum(X * X, axis=1, keepdims=True)
    squared = np.maximum(square + square.T - 2.0 * (X @ X.T), 0.0)
    np.fill_diagonal(squared, 0.0)
    return np.sqrt(squared)


def _linkage_order(n: int, gap_of, on_merge) -> tuple[np.ndarray, np.ndarray]:
    """Agglomerative clustering. Returns a linkage matrix and the leaf order."""
    if n < 2:
        return np.zeros((0, 4), dtype=np.float64), np.arange(n)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    sizes = {i: 1 for i in range(n)}
    active = set(range(n))
    linkage = np.zeros((n - 1, 4), dtype=np.float64)
    next_id = n
    for step in range(n - 1):
        best: tuple[int, int] | None = None
        best_pair: tuple[int, int] | None = None
        best_d = np.inf
        ids = sorted(active)
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                gap = float(gap_of(left, right))
                pair = (left, right) if left < right else (right, left)
                closer = gap < best_d - 1e-15
                tie = abs(gap - best_d) <= 1e-15 and best_pair is not None and pair < best_pair
                if best is None or closer or tie:
                    best = (left, right)
                    best_pair = pair
                    best_d = gap
        assert best is not None
        left, right = best
        count = sizes[left] + sizes[right]
        linkage[step] = [left, right, best_d, count]
        members[next_id] = members[left] + members[right]
        sizes[next_id] = count
        on_merge(next_id, left, right, sizes)
        active.discard(left)
        active.discard(right)
        active.add(next_id)
        del members[left], members[right], sizes[left], sizes[right]
        next_id += 1
    return linkage, np.asarray(members[next_id - 1], dtype=int)


def average_linkage(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average-linkage tree from a square distance matrix."""
    n = int(distance.shape[0])
    gaps: dict[int, dict[int, float]] = {
        i: {j: float(distance[i, j]) for j in range(n) if j != i} for i in range(n)
    }

    def on_merge(new_id, left, right, sizes):
        n_left, n_right = sizes[left], sizes[right]
        gaps[new_id] = {}
        for other in list(gaps):
            if other in (left, right) or left not in gaps[other]:
                continue
            combined = (n_left * gaps[left][other] + n_right * gaps[right][other]) / (n_left + n_right)
            gaps[new_id][other] = combined
            gaps[other][new_id] = combined
            del gaps[other][left]
            del gaps[other][right]
        del gaps[left], gaps[right]

    return _linkage_order(n, lambda left, right: gaps[left][right], on_merge)


def ward_linkage(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ward tree on Euclidean distance. Merge height is ``sqrt(na*nb/(na+nb)) * ||ca-cb||``."""
    X = np.asarray(traces, dtype=np.float64)
    n = int(X.shape[0])
    centroids = {i: X[i].copy() for i in range(n)}

    counts = {i: 1 for i in range(n)}

    def on_merge(new_id, left, right, sizes):
        n_left, n_right = sizes[left], sizes[right]
        centroids[new_id] = (n_left * centroids[left] + n_right * centroids[right]) / (n_left + n_right)
        counts[new_id] = n_left + n_right
        del centroids[left], centroids[right], counts[left], counts[right]

    def gap(left, right):
        n_left, n_right = counts[left], counts[right]
        scale = np.sqrt((n_left * n_right) / (n_left + n_right))
        return scale * float(np.linalg.norm(centroids[left] - centroids[right]))

    return _linkage_order(n, gap, on_merge)


def clusters_at_distance(linkage: np.ndarray, order: np.ndarray, threshold: float) -> list[list[int]]:
    """Contiguous leaf-position spans for a distance cut. Positions index ``order``."""
    n = int(len(order))
    if n == 0:
        return []
    if n == 1 or len(linkage) == 0:
        return [[0]]
    children: dict[int, tuple[int, int, float]] = {}
    for i, row in enumerate(linkage):
        children[n + i] = (int(row[0]), int(row[1]), float(row[2]))

    def leaves(node: int) -> list[int]:
        if node < n:
            return [node]
        left, right, _dist = children[node]
        return leaves(left) + leaves(right)

    groups: list[list[int]] = []

    def visit(node: int) -> None:
        if node < n:
            groups.append([node])
            return
        left, right, dist = children[node]
        # Relative slack only. An absolute 1e-9 would glue a whole tree whose
        # merges sit below that, so every threshold would look like one group.
        if dist <= threshold + 1e-12 * max(abs(threshold), 1.0):
            groups.append(leaves(node))
        else:
            visit(left)
            visit(right)

    visit(n + len(linkage) - 1)
    position = {int(leaf): i for i, leaf in enumerate(order)}
    spans: list[list[int]] = []
    for group in groups:
        spans.append(sorted(position[i] for i in group))
    spans.sort(key=lambda span: span[0])
    return spans


def default_cut(linkage: np.ndarray) -> float:
    """0.7 × the tallest merge, the same default scipy uses to colour a dendrogram."""
    if len(linkage) == 0:
        return 0.0
    return 0.7 * float(np.max(linkage[:, 2]))


def _linkage_and_order(traces: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    n = int(traces.shape[0])
    if n < 2:
        return np.zeros((0, 4), dtype=np.float64), np.arange(n)
    if metric == METRIC_EUCLIDEAN:
        return ward_linkage(traces)
    return average_linkage(ruzicka_distance_matrix(traces))


def _tree_order(traces: np.ndarray, metric: str) -> np.ndarray:
    _linkage, order = _linkage_and_order(traces, metric)
    return order


def _block_order(indices: list[int], traces: np.ndarray, metric: str) -> list[int]:
    if len(indices) < 2:
        return list(indices)
    order = _tree_order(traces[indices], metric)
    return [indices[int(i)] for i in order]


def _block_linkage(indices: list[int], traces: np.ndarray, metric: str) -> tuple[list[int], np.ndarray]:
    if len(indices) < 2:
        return list(indices), np.zeros((0, 4), dtype=np.float64)
    chosen = np.asarray(indices, dtype=int)
    linkage, local = _linkage_and_order(traces[chosen], metric)
    return [indices[int(i)] for i in local], linkage


def dendrogram_branches(linkage: np.ndarray, row_offset: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """U-shaped branches. ``x`` is merge distance, ``y`` is the leaf-row centre.

    Leaf ``i`` sits at ``i + 0.5 + row_offset``, the centre of matrix row ``i``.
    """
    linkage = np.asarray(linkage, dtype=np.float64)
    if linkage.ndim != 2 or linkage.shape[0] == 0:
        return []
    n = int(linkage.shape[0] + 1)
    children: dict[int, tuple[int, int, float]] = {}
    for i, row in enumerate(linkage):
        children[n + i] = (int(row[0]), int(row[1]), float(row[2]))
    start: dict[int, int] = {}
    end: dict[int, int] = {}
    height: dict[int, float] = {}
    cursor = 0

    def place(node: int) -> None:
        nonlocal cursor
        if node < n:
            start[node] = cursor
            end[node] = cursor
            height[node] = 0.0
            cursor += 1
            return
        left, right, dist = children[node]
        place(left)
        place(right)
        start[node] = start[left]
        end[node] = end[right]
        height[node] = dist

    place(n + len(linkage) - 1)
    branches: list[tuple[np.ndarray, np.ndarray]] = []
    for node, (left, right, dist) in children.items():
        y_left = (start[left] + end[left]) / 2.0 + 0.5 + row_offset
        y_right = (start[right] + end[right]) / 2.0 + 0.5 + row_offset
        branches.append((
            np.array([height[left], dist, dist, height[right]], dtype=np.float64),
            np.array([y_left, y_left, y_right, y_right], dtype=np.float64),
        ))
    return branches


def recut_view(view: dict, cut: float) -> dict:
    """Apply a new distance cut to a pooled tree without rebuilding it."""
    if view.get("order_mode") != ORDER_POOLED:
        return view
    linkage = np.asarray(view["linkage"], dtype=np.float64)
    max_height = float(view["max_height"])
    if cut is None or not np.isfinite(cut):
        cut_value = default_cut(linkage)
    else:
        cut_value = float(cut)
    if max_height > 0:
        cut_value = float(np.clip(cut_value, 0.0, max_height))
    else:
        cut_value = 0.0
    updated = dict(view)
    updated["cut"] = cut_value
    updated["clusters"] = clusters_at_distance(linkage, np.asarray(view["leaf_order"]), cut_value)
    return updated


def hierarchical_view(
    traces: np.ndarray,
    names: list[str],
    kinds: list[str],
    metric: str,
    order_mode: str,
    cut: float | None = None,
) -> dict:
    """One pairwise matrix of min–max traces, ordered by a Ružička or Ward tree.

    ``order_mode`` ``pooled`` uses one tree and a distance cut. ``separate`` puts
    specific ROIs in one block and non-specific ROIs in the other, each ordered
    by its own tree, and does not cut.
    """
    if metric not in HAC_METRICS:
        raise ValueError(f"unknown metric {metric}")
    if order_mode not in HAC_ORDERS:
        raise ValueError(f"unknown order {order_mode}")
    scaled = minmax_matrix(np.asarray(traces, dtype=np.float64))
    if len(names) != scaled.shape[0] or len(kinds) != scaled.shape[0]:
        raise ValueError("names and kinds must match the number of traces")
    mask = varying_mask(scaled)
    kept = np.flatnonzero(mask)
    dropped = [names[i] for i in np.flatnonzero(~mask)]
    X = scaled[mask]
    kept_names = [names[i] for i in kept]
    kept_kinds = [kinds[i] for i in kept]
    n = X.shape[0]
    if metric == METRIC_EUCLIDEAN:
        distance = euclidean_distance_matrix(X) if n else np.zeros((0, 0))
        linkage, pooled_order = _linkage_and_order(X, metric)
        display = "distance"
        shown = distance
    else:
        distance = ruzicka_distance_matrix(X) if n else np.zeros((0, 0))
        linkage, pooled_order = _linkage_and_order(X, metric)
        display = "similarity"
        shown = 1.0 - distance
    if order_mode == ORDER_SEPARATE:
        spec = [i for i, kind in enumerate(kept_kinds) if kind == KIND_SPECIFIC]
        nonspec = [i for i, kind in enumerate(kept_kinds) if kind != KIND_SPECIFIC]
        spec_order, spec_z = _block_linkage(spec, X, metric)
        nonspec_order, nonspec_z = _block_linkage(nonspec, X, metric)
        order = np.asarray(spec_order + nonspec_order, dtype=int)
        trees = []
        if len(spec) >= 2:
            trees.append({"linkage": spec_z, "row": 0})
        if len(nonspec) >= 2:
            trees.append({"linkage": nonspec_z, "row": len(spec)})
        cut_value = None
        clusters = None
        max_height = 0.0
        split = len(spec)
        leaf_order = order
        pooled_linkage = np.zeros((0, 4), dtype=np.float64)
    else:
        order = pooled_order
        trees = [{"linkage": linkage, "row": 0}] if n >= 2 else []
        max_height = float(np.max(linkage[:, 2])) if len(linkage) else 0.0
        cut_value = default_cut(linkage) if cut is None or not np.isfinite(cut) else float(cut)
        cut_value = float(np.clip(cut_value, 0.0, max_height if max_height > 0 else 0.0))
        clusters = clusters_at_distance(linkage, order, cut_value)
        split = None
        leaf_order = order
        pooled_linkage = linkage
    matrix = shown[np.ix_(order, order)] if n else shown
    return {
        "names": [kept_names[int(i)] for i in order],
        "kinds": [kept_kinds[int(i)] for i in order],
        "matrix": np.asarray(matrix, dtype=np.float64),
        "display": display,
        "clusters": clusters,
        "cut": cut_value,
        "max_height": max_height,
        "split": split,
        "dropped": dropped,
        "metric": metric,
        "order_mode": order_mode,
        "linkage": pooled_linkage,
        "leaf_order": np.asarray(leaf_order, dtype=int),
        "trees": trees,
    }


def ordered_rois(rois: list[dict], mode: str, n_frames: int) -> list[dict]:
    """ROI rows for the raster. Unknown modes fall back to document order."""
    rows = [roi for roi in rois if roi.get("trace") is not None]
    if mode == SORT_KIND:
        return _kind_order(rows)
    parsed = parse_hac_sort(mode)
    if parsed is not None:
        return _hac_roi_order(rows, n_frames, *parsed)
    return list(rows)


def _kind_order(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("kind") == KIND_SPECIFIC] + [
        r for r in rows if r.get("kind") != KIND_SPECIFIC
    ]


def _order_varying(rows: list[dict], n_frames: int, metric: str) -> list[dict]:
    if len(rows) < 2:
        return list(rows)
    stacked = minmax_matrix(np.vstack([fit_trace(roi["trace"], n_frames) for roi in rows]))
    mask = varying_mask(stacked)
    varying = [roi for roi, keep in zip(rows, mask) if keep]
    flat = [roi for roi, keep in zip(rows, mask) if not keep]
    if len(varying) < 2:
        return varying + flat
    order = _tree_order(stacked[mask], metric)
    return [varying[int(i)] for i in order] + flat


def _hac_roi_order(rows: list[dict], n_frames: int, metric: str, order_mode: str) -> list[dict]:
    if order_mode == ORDER_SEPARATE:
        specific = [roi for roi in rows if roi.get("kind") == KIND_SPECIFIC]
        nonspecific = [roi for roi in rows if roi.get("kind") != KIND_SPECIFIC]
        return _order_varying(specific, n_frames, metric) + _order_varying(nonspecific, n_frames, metric)
    return _order_varying(rows, n_frames, metric)


def event_onsets(ranges) -> list[int]:
    """First frame of each range, in order, without duplicate onsets."""
    onsets: list[int] = []
    for item in ranges or []:
        try:
            start, end = int(item[0]), int(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if end < start:
            start = end
        if start not in onsets:
            onsets.append(start)
    return onsets


def epoch_x(baseline: int, post: int) -> np.ndarray:
    """Frame offsets from the onset: ``-baseline .. post-1``."""
    return np.arange(-int(baseline), int(post), dtype=np.float64)


def trial_zscores(trace, onsets, baseline: int, post: int) -> tuple[np.ndarray, int]:
    """Z-score each complete trial by its own baseline.

    Returns ``(n_kept, baseline + post)`` and how many events were left out
    (window past either end of the trace, a non-finite sample, or a flat baseline).
    Baseline must be at least 2 frames so the sample standard deviation exists.
    """
    values = np.asarray(trace, dtype=np.float64).ravel()
    baseline = int(baseline)
    post = int(post)
    if baseline < 2 or post < 1:
        raise ValueError("baseline must be at least 2 frames and post-stim at least 1")
    length = baseline + post
    kept: list[np.ndarray] = []
    skipped = 0
    n = values.shape[0]
    for onset in onsets:
        start = int(onset) - baseline
        stop = int(onset) + post
        if start < 0 or stop > n:
            skipped += 1
            continue
        segment = values[start:stop]
        if segment.shape[0] != length or not np.isfinite(segment).all():
            skipped += 1
            continue
        base = segment[:baseline]
        sd = float(base.std(ddof=1))
        if not np.isfinite(sd) or sd < 1e-8:
            skipped += 1
            continue
        kept.append((segment - float(base.mean())) / sd)
    if not kept:
        return np.zeros((0, length), dtype=np.float64), skipped
    return np.vstack(kept), skipped


def mean_and_sem(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SEM across rows. SEM is 0 when there is only one row."""
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("need at least one trace")
    mean = samples.mean(axis=0)
    if samples.shape[0] < 2:
        return mean, np.zeros_like(mean)
    sem = samples.std(axis=0, ddof=1) / np.sqrt(samples.shape[0])
    return mean, sem
