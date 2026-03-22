"""
Byzantine-robust aggregation helpers for Log-Sieve (trimmed mean, Krum, distance filter).

Used by the Flower server instead of trusting every client update equally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from flwr.common import NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

if TYPE_CHECKING:
    pass


def _flatten_parameters(ndarrays: NDArrays) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in ndarrays])


def filter_results_by_distance_to_median(
    results: list[tuple[ClientProxy, FitRes]],
    outlier_frac: float,
) -> list[tuple[ClientProxy, FitRes]]:
    """
    Discard the most distant client updates (Euclidean to the coordinate-wise median).

    Keeps the ``(1 - outlier_frac)`` fraction of clients with smallest L2 distance
    to the per-coordinate median vector. If ``outlier_frac <= 0`` or too few
    clients, returns ``results`` unchanged.
    """
    if outlier_frac <= 0.0 or len(results) <= 2:
        return list(results)

    vectors = [_flatten_parameters(parameters_to_ndarrays(fr.parameters)) for _, fr in results]
    stack = np.stack(vectors, axis=0)
    med = np.median(stack, axis=0)
    dists = np.linalg.norm(stack - med, axis=1)
    order = np.argsort(dists)
    n = len(results)
    drop = int(np.ceil(n * outlier_frac))
    drop = min(drop, n - 1)
    keep = max(1, n - drop)
    keep_idx = set(order[:keep].tolist())
    return [results[i] for i in range(n) if i in keep_idx]


def _trimmed_mean_1d(values: np.ndarray, trim_ratio: float) -> np.floating:
    n = values.size
    if n == 1:
        return float(values[0])
    k = int(np.floor(n * trim_ratio))
    if 2 * k >= n:
        return float(np.median(values))
    sorted_v = np.sort(values)
    if k == 0:
        return float(np.mean(sorted_v))
    return float(np.mean(sorted_v[k : n - k]))


def trimmed_mean_ndarrays(
    weights: list[NDArrays],
    trim_ratio: float,
) -> NDArrays:
    """
    Coordinate-wise trimmed mean across clients (Blanchard et al. / Yin et al. style).

    For each tensor position, sort client values along the client dimension,
    drop the lowest/highest ``trim_ratio`` fraction, average the rest.
    """
    if not weights:
        raise ValueError("trimmed_mean_ndarrays: empty weights")
    if len(weights) == 1:
        return [np.asarray(a, dtype=np.float64).copy() for a in weights[0]]

    n_clients = len(weights)
    n_layers = len(weights[0])
    out: NDArrays = []
    for layer_idx in range(n_layers):
        stack = np.stack([np.asarray(w[layer_idx], dtype=np.float64) for w in weights], axis=0)
        flat = stack.reshape(n_clients, -1)
        trimmed_cols = np.array(
            [_trimmed_mean_1d(flat[:, j], trim_ratio) for j in range(flat.shape[1])],
            dtype=np.float64,
        )
        out.append(trimmed_cols.reshape(stack.shape[1:]))
    return out


def krum_select_ndarrays(
    weights: list[NDArrays],
    f: int,
) -> NDArrays:
    """
    Single-Krum: pick one client update minimizing the sum of squared distances
    to its ``n - f - 2`` nearest neighbors (Blanchard et al., 2017).

    ``f`` is an upper bound on Byzantine workers. If ``n`` is too small, falls
    back to the first client.
    """
    if not weights:
        raise ValueError("krum_select_ndarrays: empty weights")
    n = len(weights)
    if n == 1:
        return [np.asarray(a, dtype=np.float64).copy() for a in weights[0]]

    vectors = [_flatten_parameters(w) for w in weights]
    dist_sq = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sum((vectors[i] - vectors[j]) ** 2))
            dist_sq[i, j] = dist_sq[j, i] = d

    m = n - f - 2
    if m < 1:
        m = 1
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        row = np.sort(dist_sq[i])
        # row[0] == 0 (self); take next m smallest distances to other clients
        take = min(m, max(0, n - 1))
        scores[i] = float(np.sum(row[1 : 1 + take]))

    best = int(np.argmin(scores))
    return [np.asarray(a, dtype=np.float64).copy() for a in weights[best]]


def weighted_fedavg_ndarrays(
    weights_results: list[tuple[NDArrays, int]],
) -> NDArrays:
    """Sample-size-weighted average (standard FedAvg on tensors)."""
    if not weights_results:
        raise ValueError("weighted_fedavg_ndarrays: empty input")
    total = sum(n for _, n in weights_results)
    if total == 0:
        raise ValueError("weighted_fedavg_ndarrays: zero total examples")

    acc: list[np.ndarray] | None = None
    for ndarrays, num in weights_results:
        scale = num / total
        if acc is None:
            acc = [np.asarray(a, dtype=np.float64) * scale for a in ndarrays]
        else:
            for i, a in enumerate(ndarrays):
                acc[i] = acc[i] + np.asarray(a, dtype=np.float64) * scale
    assert acc is not None
    return acc


def robust_aggregate_parameters(
    results: list[tuple[ClientProxy, FitRes]],
    *,
    aggregation_mode: str,
    trim_ratio: float,
    outlier_frac: float,
    krum_f: int,
    use_distance_filter: bool,
) -> tuple[NDArrays, dict[str, float]]:
    """Apply distance filter (optional) then trimmed-mean / Krum / FedAvg."""
    cohort = list(results)
    if use_distance_filter and outlier_frac > 0:
        cohort = filter_results_by_distance_to_median(cohort, outlier_frac)

    weights_results: list[tuple[NDArrays, int]] = [
        (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        for _, fit_res in cohort
    ]
    weights_only = [w for w, _ in weights_results]
    if not weights_only:
        raise ValueError("robust_aggregate_parameters: no client tensors after filtering.")

    if aggregation_mode == "fedavg":
        aggregated = weighted_fedavg_ndarrays(weights_results)
    elif aggregation_mode == "trimmed_mean":
        aggregated = trimmed_mean_ndarrays(weights_only, trim_ratio)
    elif aggregation_mode == "krum":
        aggregated = krum_select_ndarrays(weights_only, krum_f)
    else:
        raise ValueError(f"Unknown aggregation_mode: {aggregation_mode!r}")

    meta = {
        "robust_clients_used": float(len(cohort)),
        "robust_clients_total": float(len(results)),
    }
    return aggregated, meta


def aggregate_fit_results_to_parameters(
    results: list[tuple[ClientProxy, FitRes]],
    failures: list[tuple[ClientProxy, FitRes] | BaseException],
    *,
    accept_failures: bool,
    aggregation_mode: str,
    trim_ratio: float,
    outlier_frac: float,
    krum_f: int,
    use_distance_filter: bool,
    fit_metrics_aggregation_fn: Callable[[list[tuple[int, dict[str, Scalar]]]], dict[str, Scalar]]
    | None,
    aggregation_mode_label: str,
) -> tuple[Parameters | None, dict[str, Scalar]]:
    """Shared server-side ``aggregate_fit`` body for Log-Sieve strategies."""
    if not results:
        return None, {}
    if not accept_failures and failures:
        return None, {}

    aggregated_ndarrays, meta = robust_aggregate_parameters(
        results,
        aggregation_mode=aggregation_mode,
        trim_ratio=trim_ratio,
        outlier_frac=outlier_frac,
        krum_f=krum_f,
        use_distance_filter=use_distance_filter,
    )
    parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

    metrics_aggregated: dict[str, Scalar] = {
        "robust_clients_used": meta["robust_clients_used"],
        "robust_clients_total": meta["robust_clients_total"],
        "aggregation_mode": aggregation_mode_label,
    }
    if fit_metrics_aggregation_fn:
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        metrics_aggregated.update(fit_metrics_aggregation_fn(fit_metrics))
    return parameters_aggregated, metrics_aggregated
