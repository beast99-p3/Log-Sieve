#!/usr/bin/env python3
"""
Log-Sieve — Tier 1 gateway client (Company Aggregator) for Hierarchical FL.

This client sits between multiple internal sensors (CSV files) and the Tier 2
global Flower server:
1. Train a separate local LogisticRegression model per internal CSV.
2. Compute a sample-weighted FedAvg across internal models into a single
   "Company Rulebook" (one parameter update).
3. Upload only this single aggregated update to the Tier 2 server.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any

import numpy as np
from flwr.client import NumPyClient, start_client
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score

from fedprox_fit import fedprox_binary_logistic_fit
from utils import (
    NetworkTracker,
    SyncedFeatureSpace,
    load_network_csv,
    set_sklearn_logistic_params,
    sklearn_logistic_params,
    sync_feature_space_from_csvs,
)

# Reuse the upload-side noise/compression helpers from the single-sensor client.
from client import apply_laplace_noise_to_weights, compress_parameter_arrays


def _weighted_average_param_lists(
    local_params: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    if not local_params:
        raise ValueError("_weighted_average_param_lists: empty local_params")
    total_n = sum(n for _, n in local_params)
    if total_n <= 0:
        raise ValueError("_weighted_average_param_lists: zero total examples")

    coef_acc: np.ndarray | None = None
    intercept_acc: np.ndarray | None = None
    for params, n in local_params:
        scale = n / total_n
        coef = np.asarray(params[0], dtype=np.float64)
        intercept = np.asarray(params[1], dtype=np.float64)
        if coef_acc is None:
            coef_acc = coef * scale
            intercept_acc = intercept * scale
        else:
            coef_acc += coef * scale
            intercept_acc += intercept * scale

    assert coef_acc is not None and intercept_acc is not None
    return [coef_acc, intercept_acc]


class LogSieveGatewayClient(NumPyClient):
    """Flower NumPyClient that locally aggregates multiple internal sensors."""

    def __init__(
        self,
        data_paths: list[str],
        label_column: str | None,
        *,
        client_id: str = "gateway",
        schema_json: str | None = None,
        schema_categorical_oov: str | None = None,
        quantize: str = "none",
        sparsify_threshold: float | None = None,
        dp_laplace_scale: float = 0.0,
        dp_seed: int = 42,
    ) -> None:
        self.client_id = client_id
        self.data_paths = data_paths
        self.internal_clients = len(data_paths)
        if self.internal_clients < 1:
            raise ValueError("gateway_client: need at least one --data-paths entry")

        self.tracker = NetworkTracker()
        self.quantize = quantize
        self.sparsify_threshold = sparsify_threshold
        self.dp_laplace_scale = float(dp_laplace_scale)
        self._dp_rng = np.random.default_rng(dp_seed)

        self._last_global_params: list[np.ndarray] | None = None

        if schema_json:
            sf = SyncedFeatureSpace.load_json(schema_json)
            if schema_categorical_oov is not None:
                if schema_categorical_oov not in ("error", "zero"):
                    raise ValueError("--schema-categorical-oov must be 'error' or 'zero'")
                sf.categorical_oov = schema_categorical_oov  # type: ignore[assignment]
        else:
            cat_oov = schema_categorical_oov or "error"
            sf = sync_feature_space_from_csvs(
                data_paths, label_column, categorical_oov=cat_oov  # type: ignore[arg-type]
            )

        self._sf = sf
        self.X_list: list[np.ndarray] = []
        self.y_list: list[np.ndarray] = []
        for p in data_paths:
            df = load_network_csv(p)
            X, y = sf.transform(df)
            self.X_list.append(X)
            self.y_list.append(y)

        self.model = LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
            warm_start=True,
        )

        # Initial "Company Rulebook" used for get_parameters and until the first fit.
        with self.tracker.training_timer(phase="initial"):
            self._company_params = self._build_initial_company_rulebook()
        set_sklearn_logistic_params(self.model, self._company_params)

    def _build_initial_company_rulebook(self) -> list[np.ndarray]:
        local_params: list[tuple[list[np.ndarray], int]] = []
        for X, y in zip(self.X_list, self.y_list):
            if np.unique(y).size < 2:
                # LogisticRegression can't fit a single-class partition. Skip for initialization.
                continue
            m = LogisticRegression(
                max_iter=500,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            )
            m.fit(X, y)
            local_params.append((sklearn_logistic_params(m), len(y)))

        if not local_params:
            n_features = int(self.X_list[0].shape[1])
            zeros_w = np.zeros((1, n_features), dtype=np.float64)
            zeros_b = np.zeros(1, dtype=np.float64)
            return [zeros_w, zeros_b]

        return _weighted_average_param_lists(local_params)

    def _finalize_upload(
        self,
        raw_weights: list[np.ndarray],
        *,
        internal_clients: int,
    ) -> list[np.ndarray]:
        noisy = apply_laplace_noise_to_weights(raw_weights, self.dp_laplace_scale, self._dp_rng)
        packed = compress_parameter_arrays(
            noisy,
            self._last_global_params,
            quantize=self.quantize,
            sparsify_threshold=self.sparsify_threshold,
        )

        # Update WAN saving (hypothetical: if each internal dataset uploaded independently to Tier 2)
        self.tracker.record_wan_bytes_saved(internal_clients, packed)
        self.tracker.record_uplink(packed)
        return packed

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        _ = config
        # Upload the initial rulebook as a single Tier 2 update.
        return self._finalize_upload(
            self._company_params,
            internal_clients=self.internal_clients,
        )

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        if parameters and len(parameters) >= 2:
            self._last_global_params = [np.asarray(p, dtype=np.float64).copy() for p in parameters]
            set_sklearn_logistic_params(self.model, parameters)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", 1))
        proximal_mu = float(config.get("proximal_mu", 0.0))
        n_steps = min(2000, 200 + 200 * max(1, local_epochs))

        uplink_before = self.tracker.bytes_uplink_total
        wan_before = self.tracker.wan_bytes_saved_total

        with self.tracker.training_timer(phase="federated"):
            if self._last_global_params is None:
                raise RuntimeError("gateway_client.fit: missing global parameters")
            cg = self._last_global_params[0]
            bg = self._last_global_params[1]

            local_params: list[tuple[list[np.ndarray], int]] = []
            skipped_internal = 0
            for X, y in zip(self.X_list, self.y_list):
                uniq = np.unique(y)
                if uniq.size < 2:
                    skipped_internal += 1
                    local_params.append(([cg, bg], len(y)))
                    continue

                if uniq.size == 2:
                    coef_n, int_n = fedprox_binary_logistic_fit(
                        X,
                        y.astype(np.float64),
                        cg,
                        bg,
                        proximal_mu,
                        n_steps=n_steps,
                        lr=0.25,
                        seed=int(self._dp_rng.integers(0, 10_000_000)),
                    )
                    local_params.append(([coef_n, int_n], len(y)))
                    continue

                if proximal_mu > 0.0:
                    warnings.warn(
                        f"[{self.client_id}] FedProx path supports binary labels only; "
                        "falling back to LogisticRegression.fit (warm-start, no explicit proximal term).",
                        stacklevel=2,
                    )
                m = LogisticRegression(
                    max_iter=500,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=42,
                    warm_start=True,
                )
                m.max_iter = min(2000, 200 * max(1, local_epochs))
                set_sklearn_logistic_params(m, self._last_global_params)
                m.fit(X, y)
                local_params.append((sklearn_logistic_params(m), len(y)))

            company_params = _weighted_average_param_lists(local_params)
            self._company_params = company_params
            set_sklearn_logistic_params(self.model, company_params)

        packed = self._finalize_upload(
            self._company_params,
            internal_clients=self.internal_clients,
        )

        metrics: dict[str, Any] = {
            "comm_uplink_bytes_cumulative": float(self.tracker.bytes_uplink_total),
            "wan_bytes_saved": float(self.tracker.wan_bytes_saved_total),
            "train_cpu_seconds_cumulative": float(self.tracker.seconds_training_cpu_total),
            "train_cpu_seconds_initial": float(self.tracker.seconds_training_cpu_initial),
            "train_cpu_seconds_federated": float(self.tracker.seconds_training_cpu_federated),
            "dp_laplace_scale": float(self.dp_laplace_scale),
            "proximal_mu": float(proximal_mu),
        }
        metrics["comm_uplink_bytes_fit"] = float(self.tracker.bytes_uplink_total - uplink_before)
        metrics["wan_bytes_saved_fit"] = float(self.tracker.wan_bytes_saved_total - wan_before)
        metrics["skipped_internal_clients"] = float(skipped_internal)
        return packed, sum(len(y) for y in self.y_list), metrics

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        _ = config
        self.set_parameters(parameters)

        y_all: list[np.ndarray] = []
        y_hat_all: list[np.ndarray] = []
        y_prob_all: list[np.ndarray] = []
        for X, y in zip(self.X_list, self.y_list):
            y_all.append(y)
            y_hat_all.append(self.model.predict(X))
            y_prob_all.append(self.model.predict_proba(X))

        y_true = np.concatenate(y_all, axis=0)
        y_hat = np.concatenate(y_hat_all, axis=0)
        y_prob = np.concatenate(y_prob_all, axis=0)

        n_classes = len(np.unique(y_true))
        avg = "binary" if n_classes == 2 else "macro"
        acc = float(accuracy_score(y_true, y_hat))
        prec = float(precision_score(y_true, y_hat, average=avg, zero_division=0))
        rec = float(recall_score(y_true, y_hat, average=avg, zero_division=0))

        try:
            loss = float(log_loss(y_true, y_prob, labels=self.model.classes_))
        except ValueError:
            loss = float(1.0 - acc)

        return loss, int(y_true.shape[0]), {
            "comm_uplink_bytes_cumulative": float(self.tracker.bytes_uplink_total),
            "wan_bytes_saved": float(self.tracker.wan_bytes_saved_total),
            "train_cpu_seconds_cumulative": float(self.tracker.seconds_training_cpu_total),
            "train_cpu_seconds_initial": float(self.tracker.seconds_training_cpu_initial),
            "train_cpu_seconds_federated": float(self.tracker.seconds_training_cpu_federated),
            "dp_laplace_scale": float(self.dp_laplace_scale),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Log-Sieve Tier 1 gateway client (hierarchical FL).")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--data-paths", nargs="+", required=True, help="CSV files for internal sensors.")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--client-id", default="gateway")

    parser.add_argument("--schema-json", default=None)
    parser.add_argument(
        "--schema-categorical-oov",
        choices=("error", "zero"),
        default=None,
        help="Override SyncedFeatureSpace OOV policy (default: value stored in JSON; or 'error' when building a new schema).",
    )

    parser.add_argument("--quantize", choices=("none", "float16", "int8"), default="none")
    parser.add_argument("--sparsify-threshold", type=float, default=0.0)
    parser.add_argument(
        "--dp-laplace-scale",
        type=float,
        default=0.0,
        help="λ for Laplace(0,λ) noise on each uploaded weight (0 disables).",
    )
    parser.add_argument("--dp-seed", type=int, default=42, help="RNG seed for Laplace noise.")

    args = parser.parse_args()

    thresh = args.sparsify_threshold if args.sparsify_threshold > 0 else None
    client = LogSieveGatewayClient(
        args.data_paths,
        args.label_column,
        client_id=args.client_id,
        schema_json=args.schema_json,
        schema_categorical_oov=args.schema_categorical_oov,
        quantize=args.quantize,
        sparsify_threshold=thresh,
        dp_laplace_scale=args.dp_laplace_scale,
        dp_seed=args.dp_seed,
    )
    start_client(server_address=args.server_address, client=client.to_client())


if __name__ == "__main__":
    main()

