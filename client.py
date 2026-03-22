"""
Log-Sieve — Flower client (network node).

Local model: ``LogisticRegression`` + FedAvg-friendly weight tensors.

Supports **FedProx** (proximal μ from server config), optional **Laplace DP noise**
on uploaded weights, quantisation (**float16** / **int8**), sparsification,
``--malicious`` label poisoning, ``NetworkTracker`` telemetry, and a standalone
**inference throughput** benchmark (rows/sec ≈ flow PPS).
"""

from __future__ import annotations

import argparse
import json
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
    benchmark_row_classification_throughput,
    load_network_csv,
    preprocess_for_ids,
    set_sklearn_logistic_params,
    sklearn_logistic_params,
)


def apply_laplace_noise_to_weights(
    arrays: list[np.ndarray],
    scale: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """w_noisy = w + Laplace(0, scale) element-wise (client-side DP-style masking)."""
    if scale <= 0.0:
        return [np.asarray(a, dtype=np.float64) for a in arrays]
    return [
        np.asarray(a, dtype=np.float64) + rng.laplace(0.0, scale, size=np.asarray(a).shape)
        for a in arrays
    ]


def compress_parameter_arrays(
    weights: list[np.ndarray],
    reference: list[np.ndarray] | None,
    *,
    quantize: str,
    sparsify_threshold: float | None,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i, arr in enumerate(weights):
        a = np.asarray(arr, dtype=np.float64).copy()
        if reference is not None and sparsify_threshold is not None and sparsify_threshold > 0:
            ref = np.asarray(reference[i], dtype=np.float64)
            if ref.shape == a.shape:
                delta = np.abs(a - ref)
                a = np.where(delta < sparsify_threshold, ref, a)
        if quantize == "float16":
            a = a.astype(np.float16).astype(np.float64)
        elif quantize == "int8":
            m = float(np.max(np.abs(a)) + 1e-8)
            qi = np.clip(np.round(a / m * 127.0), -127.0, 127.0)
            a = (qi / 127.0) * m
        elif quantize not in ("none", ""):
            raise ValueError(f"Unknown quantize mode: {quantize!r}")
        out.append(a)
    return out


class LogSieveClient(NumPyClient):
    """Flower NumPyClient wrapping a local sklearn intrusion-detection model."""

    def __init__(
        self,
        csv_path: str,
        label_column: str | None,
        *,
        client_id: str = "node",
        malicious: bool = False,
        schema_json: str | None = None,
        quantize: str = "none",
        sparsify_threshold: float | None = None,
        dp_laplace_scale: float = 0.0,
        dp_seed: int = 42,
        schema_categorical_oov: str | None = None,
    ) -> None:
        self.client_id = client_id
        self.tracker = NetworkTracker()
        self.quantize = quantize
        self.sparsify_threshold = sparsify_threshold
        self.dp_laplace_scale = float(dp_laplace_scale)
        self._dp_rng = np.random.default_rng(dp_seed)
        self._last_global_params: list[np.ndarray] | None = None

        df = load_network_csv(csv_path)
        if malicious:
            df = self._apply_label_poisoning(df, label_column)

        if schema_json:
            sf = SyncedFeatureSpace.load_json(schema_json)
            if schema_categorical_oov is not None:
                if schema_categorical_oov not in ("error", "zero"):
                    raise ValueError("schema_categorical_oov must be 'error' or 'zero'")
                sf.categorical_oov = schema_categorical_oov
            X, y = sf.transform(df)
            self.X = X
            self.y = y
            self.scaler = sf.scaler
            self.encoders = sf.encoders
        else:
            X, y, self.scaler, self.encoders = preprocess_for_ids(
                df, label_column, fit=True
            )
            self.X = X
            self.y = y

        self.model = LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )
        if len(np.unique(self.y)) >= 2:
            with self.tracker.training_timer(phase="initial"):
                self.model.fit(self.X, self.y)
        else:
            warnings.warn(
                f"[{client_id}] Local partition has a single class; "
                "initial model not fit until diverse data arrives.",
                stacklevel=2,
            )

    @staticmethod
    def _apply_label_poisoning(df: Any, label_column: str | None) -> Any:
        from utils import infer_label_column

        df = df.copy()
        lc = infer_label_column(df, label_column)
        col = df[lc]
        if col.dtype == object or str(col.dtype) == "category":
            s = col.astype(str)
            uniques = sorted(s.unique())
            if len(uniques) < 2:
                return df
            inv_map = {u: uniques[len(uniques) - 1 - i] for i, u in enumerate(uniques)}
            df[lc] = s.map(inv_map)
        else:
            arr = col.to_numpy()
            u_sorted = np.sort(np.unique(arr))
            if len(u_sorted) < 2:
                return df
            rev_map = {int(v): int(u_sorted[len(u_sorted) - 1 - j]) for j, v in enumerate(u_sorted)}
            df[lc] = np.array([rev_map[int(x)] for x in arr.ravel()], dtype=arr.dtype)
        return df

    def _finalize_upload(self, raw_weights: list[np.ndarray]) -> list[np.ndarray]:
        """Laplace noise → compress → record uplink bytes; single noise draw per upload."""
        noisy = apply_laplace_noise_to_weights(raw_weights, self.dp_laplace_scale, self._dp_rng)
        packed = compress_parameter_arrays(
            noisy,
            self._last_global_params,
            quantize=self.quantize,
            sparsify_threshold=self.sparsify_threshold,
        )
        self.tracker.record_uplink(packed)
        return packed

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        _ = config
        if not hasattr(self.model, "coef_") or self.model.coef_ is None:
            n_features = self.X.shape[1]
            zeros_w = np.zeros((1, n_features), dtype=np.float64)
            zeros_b = np.zeros(1, dtype=np.float64)
            raw = [zeros_w, zeros_b]
        else:
            raw = sklearn_logistic_params(self.model)
        return self._finalize_upload(raw)

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
        self.model.max_iter = min(2000, 200 * max(1, local_epochs))

        unique = np.unique(self.y)
        metrics: dict[str, Any] = {
            "comm_uplink_bytes_cumulative": float(self.tracker.bytes_uplink_total),
            "train_cpu_seconds_cumulative": float(self.tracker.seconds_training_cpu_total),
            "train_cpu_seconds_initial": float(self.tracker.seconds_training_cpu_initial),
            "train_cpu_seconds_federated": float(self.tracker.seconds_training_cpu_federated),
            "dp_laplace_scale": float(self.dp_laplace_scale),
            "proximal_mu": float(proximal_mu),
        }
        if len(unique) < 2:
            metrics["skipped_fit"] = 1.0
            raw = sklearn_logistic_params(self.model)
            uplink_before = self.tracker.bytes_uplink_total
            packed = self._finalize_upload(raw)
            metrics["comm_uplink_bytes_fit"] = float(self.tracker.bytes_uplink_total - uplink_before)
            return packed, len(self.y), metrics

        with self.tracker.training_timer(phase="federated"):
            if proximal_mu > 0.0 and unique.size == 2 and self._last_global_params is not None:
                cg = self._last_global_params[0]
                bg = self._last_global_params[1]
                n_steps = min(2000, 200 + 200 * max(1, local_epochs))
                coef_n, int_n = fedprox_binary_logistic_fit(
                    self.X,
                    self.y.astype(np.float64),
                    cg,
                    bg,
                    proximal_mu,
                    n_steps=n_steps,
                    lr=0.25,
                    seed=int(self._dp_rng.integers(0, 10_000_000)),
                )
                self.model.coef_ = coef_n
                self.model.intercept_ = int_n
                self.model.classes_ = np.array([0, 1], dtype=np.int64)
                self.model.n_features_in_ = int(coef_n.shape[1])
            else:
                if proximal_mu > 0.0 and unique.size != 2:
                    warnings.warn(
                        f"[{self.client_id}] FedProx path supports binary labels only; "
                        "falling back to plain LogisticRegression.fit.",
                        stacklevel=2,
                    )
                self.model.fit(self.X, self.y)

        y_hat = self.model.predict(self.X)
        metrics["train_accuracy"] = float(accuracy_score(self.y, y_hat))
        raw = sklearn_logistic_params(self.model)
        uplink_before = self.tracker.bytes_uplink_total
        packed = self._finalize_upload(raw)
        metrics["comm_uplink_bytes_fit"] = float(self.tracker.bytes_uplink_total - uplink_before)
        return packed, len(self.y), metrics

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        _ = config
        self.set_parameters(parameters)
        metrics: dict[str, Any] = {
            "comm_uplink_bytes_cumulative": float(self.tracker.bytes_uplink_total),
            "train_cpu_seconds_cumulative": float(self.tracker.seconds_training_cpu_total),
            "train_cpu_seconds_initial": float(self.tracker.seconds_training_cpu_initial),
            "train_cpu_seconds_federated": float(self.tracker.seconds_training_cpu_federated),
            "dp_laplace_scale": float(self.dp_laplace_scale),
        }
        if not hasattr(self.model, "coef_") or self.model.coef_ is None:
            return float("inf"), len(self.y), {
                **metrics,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        y_prob = self.model.predict_proba(self.X)
        y_hat = self.model.predict(self.X)
        n_classes = len(np.unique(self.y))
        avg = "binary" if n_classes == 2 else "macro"
        acc = float(accuracy_score(self.y, y_hat))
        prec = float(precision_score(self.y, y_hat, average=avg, zero_division=0))
        rec = float(recall_score(self.y, y_hat, average=avg, zero_division=0))
        try:
            loss = float(log_loss(self.y, y_prob, labels=self.model.classes_))
        except ValueError:
            loss = float(1.0 - acc)
        metrics.update({"accuracy": acc, "precision": prec, "recall": rec})
        return loss, len(self.y), metrics


def run_inference_benchmark(
    data_path: str,
    label_column: str | None,
    *,
    warmup: int,
    duration_sec: float,
    schema_json: str | None,
) -> dict[str, float]:
    df = load_network_csv(data_path)
    if schema_json:
        sf = SyncedFeatureSpace.load_json(schema_json)
        X, y = sf.transform(df)
    else:
        X, y, _, _ = preprocess_for_ids(df, label_column, fit=True)
    model = LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced", random_state=42)
    if len(np.unique(y)) >= 2:
        model.fit(X, y)
    else:
        raise SystemExit("Benchmark needs at least two classes in the partition.")
    stats = benchmark_row_classification_throughput(
        model, X, warmup=warmup, duration_sec=duration_sec
    )
    stats["n_features"] = float(X.shape[1])
    stats["n_rows"] = float(X.shape[0])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Log-Sieve Flower client (local IDS node).")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--client-id", default="node")
    parser.add_argument("--malicious", action="store_true")
    parser.add_argument("--schema-json", default=None)
    parser.add_argument(
        "--schema-categorical-oov",
        choices=("error", "zero"),
        default=None,
        help="Override SyncedFeatureSpace OOV policy (default: value stored in JSON).",
    )
    parser.add_argument(
        "--quantize",
        choices=("none", "float16", "int8"),
        default="none",
    )
    parser.add_argument("--sparsify-threshold", type=float, default=0.0)
    parser.add_argument(
        "--dp-laplace-scale",
        type=float,
        default=0.0,
        help="λ for Laplace(0,λ) noise on each uploaded weight (0 disables).",
    )
    parser.add_argument("--dp-seed", type=int, default=42, help="RNG seed for Laplace draws.")
    parser.add_argument(
        "--benchmark-inference",
        action="store_true",
        help="Do not connect to Flower; print rows/sec (PPS proxy) for local predict().",
    )
    parser.add_argument("--bench-warmup", type=int, default=5)
    parser.add_argument("--bench-duration", type=float, default=2.0)
    args = parser.parse_args()

    if args.benchmark_inference:
        stats = run_inference_benchmark(
            args.data_path,
            args.label_column,
            warmup=args.bench_warmup,
            duration_sec=args.bench_duration,
            schema_json=args.schema_json,
        )
        print(json.dumps(stats, indent=2))
        return

    thresh = args.sparsify_threshold if args.sparsify_threshold > 0 else None
    client = LogSieveClient(
        args.data_path,
        args.label_column,
        client_id=args.client_id,
        malicious=args.malicious,
        schema_json=args.schema_json,
        quantize=args.quantize,
        sparsify_threshold=thresh,
        dp_laplace_scale=args.dp_laplace_scale,
        dp_seed=args.dp_seed,
        schema_categorical_oov=args.schema_categorical_oov,
    )
    start_client(server_address=args.server_address, client=client.to_client())


if __name__ == "__main__":
    main()
