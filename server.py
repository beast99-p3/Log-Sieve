"""
Log-Sieve — Flower server (central aggregator).

Weight flow
-----------
- Clients upload NumPy parameter lists after local training.
- ``LogSieveFedAvg`` can apply **distance-based outlier removal**, then either
  **coordinate-wise trimmed mean** or **Krum** (instead of plain FedAvg).
- ``ServerConfig.round_timeout`` lets slow / stuck nodes **time out** so rounds
  proceed with whoever responded (stragglers land in ``failures``; with
  ``accept_failures=True`` the server still aggregates successful results).

On recent Flower versions, ``start_server()`` is deprecated in favor of the
SuperLink / SuperNode stack; this script keeps the classic entrypoint for research.
"""

from __future__ import annotations

import argparse
from typing import Any

import flwr as fl
from flwr.common import Scalar
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx

from robust_aggregate import aggregate_fit_results_to_parameters


def weighted_metrics_aggregate(
    results: list[tuple[int, dict[str, Scalar]]],
) -> dict[str, Scalar]:
    """Example-weighted mean of numeric client metrics (skips strings / bytes)."""
    if not results:
        return {}
    total_examples = sum(num for num, _ in results)
    if total_examples == 0:
        return {}
    keys: set[str] = set()
    for _, m in results:
        keys.update(m.keys())
    out: dict[str, Scalar] = {}
    for k in keys:
        acc = 0.0
        skip = False
        for num, metrics in results:
            v = metrics.get(k, 0.0)
            if isinstance(v, (int, float, bool)):
                acc += num * float(v)
            elif isinstance(v, str):
                skip = True
                break
            else:
                skip = True
                break
        if not skip:
            out[k] = acc / total_examples
    return out


def fit_config(server_round: int) -> dict[str, Scalar]:
    _ = server_round
    return {"local_epochs": 1}


class LogSieveFedAvg(FedAvg):
    """
    FedAvg with optional robust aggregation (distance filter + trimmed mean / Krum / FedAvg).
    """

    def __init__(
        self,
        *,
        aggregation_mode: str = "trimmed_mean",
        trim_ratio: float = 0.1,
        outlier_frac: float = 0.1,
        krum_f: int = 0,
        use_distance_filter: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("inplace", False)
        kwargs.setdefault("accept_failures", True)
        super().__init__(**kwargs)
        self.aggregation_mode = aggregation_mode
        self.trim_ratio = trim_ratio
        self.outlier_frac = outlier_frac
        self.krum_f = krum_f
        self.use_distance_filter = use_distance_filter

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Any | None, dict[str, bool | bytes | float | int | str]]:
        _ = server_round
        return aggregate_fit_results_to_parameters(
            results,
            failures,
            accept_failures=self.accept_failures,
            aggregation_mode=self.aggregation_mode,
            trim_ratio=self.trim_ratio,
            outlier_frac=self.outlier_frac,
            krum_f=self.krum_f,
            use_distance_filter=self.use_distance_filter,
            fit_metrics_aggregation_fn=self.fit_metrics_aggregation_fn,
            aggregation_mode_label=self.aggregation_mode,
        )


class LogSieveFedProx(FedProx):
    """
    FedProx (Li et al.) + same robust aggregation pipeline as ``LogSieveFedAvg``.

    Clients receive ``proximal_mu`` in ``FitIns.config`` and should minimise local
    loss plus ``(μ/2)||w - w_global||²``.
    """

    def __init__(
        self,
        *,
        aggregation_mode: str = "trimmed_mean",
        trim_ratio: float = 0.1,
        outlier_frac: float = 0.1,
        krum_f: int = 0,
        use_distance_filter: bool = True,
        proximal_mu: float = 0.1,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("accept_failures", True)
        # Flower's ``FedProx`` does not accept ``inplace`` (unlike ``FedAvg``).
        kwargs.pop("inplace", None)
        super().__init__(proximal_mu=proximal_mu, **kwargs)
        self.aggregation_mode = aggregation_mode
        self.trim_ratio = trim_ratio
        self.outlier_frac = outlier_frac
        self.krum_f = krum_f
        self.use_distance_filter = use_distance_filter

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Any | None, dict[str, bool | bytes | float | int | str]]:
        _ = server_round
        label = f"fedprox_mu={self.proximal_mu}:{self.aggregation_mode}"
        return aggregate_fit_results_to_parameters(
            results,
            failures,
            accept_failures=self.accept_failures,
            aggregation_mode=self.aggregation_mode,
            trim_ratio=self.trim_ratio,
            outlier_frac=self.outlier_frac,
            krum_f=self.krum_f,
            use_distance_filter=self.use_distance_filter,
            fit_metrics_aggregation_fn=self.fit_metrics_aggregation_fn,
            aggregation_mode_label=label,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Log-Sieve Flower server (robust FedAvg).")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=8080, help="Listen port.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds.")
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Minimum clients per fit/eval (match how many nodes you start).",
    )
    parser.add_argument(
        "--round-timeout",
        type=float,
        default=300.0,
        help="Per-round timeout in seconds; clients that do not finish in time are skipped "
        "(stragglers). Use 0 or negative to disable (wait indefinitely).",
    )
    parser.add_argument(
        "--aggregation",
        choices=("trimmed_mean", "krum", "fedavg"),
        default="trimmed_mean",
        help="Robust aggregator after distance filtering (unless --no-distance-filter).",
    )
    parser.add_argument(
        "--trim-ratio",
        type=float,
        default=0.1,
        help="Fraction trimmed from each tail per coordinate (trimmed_mean mode).",
    )
    parser.add_argument(
        "--outlier-frac",
        type=float,
        default=0.1,
        help="Fraction of clients dropped as farthest from the median update (0 disables).",
    )
    parser.add_argument(
        "--no-distance-filter",
        action="store_true",
        help="Skip Euclidean outlier removal; only trimmed mean / Krum / FedAvg runs.",
    )
    parser.add_argument(
        "--krum-f",
        type=int,
        default=0,
        help="Byzantine upper bound f for Krum (n - f - 2 nearest neighbors).",
    )
    parser.add_argument(
        "--strategy",
        choices=("fedavg", "fedprox"),
        default="fedavg",
        help="fedprox enables proximal regularisation toward the global model (better for non-IID).",
    )
    parser.add_argument(
        "--proximal-mu",
        type=float,
        default=0.1,
        help="FedProx μ (only used when --strategy fedprox).",
    )
    args = parser.parse_args()

    round_timeout: float | None
    if args.round_timeout is None or args.round_timeout <= 0:
        round_timeout = None
    else:
        round_timeout = float(args.round_timeout)

    common_kw: dict[str, Any] = dict(
        aggregation_mode=args.aggregation,
        trim_ratio=args.trim_ratio,
        outlier_frac=0.0 if args.no_distance_filter else args.outlier_frac,
        krum_f=args.krum_f,
        use_distance_filter=not args.no_distance_filter,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=weighted_metrics_aggregate,
        evaluate_metrics_aggregation_fn=weighted_metrics_aggregate,
        accept_failures=True,
    )
    if args.strategy == "fedprox":
        strategy = LogSieveFedProx(proximal_mu=args.proximal_mu, **common_kw)
    else:
        strategy = LogSieveFedAvg(**common_kw)

    fl.server.start_server(
        server_address=f"{args.host}:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds, round_timeout=round_timeout),
        strategy=strategy,
        grpc_max_message_length=512 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
