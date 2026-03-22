#!/usr/bin/env python3
"""
Automated experiment sweeper for Log-Sieve paper-style grids.

Runs ``server.py`` + two ``client.py`` subprocesses per configuration, parses Flower
logs for final **accuracy** and **loss**, and appends one CSV row per run.

Notes
-----
- **Network latency** values are recorded as ``configured_netem_delay_ms`` for your
  plots. Applying ``tc`` to real interfaces requires Linux + sudo and correct iface
  names; use ``--tc-iface`` only when you know what you are doing.
- On Windows, sweeps still run FL; ``tc`` steps are skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import socket
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_log_metrics(log: str) -> tuple[float, float]:
    """Return (last_distributed_accuracy, last_distributed_loss) from captured server log."""
    text = strip_ansi(log)
    acc = float("nan")
    loss = float("nan")

    i_acc = text.rfind("'accuracy'")
    if i_acc != -1:
        chunk = text[i_acc : i_acc + 600]
        pairs = re.findall(r"\(\s*(\d+)\s*,\s*([0-9.eE+-]+)\s*\)", chunk)
        if pairs:
            acc = float(pairs[-1][1])

    losses = re.findall(r"round\s+\d+:\s*([0-9.eE+-]+)", text)
    if losses:
        loss = float(losses[-1])
    return acc, loss


def pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return int(p)


def maybe_apply_tc(iface: str | None, latency_ms: float, rate: str) -> None:
    if iface is None or latency_ms <= 0:
        return
    if platform.system() != "Linux":
        print("Skipping tc (not Linux).", file=sys.stderr)
        return
    script = ROOT / "simulate_network.sh"
    if not script.is_file():
        return
    delay = f"{int(latency_ms)}ms"
    cmd = ["sudo", str(script), "apply", iface, rate, delay]
    print("Running:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=str(ROOT), check=False)


def run_one(
    *,
    port: int,
    outlier_frac: float,
    latency_ms: float,
    quantize: str,
    strategy: str,
    proximal_mu: float,
    rounds: int,
    data_dir: Path,
    dp_scale: float,
    apply_tc_iface: str | None,
    tc_rate: str,
) -> dict[str, object]:
    maybe_apply_tc(apply_tc_iface, latency_ms, tc_rate)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    srv_cmd = [
        sys.executable,
        str(ROOT / "server.py"),
        "--port",
        str(port),
        "--rounds",
        str(rounds),
        "--min-clients",
        "2",
        "--round-timeout",
        "120",
        "--outlier-frac",
        str(outlier_frac),
        "--aggregation",
        "trimmed_mean",
        "--strategy",
        strategy,
        "--proximal-mu",
        str(proximal_mu),
    ]
    srv = subprocess.Popen(
        srv_cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    time.sleep(4.0)

    def client_cmd(node_id: int) -> list[str]:
        return [
            sys.executable,
            str(ROOT / "client.py"),
            "--server-address",
            f"127.0.0.1:{port}",
            "--data-path",
            str(data_dir / f"node_{node_id}.csv"),
            "--client-id",
            f"sweep-{node_id}",
            "--quantize",
            quantize,
            "--dp-laplace-scale",
            str(dp_scale),
        ]

    c0 = subprocess.Popen(client_cmd(0), cwd=str(ROOT), env=env)
    c1 = subprocess.Popen(client_cmd(1), cwd=str(ROOT), env=env)

    out, _ = srv.communicate(timeout=600)
    log = out or ""
    c0.wait(timeout=120)
    c1.wait(timeout=120)

    acc, loss = parse_log_metrics(log)
    return {
        "port": port,
        "outlier_frac": outlier_frac,
        "configured_netem_delay_ms": latency_ms,
        "quantize": quantize,
        "strategy": strategy,
        "proximal_mu": proximal_mu,
        "dp_laplace_scale": dp_scale,
        "final_accuracy": acc,
        "final_loss": loss,
        "rounds": rounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Log-Sieve experiment sweeper → CSV.")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "experiment_results.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data_partitions_skewed")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="Single grid point smoke test.")
    parser.add_argument(
        "--tc-iface",
        default=None,
        help="Linux only: pass to simulate_network.sh apply (e.g. veth in netns).",
    )
    parser.add_argument("--tc-rate", default="100mbit", help="HTB rate passed to simulate_network.sh.")
    parser.add_argument(
        "--dp-scales",
        default="0",
        help="Comma-separated Laplace λ list (same noise on both clients for sweep).",
    )
    args = parser.parse_args()

    if not (args.data_dir / "node_0.csv").is_file():
        print(f"Missing {args.data_dir / 'node_0.csv'} — run:", file=sys.stderr)
        print("  python utils.py skewed-demo --output-dir data_partitions_skewed --num-clients 2", file=sys.stderr)
        sys.exit(1)

    outlier_fracs = [0.1, 0.2, 0.4]
    latencies_ms = [0.0, 10.0, 100.0, 500.0]
    quantize_levels = ["none", "float16", "int8"]
    strategies = [("fedavg", 0.0), ("fedprox", 0.1)]
    dp_scales = [float(x) for x in args.dp_scales.split(",") if x.strip()]
    if not dp_scales:
        dp_scales = [0.0]

    if args.quick:
        outlier_fracs = [0.1]
        latencies_ms = [0.0]
        quantize_levels = ["float16"]
        strategies = [("fedprox", 0.1)]

    fieldnames = [
        "outlier_frac",
        "configured_netem_delay_ms",
        "quantize",
        "strategy",
        "proximal_mu",
        "dp_laplace_scale",
        "final_accuracy",
        "final_loss",
        "rounds",
        "port",
    ]
    write_header = not args.output_csv.is_file()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for of, lat, qz, (st, pmu), dps in product(
            outlier_fracs, latencies_ms, quantize_levels, strategies, dp_scales
        ):
            port = pick_free_port()
            row = run_one(
                port=port,
                outlier_frac=of,
                latency_ms=lat,
                quantize=qz,
                strategy=st,
                proximal_mu=pmu,
                rounds=args.rounds,
                data_dir=args.data_dir,
                dp_scale=dps,
                apply_tc_iface=args.tc_iface,
                tc_rate=args.tc_rate,
            )
            w.writerow({k: row.get(k, "") for k in fieldnames})
            f.flush()
            print(row, flush=True)


if __name__ == "__main__":
    main()
