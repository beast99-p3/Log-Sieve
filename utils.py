"""
Log-Sieve — data utilities for federated network IDS.

Handles loading CSV traffic logs (e.g. UNSW-NB15, CIC-IDS2017-style tables),
basic preprocessing (scaling, encodings), and IID partitioning across virtual
nodes without ever sending raw rows to the aggregator (that happens in client.py).
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_network_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a network-intrusion CSV with Pandas."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def infer_label_column(df: pd.DataFrame, label_column: str | None) -> str:
    """Pick a label column by name or common IDS dataset conventions."""
    if label_column is not None:
        if label_column not in df.columns:
            raise ValueError(f"Label column {label_column!r} not in dataframe columns.")
        return label_column
    candidates = ("label", "Label", "attack_cat", "Attack category")
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not infer label column. Pass --label-column explicitly. "
        f"Columns seen: {list(df.columns)}"
    )


def preprocess_for_ids(
    df: pd.DataFrame,
    label_column: str | None,
    scaler: StandardScaler | None = None,
    encoders: dict[str, LabelEncoder] | None = None,
    *,
    fit: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, dict[str, LabelEncoder]]:
    """
    Build X, y for sklearn: numeric scaling + label encoding for non-numeric
    feature columns (object/category).

    If ``fit`` is True, fits ``scaler`` and ``encoders`` on this dataframe
    (typical for each node's local partition). For strictly aligned categorical
    codes across nodes, you would need a shared schema or coordinated encoding;
    for numeric-only IDS exports this is usually unnecessary.

    Returns
    -------
    X, y, scaler, encoders
    """
    label_column = infer_label_column(df, label_column)
    feature_df = df.drop(columns=[label_column]).copy()
    y_raw = df[label_column]

    if encoders is None:
        encoders = {}
    if scaler is None:
        scaler = StandardScaler()

    for col in feature_df.columns:
        if feature_df[col].dtype == object or str(feature_df[col].dtype) == "category":
            if fit:
                le = LabelEncoder()
                # unseen categories on other nodes won't occur when fit is local
                feature_df[col] = le.fit_transform(feature_df[col].astype(str))
                encoders[col] = le
            else:
                le = encoders[col]
                feature_df[col] = feature_df[col].map(
                    lambda x, encoder=le: encoder.transform([str(x)])[0]
                    if str(x) in encoder.classes_
                    else -1
                )
                feature_df[col] = feature_df[col].replace(-1, 0)

    X = feature_df.to_numpy(dtype=np.float64)

    if y_raw.dtype == object or str(y_raw.dtype) == "category":
        if fit:
            y_le = LabelEncoder()
            y = y_le.fit_transform(y_raw.astype(str))
            encoders["__target__"] = y_le
        else:
            y_le = encoders["__target__"]
            y = y_le.transform(y_raw.astype(str))
    else:
        y = np.asarray(y_raw, dtype=np.int64).ravel()

    if fit:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y.astype(np.int64), scaler, encoders


def iid_partition_rows(df: pd.DataFrame, num_partitions: int, seed: int = 42) -> list[pd.DataFrame]:
    """Shuffle and split a dataframe into ``num_partitions`` IID shards."""
    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    idx_parts = np.array_split(np.arange(len(shuffled)), num_partitions)
    return [shuffled.iloc[idx].reset_index(drop=True) for idx in idx_parts]


def partition_label_skewed(
    df: pd.DataFrame,
    label_column: str | None,
    num_partitions: int,
    *,
    majority_fraction: float = 0.9,
    majority_classes: Sequence[Any] | None = None,
    rows_per_partition: int | None = None,
    seed: int = 42,
    disjoint: bool = False,
) -> list[pd.DataFrame]:
    """
    Non-IID shards: each node gets ``majority_fraction`` of rows from "its" majority class.

    For two classes (e.g. 0=Normal, 1=Attack), pass ``majority_classes=[0, 1, 0, ...]``
    so node 0 is mostly Normal, node 1 mostly Attack, etc. If ``majority_classes`` is
    omitted, alternates ``0, 1, 0, 1, ...`` for binary data.

    When ``disjoint`` is False (default), rows are sampled independently per partition
    (the same logical row may appear on multiple nodes). When ``disjoint`` is True,
    each row is assigned to at most one partition; sampling is without replacement from
    the unused pool (raises if the label distribution cannot satisfy the request).
    """
    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")
    if not 0.5 <= majority_fraction < 1.0:
        raise ValueError("majority_fraction should be in [0.5, 1.0).")

    lc = infer_label_column(df, label_column)
    if disjoint:
        df = df.reset_index(drop=True)
    y_series = df[lc]
    classes = sorted(pd.unique(y_series))

    if majority_classes is None:
        if len(classes) != 2:
            raise ValueError(
                "majority_classes required when more than two label values exist; "
                f"found {classes}."
            )
        majority_classes = [classes[i % 2] for i in range(num_partitions)]
    elif len(majority_classes) != num_partitions:
        raise ValueError("len(majority_classes) must equal num_partitions.")

    n_total = len(df)
    if rows_per_partition is None:
        n_per = n_total // num_partitions
    else:
        n_per = int(rows_per_partition)

    parts: list[pd.DataFrame] = []
    used = np.zeros(n_total, dtype=bool) if disjoint else None

    for k in range(num_partitions):
        maj_label = majority_classes[k]
        n_maj = int(round(n_per * majority_fraction))
        n_oth = max(0, n_per - n_maj)
        rs_m = seed + 10_000 * k + 1
        rs_o = seed + 10_000 * k + 2
        rs_sh = seed + 10_000 * k + 3

        if disjoint:
            assert used is not None
            rng_k = np.random.default_rng(seed + 10_000 * k + 7)
            y_eq = y_series == maj_label
            y_ne = y_series != maj_label
            avail_maj = np.where(y_eq.to_numpy() & ~used)[0]
            if len(avail_maj) < n_maj:
                raise ValueError(
                    f"disjoint=True: partition {k} needs {n_maj} unused rows with label {maj_label!r}, "
                    f"only {len(avail_maj)} available (check class balance vs majority_fraction)."
                )
            pick_m = rng_k.choice(avail_maj, size=n_maj, replace=False)
            used[pick_m] = True
            if n_oth == 0:
                ix = pick_m
            else:
                avail_oth = np.where(y_ne.to_numpy() & ~used)[0]
                if len(avail_oth) < n_oth:
                    raise ValueError(
                        f"disjoint=True: partition {k} needs {n_oth} unused rows not equal to {maj_label!r}, "
                        f"only {len(avail_oth)} available."
                    )
                pick_o = rng_k.choice(avail_oth, size=n_oth, replace=False)
                used[pick_o] = True
                ix = np.concatenate([pick_m, pick_o])
            rng_k.shuffle(ix)
            part = df.iloc[ix].reset_index(drop=True)
            parts.append(part)
            continue

        pool_maj = df[y_series == maj_label]
        pool_oth = df[y_series != maj_label]
        if len(pool_maj) < n_maj:
            part_m = pool_maj.sample(n=n_maj, replace=True, random_state=rs_m)
        else:
            part_m = pool_maj.sample(n=n_maj, replace=False, random_state=rs_m)
        if n_oth == 0:
            part_o = pool_oth.iloc[0:0]
        elif len(pool_oth) < n_oth:
            part_o = pool_oth.sample(n=n_oth, replace=True, random_state=rs_o)
        else:
            part_o = pool_oth.sample(n=n_oth, replace=False, random_state=rs_o)
        part = pd.concat([part_m, part_o], axis=0)
        part = part.sample(frac=1.0, random_state=rs_sh).reset_index(drop=True)
        parts.append(part)
    return parts


def write_skewed_demo_partitions(
    output_dir: str | Path,
    num_clients: int,
    *,
    majority_fraction: float = 0.9,
    seed: int = 42,
    disjoint: bool = True,
) -> list[Path]:
    """Synthetic data + label-skewed ``node_i.csv`` (even nodes → majority 0, odd → majority 1)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = make_synthetic_network_logs(seed=seed)
    parts = partition_label_skewed(
        df,
        None,
        num_clients,
        majority_fraction=majority_fraction,
        seed=seed,
        disjoint=disjoint,
    )
    paths: list[Path] = []
    for i, part in enumerate(parts):
        p = out / f"node_{i}.csv"
        part.to_csv(p, index=False)
        paths.append(p)
    return paths


def benchmark_row_classification_throughput(
    model: Any,
    X: np.ndarray,
    *,
    warmup: int = 5,
    duration_sec: float = 2.0,
) -> dict[str, float]:
    """
    Measure how many table rows per second ``model.predict`` sustains (IDS "packet" proxy).

    Each row is treated as one flow/packet feature vector for PPS-style reporting.
    """
    n = int(X.shape[0])
    for _ in range(warmup):
        model.predict(X)
    t0 = time.perf_counter()
    batches = 0
    while time.perf_counter() - t0 < duration_sec:
        model.predict(X)
        batches += 1
    elapsed = time.perf_counter() - t0
    rows = batches * n
    return {
        "rows_per_sec": float(rows / elapsed),
        "pps_equivalent": float(rows / elapsed),
        "batch_rows": float(n),
        "wall_sec": float(elapsed),
        "batches": float(batches),
    }


def make_synthetic_network_logs(
    n_samples: int = 4000,
    n_features: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic numeric "flow" features + binary label (1 = attack-like).
    Lets you run the federation without downloading a public dataset.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    logits = 0.9 * X[:, 0] + 0.5 * X[:, 1] + 0.2 * rng.standard_normal(n_samples)
    y = (logits > 0).astype(np.int64)
    # Balance classes so label-skewed *disjoint* partitions can satisfy per-node majority quotas.
    ix0 = np.flatnonzero(y == 0)
    ix1 = np.flatnonzero(y == 1)
    m = min(ix0.size, ix1.size)
    if m == 0:
        cols = [f"feat_{i}" for i in range(n_features)]
        return pd.DataFrame(X, columns=cols).assign(label=y)
    rng.shuffle(ix0)
    rng.shuffle(ix1)
    keep = np.concatenate([ix0[:m], ix1[:m]])
    rng.shuffle(keep)
    X = X[keep]
    y = y[keep]
    cols = [f"feat_{i}" for i in range(n_features)]
    out = pd.DataFrame(X, columns=cols)
    out["label"] = y
    return out


def write_demo_partitions(output_dir: str | Path, num_clients: int, seed: int = 42) -> list[Path]:
    """Create synthetic data and write one CSV per virtual node under ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = make_synthetic_network_logs(seed=seed)
    parts = iid_partition_rows(df, num_clients, seed=seed)
    paths: list[Path] = []
    for i, part in enumerate(parts):
        p = out / f"node_{i}.csv"
        part.to_csv(p, index=False)
        paths.append(p)
    return paths


def sklearn_logistic_params(model: Any) -> list[np.ndarray]:
    """Serialize LogisticRegression weights for Flower (NumPy lists)."""
    # coef_: (n_classes, n_features) or (1, n_features) for binary
    # intercept_: (n_classes,) or (1,)
    return [np.asarray(model.coef_, dtype=np.float64), np.asarray(model.intercept_, dtype=np.float64)]


@dataclass
class NetworkTracker:
    """
    Lightweight "systems" telemetry for federated clients.

    Tracks approximate **uplink payload size** (sum of ``ndarray.nbytes`` for tensors
    you return to Flower) and **process CPU time** spent inside training blocks
    (``time.process_time()``, so it counts on-CPU work for this process, not wall
    idle). gRPC framing overhead is not included in byte counts.

    CPU time is split into **initial** (constructor cold-start fit) vs **federated**
    (``fit`` rounds) when using ``training_timer(phase=...)``.
    """

    bytes_uplink_total: int = 0
    # Approximate WAN savings assuming each internal dataset would have uploaded
    # one model update of identical tensor shapes.
    wan_bytes_saved_total: int = 0
    seconds_training_cpu_total: float = 0.0
    seconds_training_cpu_initial: float = 0.0
    seconds_training_cpu_federated: float = 0.0

    def nbytes_of_arrays(self, arrays: list[np.ndarray]) -> int:
        return sum(int(np.asarray(a).nbytes) for a in arrays)

    def record_uplink(self, arrays: list[np.ndarray]) -> int:
        n = self.nbytes_of_arrays(arrays)
        self.bytes_uplink_total += n
        return n

    def record_wan_bytes_saved(self, internal_clients: int, arrays: list[np.ndarray]) -> int:
        """
        Track approximate WAN bytes saved by aggregating internally at Tier 1.

        If `internal_clients` separate models would have been uploaded independently,
        only a single aggregated model is uploaded from this gateway.
        """
        if internal_clients <= 1:
            return 0
        n = self.nbytes_of_arrays(arrays)
        saved = int((internal_clients - 1) * n)
        self.wan_bytes_saved_total += saved
        return saved

    @contextmanager
    def training_timer(self, phase: Literal["initial", "federated"] = "federated") -> Iterator[None]:
        t0 = time.process_time()
        try:
            yield
        finally:
            dt = time.process_time() - t0
            self.seconds_training_cpu_total += dt
            if phase == "initial":
                self.seconds_training_cpu_initial += dt
            else:
                self.seconds_training_cpu_federated += dt


@dataclass
class SyncedFeatureSpace:
    """
    Global schema for categoricals + target + scaler so all nodes share encodings.

    Build once from **all** partition CSV paths (or equivalent dataframes), save to
    JSON, then each node loads the same file so e.g. ``'UDP'`` and ``'TCP'`` map to
    consistent integers everywhere.

    ``categorical_oov`` controls unknown categorical values at transform time:
    ``"error"`` raises; ``"zero"`` maps to class index 0 (legacy behaviour).
    """

    label_column: str
    feature_columns: list[str]
    categorical_columns: list[str]
    encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    target_encoder: LabelEncoder | None = None
    scaler: StandardScaler = field(default_factory=StandardScaler)
    categorical_oov: Literal["error", "zero"] = "error"

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Apply global encodings + scaling to a single node's dataframe."""
        if self.label_column not in df.columns:
            raise ValueError(f"Missing label column {self.label_column!r}")
        feature_df = df[self.feature_columns].copy()
        y_raw = df[self.label_column]

        for col in self.categorical_columns:
            le = self.encoders[col]
            classes = set(le.classes_.tolist())

            def _map_one(val: object) -> int:
                s = str(val)
                if s in classes:
                    return int(le.transform([s])[0])
                if self.categorical_oov == "zero":
                    return 0
                raise ValueError(
                    f"SyncedFeatureSpace: out-of-vocabulary categorical value {s!r} in column {col!r}. "
                    "Rebuild the schema from data that includes this level, set categorical_oov='zero', "
                    "or add an explicit OOV bucket to the encoder."
                )

            feature_df[col] = feature_df[col].map(_map_one)

        X_num = feature_df.to_numpy(dtype=np.float64)
        X = self.scaler.transform(X_num)

        if self.target_encoder is not None:
            y = self.target_encoder.transform(np.asarray(y_raw).astype(str).ravel())
        else:
            y = np.asarray(y_raw, dtype=np.int64).ravel()

        return X, y.astype(np.int64)

    def to_json_dict(self) -> dict[str, Any]:
        enc_map = {k: list(v.classes_) for k, v in self.encoders.items()}
        tgt = list(self.target_encoder.classes_) if self.target_encoder else None
        return {
            "label_column": self.label_column,
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "encoders": enc_map,
            "target_classes": tgt,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "categorical_oov": self.categorical_oov,
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> SyncedFeatureSpace:
        encoders: dict[str, LabelEncoder] = {}
        for col, classes in d["encoders"].items():
            le = LabelEncoder()
            le.classes_ = np.asarray(classes, dtype=object)
            encoders[col] = le
        tgt_enc: LabelEncoder | None = None
        if d.get("target_classes") is not None:
            tgt_enc = LabelEncoder()
            tgt_enc.classes_ = np.asarray(d["target_classes"], dtype=object)
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(d["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.asarray(d["scaler_scale"], dtype=np.float64)
        scaler.n_features_in_ = int(len(scaler.mean_))
        scaler.feature_names_in_ = None
        oov_raw = d.get("categorical_oov", "zero")
        oov: Literal["error", "zero"] = "zero" if oov_raw not in ("error", "zero") else oov_raw
        return cls(
            label_column=d["label_column"],
            feature_columns=list(d["feature_columns"]),
            categorical_columns=list(d["categorical_columns"]),
            encoders=encoders,
            target_encoder=tgt_enc,
            scaler=scaler,
            categorical_oov=oov,
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> SyncedFeatureSpace:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_json_dict(d)


def sync_feature_space_from_csvs(
    csv_paths: Sequence[str | Path],
    label_column: str | None = None,
    *,
    categorical_oov: Literal["error", "zero"] = "error",
) -> SyncedFeatureSpace:
    """
    Scan every partition CSV, union categorical levels, fit a global ``StandardScaler``.

    All files must share the same column layout. Numeric columns are scaled using
    global mean/variance after categorical columns are label-encoded with a **shared**
    vocabulary per column.

    String/category labels and **numeric** labels both get a global ``LabelEncoder``
    (numeric levels are encoded via stable string forms of the sorted union of
    values seen across all files).
    """
    paths = [Path(p) for p in csv_paths]
    if not paths:
        raise ValueError("sync_feature_space_from_csvs: need at least one CSV path")

    dfs = [load_network_csv(p) for p in paths]
    label_col = infer_label_column(dfs[0], label_column)
    for df in dfs[1:]:
        infer_label_column(df, label_col)

    feature_cols = [c for c in dfs[0].columns if c != label_col]
    for df in dfs[1:]:
        other = [c for c in df.columns if c != label_col]
        if other != feature_cols:
            raise ValueError("All CSVs must share the same feature columns (order and names).")

    cat_cols = [
        c
        for c in feature_cols
        if dfs[0][c].dtype == object or str(dfs[0][c].dtype) == "category"
    ]

    union_vals: dict[str, set[str]] = {c: set() for c in cat_cols}
    target_vals: set[str] = set()
    target_numeric_union: set[str] = set()

    y0 = dfs[0][label_col]
    string_like_target = y0.dtype == object or str(y0.dtype) == "category"

    for df in dfs:
        feat = df[feature_cols].copy()
        for c in cat_cols:
            union_vals[c].update(feat[c].astype(str).unique().tolist())
        y_raw = df[label_col]
        if string_like_target:
            target_vals.update(y_raw.astype(str).unique().tolist())
        else:
            for v in np.unique(y_raw.to_numpy()):
                target_numeric_union.add(str(v))

    encoders: dict[str, LabelEncoder] = {}
    for c in cat_cols:
        le = LabelEncoder()
        le.fit(sorted(union_vals[c]))
        encoders[c] = le

    target_encoder: LabelEncoder | None = None
    if string_like_target and target_vals:
        target_encoder = LabelEncoder()
        target_encoder.fit(sorted(target_vals))
    elif not string_like_target and target_numeric_union:
        target_encoder = LabelEncoder()
        target_encoder.fit(sorted(target_numeric_union))

    encoded_rows: list[np.ndarray] = []
    for df in dfs:
        feat = df[feature_cols].copy()
        for c in cat_cols:
            le = encoders[c]
            feat[c] = le.transform(feat[c].astype(str))
        encoded_rows.append(feat.to_numpy(dtype=np.float64))

    scaler = StandardScaler()
    scaler.fit(np.vstack(encoded_rows))

    return SyncedFeatureSpace(
        label_column=label_col,
        feature_columns=feature_cols,
        categorical_columns=cat_cols,
        encoders=encoders,
        target_encoder=target_encoder,
        scaler=scaler,
        categorical_oov=categorical_oov,
    )


def set_sklearn_logistic_params(model: Any, params: list[np.ndarray]) -> None:
    """Load weights aggregated on the server back into a local LogisticRegression."""
    coef = np.asarray(params[0], dtype=np.float64)
    intercept = np.asarray(params[1], dtype=np.float64).reshape(-1)
    model.coef_ = coef
    model.intercept_ = intercept
    n_out = coef.shape[0]
    model.classes_ = np.arange(n_out) if n_out > 1 else np.array([0, 1], dtype=np.int64)
    model.n_features_in_ = int(coef.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Log-Sieve data utilities.")
    sub = parser.add_subparsers(dest="cmd")

    p_demo = sub.add_parser("demo", help="Write synthetic IID node_*.csv partitions.")
    p_demo.add_argument("--output-dir", default="data_partitions")
    p_demo.add_argument("--num-clients", type=int, default=3)
    p_demo.add_argument("--seed", type=int, default=42)

    p_schema = sub.add_parser("schema", help="Build global feature schema JSON from partition CSVs.")
    p_schema.add_argument("--out", required=True, help="Output path, e.g. global_schema.json")
    p_schema.add_argument(
        "csv_paths",
        nargs="+",
        help="All node CSV paths used to union categoricals and fit global scaling.",
    )
    p_schema.add_argument("--label-column", default=None)
    p_schema.add_argument(
        "--categorical-oov",
        choices=("error", "zero"),
        default="error",
        help="How to handle unknown categoricals at transform time (stored in JSON).",
    )

    p_skew = sub.add_parser("skewed-demo", help="Synthetic label-skewed partitions (non-IID).")
    p_skew.add_argument("--output-dir", default="data_partitions_skewed")
    p_skew.add_argument("--num-clients", type=int, default=2)
    p_skew.add_argument("--majority-fraction", type=float, default=0.9)
    p_skew.add_argument("--seed", type=int, default=42)
    p_skew.add_argument(
        "--overlap",
        action="store_true",
        help="Allow the same row in multiple partitions (legacy IID-style oversampling).",
    )

    args = parser.parse_args()
    if args.cmd == "schema":
        sf = sync_feature_space_from_csvs(
            args.csv_paths, args.label_column, categorical_oov=args.categorical_oov
        )
        sf.save_json(args.out)
        print(f"Wrote schema to {args.out}")
        return
    if args.cmd == "skewed-demo":
        paths = write_skewed_demo_partitions(
            args.output_dir,
            args.num_clients,
            majority_fraction=args.majority_fraction,
            seed=args.seed,
            disjoint=not args.overlap,
        )
        print("Wrote label-skewed partitions:")
        for p in paths:
            print(f"  {p}")
        return
    if args.cmd == "demo" or args.cmd is None:
        if args.cmd is None:
            parser.print_help()
            return
        paths = write_demo_partitions(args.output_dir, args.num_clients, seed=args.seed)
        print("Wrote partitions:")
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
