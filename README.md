# Log-Sieve

**Log-Sieve** is a Python research prototype of a **federated learning (FL)** pipeline for a **network intrusion detection system (IDS)**. Multiple network **nodes** each train a **local** model on **their own traffic logs**. **Raw logs are not sent** to a central server; only **model parameters** (weights) and **scalar metrics** are exchanged with an aggregator.

This document is the **project specification**: goals, stack, architecture, modules, behavior, and how to run the code.

---

## Objectives

| Goal | Description |
|------|-------------|
| **Privacy-preserving training** | Keep CSV rows on the owning node; the center only aggregates numerical updates. |
| **Federated coordination** | Broadcast global model → local training → aggregate → repeat. |
| **IDS-oriented evaluation** | Report **accuracy**, **precision**, and **recall** alongside loss. |
| **Byzantine / poisoning resilience** | Server uses **distance filtering** plus **trimmed-mean** or **Krum** aggregation (not plain “everyone is honest” FedAvg). |
| **Systems-style measurement** | Clients report **approximate uplink tensor bytes** and **process CPU time** during training; optional **quantised / sparse-ish** uploads. |
| **Straggler tolerance** | Configurable **`round_timeout`**: slow clients time out and the round continues with whoever finished. |
| **Non-IID / heterogeneity** | Label-skewed partitions in **`utils.py`**; server **`--strategy fedprox`** with proximal **`μ`**. |
| **DP-style uploads** | Optional **Laplace noise** on outgoing weights (privacy–utility knob, not formal DP guarantee). |
| **Throughput benchmarking** | **`--benchmark-inference`** reports in-memory **rows/sec** (PPS proxy) for `predict`. |
| **Experiment sweeps** | **`run_experiments.py`** grids hyperparameters and writes a **CSV** for plots. |

---

## Stack

| Layer | Technology |
|-------|------------|
| **Federated framework** | [Flower](https://flower.ai/) (`flwr`) — **`LogSieveFedAvg`** / **`LogSieveFedProx`**. |
| **ML model** | [scikit-learn](https://scikit-learn.org/) **`LogisticRegression`** + optional **`fedprox_fit`** (binary GD with proximal term). |
| **Data** | [Pandas](https://pandas.pydata.org/) for CSV I/O; [NumPy](https://numpy.org/) for tensors. |
| **Preprocessing** | `StandardScaler`, `LabelEncoder`; optional **global schema** via `SyncedFeatureSpace`. |
| **Robust aggregation** | `robust_aggregate.py`: Euclidean filter to median, **trimmed mean**, **Krum**. |

Pinned loosely in `requirements.txt` (Flower 1.x line).

---

## Architecture

### High-level control flow

1. **Server** waits for **`min_available_clients`**, then runs **`num_rounds`**.
2. **Initialization**: server may pull initial parameters from one client (`get_parameters`).
3. Each **round**:
   - Server sends **global parameters** to sampled clients (subject to **`round_timeout`**).
   - Clients run **`fit`**: decode global weights, **local** `sklearn` training, optionally **compress** outgoing tensors, return updates + metrics.
   - **`LogSieveFedAvg` / `LogSieveFedProx.aggregate_fit`**:
     - Successful updates are in **`results`**; timeouts/errors are in **`failures`** (ignored when **`accept_failures=True`**).
     - **Distance filter**: drop the **`outlier_frac`** fraction of clients whose flattened update is farthest in L2 from the **coordinate-wise median** update (geometric outlier removal; it does **not** guarantee identifying a specific attacker or a label-flip client—see **Limitations**).
     - **Aggregation**: **trimmed mean** (per coordinate, trim tails), **Krum** (pick one surviving client update), or plain **FedAvg** on the surviving set.
   - **`evaluate`** may run; metrics are **example-weighted** at the server.
4. Run ends; clients disconnect.

### What crosses the network

- **Client → server**: serialised **parameter tensors** (after optional float16 round-trip / sparsification snap), **sample counts**, **metrics** (accuracy, precision, recall, communication / CPU telemetry, etc.).
- **Not sent**: raw CSV rows.

Byte counts on the client sum **`ndarray.nbytes`** for tensors **about to be returned** to Flower — a **lower bound**; gRPC adds framing and metadata overhead.

---

## Repository layout

| Path | Role |
|------|------|
| `server.py` | Flower server: **`LogSieveFedAvg`** / **`LogSieveFedProx`**, **`round_timeout`**, aggregation CLI. |
| `client.py` | **`NumPyClient`**: DP Laplace noise, quantisation (**`float16`/`int8`**), FedProx local solve, **`--benchmark-inference`**. |
| `utils.py` | Data I/O, IID + **label-skew** partitions, **`NetworkTracker`**, **`SyncedFeatureSpace`**, throughput helper. |
| `fedprox_fit.py` | Binary logistic **FedProx** via gradient descent + proximal term. |
| `robust_aggregate.py` | Distance filter, trimmed-mean tensors, Krum, shared **`aggregate_fit`** plumbing. |
| `run_experiments.py` | Grid sweeper → **CSV** (outlier frac × latency × quantize × strategy × DP λ). |
| `simulate_network.sh` | **Linux `tc`** examples to emulate slow vs fast client links (systems experiments). |
| `requirements.txt` | Python dependencies. |

---

## Server: robust aggregation and stragglers

### Distance filter

For each client, flatten all parameter arrays to one vector. Compute the **per-coordinate median** across clients, then each client’s **L2 distance** to that median. Remove the **`--outlier-frac`** fraction of clients with **largest** distances (possible poisoners or bad links). Set **`--outlier-frac 0`** or **`--no-distance-filter`** to disable.

**Implementation details**: With **two or fewer** successful client updates in a round, the filter is a **no-op** (nothing is dropped). The number dropped uses **`ceil(n × outlier_frac)`** farthest clients, so the realised drop count is a discrete approximation to the configured fraction. **Label poisoning** (`--malicious`) affects training on the client; the server only sees **weights**. A poisoned update **may** lie far from the median, but that is **not** guaranteed—do not treat this filter as ground-truth attack detection.

### Trimmed mean (default)

On the remaining clients, for **each scalar in each tensor**, sort values across clients, drop the lowest/highest **`--trim-ratio`** fraction, average the rest (Byzantine-trimmed-mean style). With very few clients, trimming may reduce to the **median** or **mean** fallback inside `robust_aggregate.py`.

### Krum

**`--aggregation krum`** runs **single-Krum** on the surviving vectors (after the distance filter unless disabled). **`--krum-f`** is the Byzantine upper bound `f` (score uses **`n - f - 2`** nearest neighbours by squared L2 distance). Krum returns **one** client’s full update (not an average).

### FedAvg baseline

**`--aggregation fedavg`** applies **sample-weighted averaging** on the post-filter client set (standard FedAvg tensors).

### Round timer

**`--round-timeout SECONDS`** sets `ServerConfig.round_timeout`. When a client exceeds this window, Flower places the attempt in **`failures`**; with **`accept_failures=True`** (default on `LogSieveFedAvg`), the server **aggregates successful `results` only** and the round completes.

Use **`--round-timeout 0`** (or any non-positive value) to **disable** the timeout (wait indefinitely).

### FedProx (non-IID)

**`--strategy fedprox`** uses Flower’s **`FedProx`** with **`LogSieveFedProx`**: the same robust aggregation as above, and each **`FitIns`** includes **`proximal_mu`** (CLI **`--proximal-mu`**, default `0.1`). Clients minimise local loss plus **(μ/2)||w − w_global||²** using **`fedprox_fit.fedprox_binary_logistic_fit`** when labels are **binary**; multi-class falls back to plain **`LogisticRegression.fit`** with a warning.

**`--strategy fedavg`** keeps **`μ = 0`** semantics on the client (no proximal term in the custom solver).

---

## Client: communication efficiency and poisoning switch

| Flag | Effect |
|------|--------|
| **`--quantize float16`** | Round-trip weights through **`float16`** before upload (quantisation error; still NumPy **`float64`** containers for Flower compatibility). |
| **`--sparsify-threshold δ`** | If **`|w - w_global| < δ`** on a coordinate, snap **`w`** back to **`w_global`** so the “update” is zero there (mimics small-magnitude pruning). |
| **`--malicious`** | **Label poisoning**: before training, **invert** the label mapping (sorted string classes reversed; binary numeric **`0↔1`**; multi-class numeric uses reversed sorted unique mapping). The **aggregator never observes labels**; robust aggregation operates on **parameter vectors** only. Experiments should treat distance trimming / trimmed mean / Krum as **heuristic** defences, not proofs that the malicious node was identified. |
| **`--schema-json PATH`** | Load a **`SyncedFeatureSpace`** JSON so all nodes share **categorical encodings** and **global scaling** (see below). |
| **`--schema-categorical-oov`** | Optional override: `error` (raise on unknown category) or `zero` (map unknown to index 0). Defaults to the policy stored in the JSON. |
| **`--dp-laplace-scale λ`** | **Laplace(0, λ)** noise on uploaded weights: *w\_noisy = w + Laplace(0, λ)* (local model untouched). Privacy–utility knob; not full formal DP. |
| **`--dp-seed`** | RNG seed for Laplace noise. |
| **`--quantize int8`** | Per-tensor **min–max style** linear mapping to **127 bins** and back (values still float64 for Flower). This **stress-tests** narrow uploads; it is **not** a calibrated model of fixed-point ASIC/FPGA arithmetic. |

**`NetworkTracker`** (in `utils.py`) records:

- **`comm_uplink_bytes_fit`**, **`comm_uplink_bytes_cumulative`**: nbytes of outgoing parameter arrays (after compression helpers).
- **`train_cpu_seconds_cumulative`**: cumulative **`time.process_time()`** inside training blocks (on-CPU time for this process, not wall-clock sleep or network wait).
- **`train_cpu_seconds_initial`**: CPU time for the **constructor** local `fit` (cold start), when run.
- **`train_cpu_seconds_federated`**: CPU time for **`fit`** rounds only (FedProx path or plain `LogisticRegression.fit`).

These appear in **fit** / **evaluate** metrics and are **weighted** in the server log.

### Inference latency (PPS proxy)

```bash
python client.py --data-path data_partitions/node_0.csv --benchmark-inference
```

Trains a local **`LogisticRegression`** on that partition, then measures how many **table rows per second** repeated **`predict(X)`** achieves over **`--bench-duration`** seconds (after **`--bench-warmup`** batches). The timer wraps **only** the **`predict`** loop (data load and one-time `fit` are outside the timed region). Each row stands in for one flow/feature vector (“packet” in a tabular IDS). This is **in-memory** scoring (no NIC); use it to compare models and hardware, not to claim true **line-rate** NIC forwarding without a packet I/O path.

---

## Global feature alignment (`sync_feature_space`)

**Problem**: if node A sees only **`"UDP"`** and node B only **`"TCP"`**, local `LabelEncoder`s can assign **inconsistent integers**, breaking a global linear model.

**Solution**: build a **`SyncedFeatureSpace`** once from **all** partition CSVs:

```bash
python utils.py schema --out global_schema.json data_partitions/node_0.csv data_partitions/node_1.csv
```

Optional: **`--categorical-oov error`** (default) or **`zero`** — stored in the JSON and used at transform time unless overridden by **`--schema-categorical-oov`** on the client.

Each client then runs with **`--schema-json global_schema.json`**. The schema stores:

- Union of **categorical** levels per column (consistent `LabelEncoder` classes).
- **Target** `LabelEncoder` for **string/category** labels, or for **numeric** labels (sorted union of values across all listed CSVs, encoded via stable string forms).
- **Global** `StandardScaler` **mean** / **scale** fit on **stacked** encoded features from every file.
- **`categorical_oov`** policy (`error` or `zero`). JSON files **without** this key default to **`zero`** when loading (backward compatibility).

All partitions must share the **same feature columns** (names and order).

**Behaviour notes**:

- **Categoricals**: at **transform** time, unknown values **`raise`** when the policy is **`error`** (default for newly built schemas). Use **`zero`** only if you intentionally want unknowns mapped to class index **0** (legacy behaviour).
- **Targets**: numeric labels now use the same **global** `LabelEncoder` as string labels when you build the schema from all partitions.

---

## `utils.py` CLI

| Command | Purpose |
|---------|---------|
| **`python utils.py demo --output-dir DIR --num-clients N`** | Synthetic IID **`node_i.csv`** shards. |
| **`python utils.py skewed-demo ...`** | Synthetic **label-skewed** shards (default: **disjoint** rows across nodes). Pass **`--overlap`** to allow the legacy behaviour (same row may appear on multiple nodes). |
| **`python utils.py schema --out FILE.json ...`** | Emit **`global_schema.json`** from listed CSVs (see **`--categorical-oov`**). |

**`partition_label_skewed(..., disjoint=True)`** assigns each dataframe row to **at most one** partition (no duplicate rows across clients). Feasibility depends on class counts vs. **`majority_fraction`**; use **`disjoint=False`** (or **`--overlap`** on **`skewed-demo`**) if you need the older **with-replacement** sampling. Synthetic **`make_synthetic_network_logs`** balances classes so two-node **0.9** skew is usually feasible.

---

## `run_experiments.py` (automated sweeps)

Builds a Cartesian grid (configurable) and appends one row per run to a CSV (default **`experiment_results.csv`**).

- **Factors**: `outlier_frac` ∈ {0.1, 0.2, 0.4}, nominal **`configured_netem_delay_ms`** ∈ {0, 10, 100, 500}, **`quantize`** ∈ {none, float16, int8}, **`strategy`** ∈ {(fedavg,0), (fedprox,0.1)}, **`--dp-scales`** (comma-separated λ list, default `0`).
- **Procedure**: spawn **`server.py`** + two **`client.py`** processes; parse the server log for final **accuracy** and **loss**.
- **`--quick`**: one configuration for smoke testing.
- **`--tc-iface IFACE`** (Linux + sudo): runs **`simulate_network.sh apply`** with that delay before each run when **`configured_netem_delay_ms` > 0**; if delay is **0**, `tc` is skipped (delay is still **metadata** for plots). Sweeps use a fixed **`--aggregation trimmed_mean`** and **`--round-timeout 120`**; they do not sweep Krum/FedAvg server modes unless you edit the script.

Prerequisite: **`python utils.py skewed-demo --output-dir data_partitions_skewed --num-clients 2`** (or point **`--data-dir`** at your own **`node_0.csv` / `node_1.csv`**).

```bash
python run_experiments.py --output-csv results_grid.csv --rounds 3
python run_experiments.py --quick
```

---

## `simulate_network.sh` (Linux `tc`)

Bash helper to attach **HTB** rate limits and **netem** delay to a **specific interface** (use a **veth** or container NIC — avoid throttling your SSH link).

```bash
chmod +x simulate_network.sh
sudo ./simulate_network.sh apply veth-slow 10mbit 80ms
sudo ./simulate_network.sh apply veth-fast 1000mbit 2ms
# ... run Log-Sieve ...
sudo ./simulate_network.sh clear veth-slow
sudo ./simulate_network.sh clear veth-fast
```

Combine with **`--round-timeout`** so a **slow straggler** does not block convergence experiments.

**Windows**: use **WSL2**, a **Linux VM**, or **Docker** on Linux hosts; native Windows does not ship `tc`.

---

## Model choice

**`LogisticRegression`** exposes **`coef_`** and **`intercept_`**, which pair naturally with **vector aggregation**. **`RandomForestClassifier`** does not map cleanly to vanilla FedAvg; tree FL needs specialised protocols.

---

## Installation

```bash
pip install -r requirements.txt
```

Python **3.10+** recommended.

---

## Running a federated job

1. **Optional demo data**

   ```bash
   python utils.py demo --output-dir data_partitions --num-clients 2
   ```

2. **Optional global schema** (for categorical alignment)

   ```bash
   python utils.py schema --out global_schema.json data_partitions/node_0.csv data_partitions/node_1.csv
   ```

3. **Server**

   ```bash
   python server.py --host 0.0.0.0 --port 8080 --rounds 5 --min-clients 2 \
     --round-timeout 300 --aggregation trimmed_mean --outlier-frac 0.1 --trim-ratio 0.1
   ```

   **FedProx + non-IID data** (after `skewed-demo`):

   ```bash
   python server.py ... --strategy fedprox --proximal-mu 0.1
   ```

   **Krum example**: `--aggregation krum --krum-f 1`

4. **Clients** (one terminal each; add **`--schema-json`** if you built one)

   ```bash
   python client.py --server-address 127.0.0.1:8080 --data-path data_partitions/node_0.csv \
     --client-id site-A --quantize float16 --sparsify-threshold 0.001

   python client.py --server-address 127.0.0.1:8080 --data-path data_partitions/node_1.csv \
     --client-id site-B --schema-json global_schema.json
   ```

   **DP noise on uploads** (privacy–utility trade-off):

   ```bash
   python client.py ... --dp-laplace-scale 0.01 --dp-seed 0
   ```

5. **Malicious node** (poisoning experiment)

   ```bash
   python client.py ... --malicious
   ```

---

## Flower runtime note

On **Flower 1.27+**, `start_server` / `start_client` may print **deprecation** notices (SuperLink / SuperNode). This repo keeps simple **`python server.py`** / **`python client.py`** entrypoints for research. See the [Flower documentation](https://flower.ai/docs/) for production deployment.

---

## Hierarchical Deployment (Tier 1/Tier 2)

Log-Sieve supports a simple Tier 1 / Tier 2 Hierarchical Federated Learning (HFL) layout to address “Cloud Segmentation” scaling: multiple internal sensors inside the same company (e.g., EC2 instances in one VPC) keep raw logs locally, while a single gateway uploads only one aggregated model update to the global Tier 2 server.

Tier 2 (global): run the existing `server.py` as before (e.g., `--strategy fedprox` or `--strategy fedavg`), aggregating the single updates it receives from each company gateway.

Tier 1 (company gateway): run `gateway_client.py` instead of `client.py`. The gateway accepts a list of internal CSVs via `--data-paths` (for example, one CSV per sensor), trains a separate local `LogisticRegression` per CSV, and then computes a sample-weighted FedAvg across those internal models to form a single “Company Rulebook”. Only this single averaged weight update is uploaded to Tier 2 along with the combined sample count.

WAN bandwidth benefit: the gateway uploads one parameter vector per round instead of one per internal dataset. The code tracks the approximate avoided WAN payload size in the new metric `wan_bytes_saved` (logged by `NetworkTracker`), computed as `(N_internal - 1) * size(one upload)`.

Security boundary benefit: because the gateway aggregates intra-company models locally, raw traffic rows are never sent to Tier 2, preserving company VPC / tenant isolation while still enabling global coordination.

Example (two gateways, each aggregating two internal sensors):

```bash
python server.py --port 8080 --rounds 5 --min-clients 2 --strategy fedprox --proximal-mu 0.1
python gateway_client.py --server-address 127.0.0.1:8080 --data-paths data_partitions/node_0.csv data_partitions/node_1.csv --client-id company-A
python gateway_client.py --server-address 127.0.0.1:8080 --data-paths data_partitions_skewed/node_0.csv data_partitions_skewed/node_1.csv --client-id company-B
```

---

## Limitations (explicit)

- **Prototype only**: no TLS or auth.  
- **Evaluation** defaults to the **same local partition** used for training unless you add a hold-out split.  
- **Uplink byte counts** omit protobuf overhead.  
- **`float16` / `int8` quantisation** still uses float64 containers for Flower; **`float16`** approximates narrow-float uploads; **`int8`** is a **linear binning** stressor, not a calibrated hardware quantisation model.  
- **Laplace noise** is added only to **uploaded** tensors (`get_parameters` / `fit` returns); the **local** `sklearn` weights stay unnoised after training. It is a **heuristic** upload mask, not **(ε,δ)-DP** with clipping and composition accounting.  
- **FedProx** custom solver is **binary-only**; multi-class uses standard local fit without the proximal term.  
- **`tc` simulation** is **Linux-only** and requires care not to lock yourself out of the network.  
- **Robust aggregation**: trimmed mean and Krum follow standard **coordinate-wise / single-Krum** constructions in code; the **L2 distance filter** removes geometric outliers and does **not** imply identification of Byzantine nodes or success against **all** poisoning strategies.  
- **SyncedFeatureSpace**: default policy for new schemas is **`error`** on unseen categoricals; legacy JSON without **`categorical_oov`** still defaults to **`zero`** on load. Rebuild schemas when traffic vocabulary drifts.

---

## License and attribution

If you publish work based on Log-Sieve, cite **Flower** and any **datasets** (e.g. UNSW-NB15, CIC-IDS2017) under their terms.
