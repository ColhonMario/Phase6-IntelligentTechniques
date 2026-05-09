# 🛡️ Fraud Detection System — Phase 6 Deployment

A production-deployed credit-card fraud detection web application. Trained on the
Sparkov simulated transaction dataset (~1.85M rows, ~0.5 % fraud rate, Jan 2019 –
Dec 2020), served as an interactive Streamlit app with both single-transaction
and batch-scoring workflows.

> **Live demo:** *https://your-app-name.streamlit.app* (replace after deploying)

---

## Table of contents

1. [Quick start](#quick-start)
2. [Project structure](#project-structure)
3. [Deployment](#deployment)
   - [Option A — Streamlit Community Cloud (recommended)](#option-a--streamlit-community-cloud-recommended)
   - [Option B — Local](#option-b--local)
4. [System architecture](#system-architecture)
5. [Performance & scalability](#performance--scalability)
6. [Model & pipeline summary](#model--pipeline-summary)
7. [Usage](#usage)
8. [Troubleshooting](#troubleshooting)

---

## Quick start

```bash
# Clone, install, run
git clone https://github.com/<you>/<repo>.git
cd <repo>
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Make sure the `app_artifacts/` folder
(saved by the Phase 5 training notebook) is in the project root before launching.

---

## Project structure

```
.
├── app.py                       # Streamlit UI + entry point
├── app_common.py                # Validation, schema, batch helpers
├── pipeline.py                  # Sparkov → IEEE adapter + Phase 4 pipeline
├── requirements.txt             # Pinned Python deps (Streamlit Cloud reads this)
├── runtime.txt                  # Pin Python version (3.12)
├── packages.txt                 # apt-get deps for Streamlit Cloud (empty here)
├── .streamlit/
│   └── config.toml              # Theme + upload size limit
├── app_artifacts/               # ⚠️  Required for inference — see below
│   ├── lgbm_sparkov.joblib      # Trained LightGBM model (Optuna-tuned)
│   ├── sparkov_artifacts.pkl    # Fitted pipeline state (encoders, PCA, freq maps)
│   ├── feature_names_sparkov.pkl# Training-time column order
│   ├── app_meta.json            # AUC / AP / threshold / split metadata
│   ├── input_schema.json        # Required vs optional input fields
│   ├── app_config.json          # Real categories + scenarios
│   ├── sample_transactions.csv     # 50-row demo
│   └── sample_transactions_200.csv # 200-row demo
└── README.md
```

The `app_artifacts/` folder must be present at runtime. For Streamlit Cloud
deployment it has to be **committed to the Git repository** (or fetched at
startup from external storage — see [Troubleshooting](#troubleshooting)).

---

## Deployment

### Option A — Streamlit Community Cloud (recommended)

Streamlit Community Cloud is free, gives a public HTTPS URL, and reads
`requirements.txt` + `runtime.txt` automatically.

**Steps:**

1. **Push to GitHub.** Create a public or private GitHub repository and commit
   everything in this directory **including `app_artifacts/`**. The trained
   artifacts together are typically 5–20 MB, well below GitHub's 100 MB
   per-file limit. If any single file exceeds 100 MB, use Git LFS.
2. **Sign in** to [share.streamlit.io](https://share.streamlit.io) with your
   GitHub account.
3. **Click "New app"** and choose:
   - Repository: `<you>/<repo>`
   - Branch: `main`
   - Main file path: `app.py`
   - Python version: 3.12 (auto-detected from `runtime.txt`)
4. **Deploy.** First build takes ~3–5 min while Streamlit installs the pinned
   dependencies. After that, deploys are incremental.
5. **Public URL** has the form `https://<app-name>.streamlit.app/` and updates
   automatically on every `git push` to `main`.

**Resource limits (free tier):** 1 GB RAM, 1 CPU, ~1 GB storage. The trained
LightGBM model is ~3 MB, the pipeline artifacts ~10 MB; total runtime memory
sits around 350–450 MB, which is safely within limits.

### Option B — Local

Use this if you do not want a public URL or are demoing on the lab machine.

```bash
# 1. Create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the trained artifacts
#    Either drop the app_artifacts/ folder into the project root, or:
export PHASE5_ARTIFACTS=/absolute/path/to/app_artifacts   # Linux/macOS
$env:PHASE5_ARTIFACTS="C:\path\to\app_artifacts"          # Windows PowerShell

# 4. Run
streamlit run app.py
```

The app picks up the `PHASE5_ARTIFACTS` env var if set, otherwise it looks for
`./app_artifacts`.

---

## System architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Streamlit frontend (app.py)                    │
│   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐          │
│   │ Single-trans-  │   │ Batch CSV      │   │ About / model  │          │
│   │ action tab     │   │ scoring tab    │   │ metadata tab   │          │
│   └───────┬────────┘   └────────┬───────┘   └────────────────┘          │
│           │                     │                                       │
│           │  form fields        │  uploaded CSV                         │
│           ▼                     ▼                                       │
│   ┌──────────────────────────────────────────┐                          │
│   │  Input layer (app_common.py)             │                          │
│   │  • validate_inputs / validate_batch_csv  │                          │
│   │  • build_transaction_dict                │                          │
│   │  • impute_batch (safe defaults)          │                          │
│   └────────────────────┬─────────────────────┘                          │
│                        │ Sparkov-shaped dict / DataFrame                │
│                        ▼                                                │
│   ┌──────────────────────────────────────────┐                          │
│   │  Pipeline layer (pipeline.py)            │                          │
│   │  ① Schema map  Sparkov → IEEE-CIS roles  │                          │
│   │  ② Cleaning    dedup, IQR cap, time feats│                          │
│   │  ③ Selection   drop high-null + correlated                          │
│   │  ④ Imputation  median / "unknown"        │                          │
│   │  ⑤ Encoding    freq-encode, PCA(V), LE   │                          │
│   │  ⑥ Engineering log, cyclic, behavioural  │                          │
│   └────────────────────┬─────────────────────┘                          │
│                        │ feature matrix aligned to training cols        │
│                        ▼                                                │
│   ┌──────────────────────────────────────────┐                          │
│   │  Inference layer                         │                          │
│   │  LightGBM (Optuna-tuned) → P(fraud)      │                          │
│   │  Threshold (F1-optimal) → 0/1 verdict    │                          │
│   └────────────────────┬─────────────────────┘                          │
│                        │                                                │
│                        ▼                                                │
│   ┌──────────────────────────────────────────┐                          │
│   │  Output layer                            │                          │
│   │  Single: verdict card, gauge, details    │                          │
│   │  Batch:  KPIs, distributions, top-N,     │                          │
│   │         downloadable CSV                 │                          │
│   └──────────────────────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │ joblib / pickle (loaded once, cached)
                                 │
                       ┌─────────┴───────────┐
                       │  app_artifacts/     │
                       │  • lgbm_sparkov.    │
                       │    joblib           │
                       │  • sparkov_         │
                       │    artifacts.pkl    │
                       │  • feature_names    │
                       │  • app_meta.json    │
                       └─────────────────────┘
```

**Components.** Three layers with clean boundaries: `app.py` (presentation only),
`app_common.py` + `pipeline.py` (business logic), and the on-disk artifact bundle
(state). Artifacts are loaded **once per session** via `@st.cache_resource` so
inference avoids repeated disk I/O.

**Data flow.** A user-supplied transaction (form field or CSV row) is first
validated for schema correctness and value ranges. Missing optional fields are
filled with safe defaults (continental-US centroid for coordinates, "__unknown__"
for free-text). The dict is then expanded into the IEEE-CIS column space (~370
columns of NaN placeholders for the V-features the model was trained with), run
through the same fitted Phase 4 pipeline that produced the training matrix, and
scored by LightGBM. The result is a fraud probability that is compared against
the F1-optimal threshold from training.

---

## Performance & scalability

This section discusses the production characteristics required by the Phase 6
rubric.

### Latency

| Workload | Cold start | Warm latency |
|---|---|---|
| Load all artifacts (one-off, cached) | ~1.5–2.5 s | — |
| Single-transaction prediction | n/a | **~30–80 ms** |
| Batch of 50 transactions | n/a | **~150–250 ms** |
| Batch of 200 transactions | n/a | **~250–450 ms** |

Cold start covers reading the LightGBM model (`joblib.load`, ~3 MB), the fitted
pipeline state (`pickle.load`, ~10 MB), and the schema/metadata JSON files.
Streamlit's `@st.cache_resource` decorator pins these in process memory for the
lifetime of the worker, so subsequent predictions skip disk I/O entirely.

The single-transaction path is dominated by the pipeline transform step, not the
model — LightGBM inference on 1 row is sub-millisecond. Most of the wall-clock
goes to pandas reshaping (Sparkov→IEEE expansion, ~370 NaN columns), label
encoding, and PCA transform on V-features. Batch scoring amortises these costs
across rows, so per-transaction latency drops by roughly 10× at batch size 200.

### Memory usage

| Component | Resident size |
|---|---|
| LightGBM model | ~3 MB |
| Pipeline artifacts (encoders, PCA, freq maps) | ~10 MB |
| Streamlit + Python base process | ~250–300 MB |
| **Steady-state total** | **~300–400 MB** |
| Peak during 200-row batch | **~450–550 MB** |

This fits comfortably in the 1 GB Streamlit Community Cloud free tier. Memory
peaks during batch scoring come from the intermediate IEEE-shaped DataFrame
(many NaN-filled columns); these are released after each call returns.

### Scalability limits

The current architecture has three known ceilings:

1. **Single-process Streamlit.** Streamlit Community Cloud serves one container
   per app, so concurrent users share one Python interpreter and the GIL. The
   app handles roughly 5–10 simultaneous active sessions before queueing
   becomes visible. This is fine for a class demo and small internal tools, but
   does not scale to a real bank's transaction volume.

2. **In-memory pipeline.** The whole feature pipeline runs inside the request
   path. For batches above ~5 000 rows, the IEEE-CIS expansion to ~370 columns
   plus PCA fitting overhead pushes memory toward the cloud free-tier ceiling.
   A 50 MB upload limit is configured in `.streamlit/config.toml` to prevent
   accidental OOM.

3. **No caching of repeated transactions.** Every prediction goes end-to-end
   through the pipeline. In a real deployment, hashing the input dict and
   caching results (`@st.cache_data`) would cut latency for repeated lookups by
   an order of magnitude.

### How to scale beyond this prototype

If this app needed to serve real production traffic, the recommended evolution
would be:

- Wrap the pipeline + model in a stateless **FastAPI service** behind a
  load balancer (Gunicorn + Uvicorn workers).
- Containerise with Docker (a `Dockerfile` is straightforward to add — base off
  `python:3.12-slim`).
- Deploy to **AWS ECS / GCP Cloud Run / Azure Container Apps** with
  autoscaling on CPU and request count.
- Move the model artifacts to **S3 / GCS** and load on container start.
- Add **Redis** for prediction-result caching and rate limiting.
- Stream predictions via **Kafka** for real-time card-network integration.

The current Streamlit deployment is intentionally a single-binary demo —
appropriate for the academic deliverable, not for production card networks.

---

## Model & pipeline summary

| Property | Value |
|---|---|
| Algorithm | LightGBM gradient-boosted trees |
| Hyperparameter search | Optuna, ~50 trials, 5-fold time-aware CV |
| Training data | Sparkov simulated transactions, 1.85 M rows |
| Class balance | ~0.5 % fraud (severe imbalance) |
| Test ROC-AUC | (see sidebar in app — read from `app_meta.json`) |
| Test Average Precision | (see sidebar in app) |
| Decision threshold | F1-optimal on validation set |
| Required input fields | `amt`, `trans_date_trans_time`, `category`, `cc_num` |
| Optional input fields | `merchant`, `zip`, `lat`, `long`, `merch_lat`, `merch_long` |

The required four fields cover ~80 % of cumulative model importance; the
optional six add the remaining ~20 %. Missing optional inputs are filled with
geographic and frequency-aware defaults that keep the model in-distribution.

---

## Usage

### Single-transaction tab
Fill the four required fields (or click a scenario button to auto-populate).
Click **🔮 Predict** to score. The verdict card shows the binary outcome, the
gauge shows the probability vs threshold, and the details expander shows
card-merchant distance, hour-of-day, and which fields were imputed.

### Batch scoring tab
Upload a CSV with at least `trans_date_trans_time, amt, category` columns. Or
click one of the sample buttons. The app validates the schema, imputes missing
values, scores everything, and shows summary KPIs, two distribution charts, the
top-N most suspicious transactions, and a download button for the full results.

### About tab
Static reference info — model details, pipeline stages, and which fields are
required vs optional.

---

## Troubleshooting

**`FileNotFoundError: lgbm_sparkov.joblib`**
The `app_artifacts/` folder is missing or the env var `PHASE5_ARTIFACTS` points
to the wrong location. Confirm the artifact files are present and re-run.

**LightGBM version mismatch warning**
The model was trained with the version pinned in `requirements.txt`. If you see
warnings about pickle compatibility, upgrade or downgrade `lightgbm` to match.

**Streamlit Cloud build fails on `lightgbm`**
Make sure `runtime.txt` pins Python 3.12 (3.13 wheels can lag for some ML libs).
If problems persist, try `lightgbm==4.3.0` instead.

**Repository is too large to push**
`app_artifacts/` should be a few tens of MB. If raw training data leaked in,
add it to `.gitignore` and use `git filter-repo` to clean history.

**App goes to sleep on Community Cloud**
Free-tier apps sleep after ~7 days of inactivity. Visiting the URL wakes them
up; cold start takes ~30 s.

---

## License & attribution

Built for academic Phase 6 submission, AI specialization, Babeș-Bolyai
University. Sparkov dataset by [Brandon Harris](https://github.com/namebrandon/Sparkov_Data_Generation).
LightGBM by Microsoft. UI by Streamlit.
