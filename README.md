# Large Language Model-Based Context-Rich Solar Energy Forecasting: A Novel Multimodal Symbolic-Numeric Approach
_A multimodal symbolic–numeric framework for PV power prediction_

---

## 1. Overview

This repository contains the full codebase for the paper:

> **“Large Language Model-Based Context-Rich Renewable Energy Forecasting:  
> A Novel Multimodal Symbolic–Numeric Approach”**

The work addresses a key gap in photovoltaic (PV) forecasting: **operational
context** (e.g., maintenance logs, faults, soiling, curtailment, sensor issues)
is almost never incorporated into data-driven power prediction pipelines.
Existing models typically rely only on structured numerical data such as solar
generation and weather, which limits their ability to anticipate behaviour
during anomalous or degraded conditions.

We propose a **multimodal forecasting framework** that fuses:

- Numerical time-series data (solar generation + weather covariates), and  
- **Textual maintenance logs**, encoded as 384-dimensional sentence embeddings
  using the Sentence-BERT MiniLM model (`all-MiniLM-L6-v2`).

The framework is evaluated on:

- **Dataset 1** – multi-year, multi-campus PV dataset (15-min resolution),  
- **Dataset 2** – utility-scale PV plant (hourly MW-scale data).

We provide **leakage-free synthetic maintenance logs**, multimodal dataset
construction, and forecasting experiments using LightGBM, XGBoost, CatBoost,
and MLP across multiple configuration cases.

---

## 2. Problem Statement

- PV forecasting pipelines usually treat the problem as purely numerical:
  `PV(t+h) = f( PV_history, Weather_forecast )`.
- In real systems, however, **maintenance events** (inverter trips, panel
  cleaning, curtailment orders, sensor faults, etc.) strongly affect the
  generation profile.
- These events are documented in **unstructured maintenance logs** (CMMS,
  O&M tickets, notes), which are difficult to integrate directly into
  time-series models.

**Goal:** Design a reproducible and scalable framework that can inject
maintenance-log semantics into PV forecasting, while remaining compatible with
standard ML models and strict causal evaluation protocols.

---

## 3. Proposed Solution

1. **Leakage-free synthetic maintenance logs**

   - Generate realistic but fully synthetic logs from solar + weather data.
   - Enforce **Δ ≥ h** (lag ≥ forecast horizon) so that logs never encode
     same-time or future information about the target.
   - Trigger families:
     - Relative generation drops (fault/curtailment-like behaviour),
     - High temperature (sensor/overheating),
     - High humidity (soiling/cleaning),
     - Low-rate exogenous “routine check” events.
   - Rate controls: target events/day/site, cooldown between logs, type
     balance, monotonic severity → duration mapping.

2. **LLM-based log embeddings**

   - Use Sentence-BERT (`all-MiniLM-L6-v2`) via
     `sentence-transformers` to encode each log description.
   - Apply the default tokenizer and **mean pooling over last-layer token
     embeddings** to obtain a 384-D vector.
   - Store embeddings as `text_emb_mean_0`–`text_emb_mean_383`.

3. **Multimodal dataset construction**

   - Align solar + weather time series to a common timestamp grid.
   - Aggregate logs per site and timestamp using `AvailableToModelAt ≤ t` with
     a trailing window `[t−W, t]` to compute:
     - `logs_count_window`: number of logs in the window,  
     - `logs_recency_min`: time since last log event,  
     - mean-pooled MiniLM embeddings over logs in the window.
   - Create forecasting targets `Target_t_plus_H = SolarGeneration.shift(-H)`.

4. **Forecasting models**

   - **LightGBM, XGBoost, CatBoost, and MLP** are trained under **strict
     time-aware splits**:
     - Rolling-origin expanding-window (main experiments), and
     - Additional K-fold analyses for specific ablations.
   - Quantile LightGBM (q10 / q50 / q90) provides **prediction intervals** and
     **pinball loss** for operational uncertainty.

---

## 4. Data Configuration Cases

We evaluate five configurations to ablate the contribution of each modality:

1. **Case 1 – Solar Generation + Weather Forecasting (Baseline)**  
   - Numerical features: solar generation, weather covariates.

2. **Case 2 – Solar Generation + Text Embedding Logs**  
   - Solar generation + log-derived numeric features
     (`logs_count_window`, `logs_recency_min`), without weather.

3. **Case 3 – Weather Forecasting + Text Embedding Logs**  
   - Weather covariates + log embeddings and log-derived numeric features.

4. **Case 4 – Text Embedding Logs Only**  
   - Log embeddings + log-derived numeric features without solar or weather.

5. **Case 5 – Full Multimodal (Solar + Weather + Text Embedding)**  
   - **Primary configuration**: combines solar, weather, and MiniLM log
     embeddings (plus numeric log features).

All forecasting scripts allow selection of the case either by:

- A command-line argument (e.g., `--case 5`), or  
- A simple configuration flag at the top of the script.

---

## 5. Repository Structure

```text
.
├── data/
│   ├── raw/                        # Original CSVs (solar, weather, etc.)
│   ├── processed/                  # Merged solar+weather, multimodal CSVs
│   └── splits/                     # Published time-based split indices (CSV)
│
├── logs_generation/
│   ├── Synthetic_Maintenance_Logs_LeakageFree_v5.py
│   ├── Synthetic_Maintenance_Logs_LeakageFree_v4.csv  # example output
│   └── LOG_USAGE_PROTOCOL.txt
│
├── embeddings/
│   ├── encode_logs_minilm.py       # MiniLM sentence embedding generation
│   └── Leakage_Free_Embeddings.csv
│
├── multimodal_build/
│   ├── build_multimodal_dataset_d1.py  # builds Dataset 1 multimodal CSV
│   └── build_multimodal_dataset_d2.py  # builds Dataset 2 multimodal CSV
│
├── forecasting/
│   ├── lgbm_caseX.py               # LightGBM experiments for Cases 1–5
│   ├── xgboost_caseX.py            # XGBoost experiments for Cases 1–5
│   ├── catboost_caseX.py           # CatBoost experiments for Cases 1–5
│   ├── mlp_caseX.py                # MLP experiments for Cases 1–5
│   ├── multihorizon_lgbm.py        # 1 h / 6 h / 24 h baseline table (Table 10)
│   └── reviewer_safe_solar_lgbm.py # rolling-origin, persistence vs LGBM
│
├── analysis/
│   ├── plotting_scripts.py         # EDA, reliability plots, interval plots
│   └── tsne_umap_visualizations.py
│
└── README.md
