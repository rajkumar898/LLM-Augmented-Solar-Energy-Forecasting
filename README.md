# Leakage Free Maintenance Logs Generation 

# ===============================================================
# Leakage-free synthetic maintenance logs with realistic rates
# - Causal: uses only past info (Δ ≥ h enforced)
# - Controls event density (target events/day/site, cooldown)
# - Caps type dominance per day/site (balance)
# - Monotonic Severity → Duration mapping
# - Uniqueness on (Timestamp, SiteKey, MaintenanceType, Component)
# ===============================================================

import os, random
from datetime import timedelta
import numpy as np
import pandas as pd

# -------------------- CONFIG --------------------
MERGED_FILE = r"/home/raj/RE Revision/Merged_Solar_Weather.csv"

DATA_FREQ_MINUTES   = 15
HORIZON_STEPS       = 4         # forecast horizon (steps)
LAG_STEPS           = 4         # Δ; must satisfy Δ ≥ h
assert LAG_STEPS >= HORIZON_STEPS, "Set LAG_STEPS >= HORIZON_STEPS to ensure Δ ≥ h"

USE_FIXED_THRESHOLDS = True     # else uses expanding robust thresholds (causal)
GEN_REL_DROP_THRESH  = -0.15    # ≥15% drop between (t-Δ-1) and (t-Δ)
HIGH_TEMP_THRESH     = 40.0     # °C
HIGH_HUM_THRESH      = 90.0     # %

K_MAD                = 2.0      # for expanding thresholds (if enabled)

# ---- Rate & balance controls ----
TARGET_EVENTS_PER_DAY_PER_SITE = 8   # realistic: 5–15/day/site
COOLDOWN_MIN                  = 60   # min minutes between logs per site
TYPE_MAX_SHARE                = 0.50 # ≤ 50% of selected logs/day/site by any one type
RANDOM_BG_RATE                = 0.0005  # ~0.05% of rows as random candidates

OUTPUT_BASENAME = "Synthetic_Maintenance_Logs_LeakageFree_v5.csv"
PROTOCOL_TXT    = "LOG_USAGE_PROTOCOL.txt"
# ------------------------------------------------

# --------------- Load & validate ---------------
if not os.path.exists(MERGED_FILE):
    raise FileNotFoundError(f"Cannot find merged file at: {MERGED_FILE}")

df = pd.read_csv(MERGED_FILE, parse_dates=["Timestamp_Aligned"])
required_cols = ["Timestamp_Aligned", "SolarGeneration", "AirTemperature", "RelativeHumidity"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

if "SiteKey" not in df.columns:
    df["SiteKey"] = "GLOBAL"

df = df.sort_values(["SiteKey", "Timestamp_Aligned"]).reset_index(drop=True)

# --------------- Build causal lags ---------------
df["SolarGeneration_Lag"]  = df["SolarGeneration"].shift(LAG_STEPS)
df["AirTemperature_Lag"]   = df["AirTemperature"].shift(LAG_STEPS)
df["RelativeHumidity_Lag"] = df["RelativeHumidity"].shift(LAG_STEPS)

# Relative change between (t-Δ-1) -> (t-Δ)
df["GenPrev"]        = df["SolarGeneration"].shift(LAG_STEPS + 1)
df["GenRelDrop_Lag"] = (df["SolarGeneration_Lag"] - df["GenPrev"]) / df["GenPrev"].clip(lower=1e-6)

# Drop rows without full history
df = df.groupby("SiteKey", group_keys=False).apply(lambda g: g.iloc[LAG_STEPS + 1:]).reset_index(drop=True)

# --------------- Candidate triggers (causal) ---------------
if USE_FIXED_THRESHOLDS:
    gen_mask   = df["GenRelDrop_Lag"] < GEN_REL_DROP_THRESH
    temp_mask  = df["AirTemperature_Lag"] > HIGH_TEMP_THRESH
    hum_mask   = df["RelativeHumidity_Lag"] > HIGH_HUM_THRESH
else:
    # Past-only expanding robust thresholds
    def expanding_thresh(series, q=None, is_drop=False):
        if q is not None:
            return series.expanding().quantile(q).shift(1)
        med = series.expanding().median().shift(1)
        mad = (series - med).abs().expanding().median().shift(1)
        return med - K_MAD * mad if is_drop else med + K_MAD * mad

    drop_thr  = df.groupby("SiteKey", group_keys=False)["GenRelDrop_Lag"].apply(
        lambda s: expanding_thresh(s, is_drop=True))
    temp_thr  = df.groupby("SiteKey", group_keys=False)["AirTemperature_Lag"].apply(
        lambda s: expanding_thresh(s, q=0.95))
    hum_thr   = df.groupby("SiteKey", group_keys=False)["RelativeHumidity_Lag"].apply(
        lambda s: expanding_thresh(s, q=0.95))

    gen_mask  = df["GenRelDrop_Lag"] < drop_thr
    temp_mask = df["AirTemperature_Lag"] > temp_thr
    hum_mask  = df["RelativeHumidity_Lag"] > hum_thr

# Build candidate table
cand = pd.DataFrame({
    "SiteKey": df["SiteKey"],
    "Timestamp": df["Timestamp_Aligned"],
    "Trig_gen": gen_mask.values,
    "Trig_temp": temp_mask.values,
    "Trig_hum": hum_mask.values
})
# Small exogenous random candidates
np.random.seed(42); random.seed(42)
rand_mask = np.zeros(len(cand), dtype=bool)
if RANDOM_BG_RATE > 0:
    rand_idx = np.random.choice(cand.index, size=int(len(cand)*RANDOM_BG_RATE), replace=False)
    rand_mask[rand_idx] = True
cand["Trig_rand"] = rand_mask

# Melt into one row per (site, time, trigger)
long = cand.melt(id_vars=["SiteKey", "Timestamp"],
                 value_vars=["Trig_gen","Trig_temp","Trig_hum","Trig_rand"],
                 var_name="Trig", value_name="Fired").query("Fired").drop(columns="Fired")

# Priority and mapping to Type/Component/Severity (monotonic duration later)
PRIORITY = {"Trig_gen": 0, "Trig_temp": 1, "Trig_hum": 2, "Trig_rand": 3}
MAP_TYPE = {
    "Trig_gen":  ("Inverter Fault", "Inverter",     "Critical"),
    "Trig_temp": ("Sensor Calibration", "Sensor",   "Warning"),
    "Trig_hum":  ("Panel Cleaning", "PV Panel",     "Routine"),
    "Trig_rand": ("Routine Check", "Sensor",        "Minor")
}

long["prio"] = long["Trig"].map(PRIORITY)
long["MaintenanceType"] = long["Trig"].map(lambda t: MAP_TYPE[t][0])
long["Component"]       = long["Trig"].map(lambda t: MAP_TYPE[t][1])
long["Severity"]        = long["Trig"].map(lambda t: MAP_TYPE[t][2])
long["Date"]            = long["Timestamp"].dt.floor("D")

# --------------- Per-site/day selection with cooldown & balance ---------------
def select_for_group(g):
    # g rows: candidate triggers for one (SiteKey, Date)
    if g.empty:
        return g

    # sort by priority then time
    g = g.sort_values(["prio", "Timestamp"]).copy()

    # quotas
    target = TARGET_EVENTS_PER_DAY_PER_SITE
    type_cap = max(1, int(TYPE_MAX_SHARE * target))

    selected = []
    last_time = None
    type_counts = {}

    for _, r in g.iterrows():
        if len(selected) >= target:
            break
        # cooldown
        if last_time is not None:
            if (r["Timestamp"] - last_time).total_seconds() / 60.0 < COOLDOWN_MIN:
                continue
        # type balance
        t = r["MaintenanceType"]
        if type_counts.get(t, 0) >= type_cap:
            continue
        # uniqueness at this (Timestamp, Type, Component)
        if selected and any((r["Timestamp"] == s["Timestamp"]) and
                            (r["MaintenanceType"] == s["MaintenanceType"]) and
                            (r["Component"] == s["Component"]) for s in selected):
            continue

        selected.append(r.to_dict())
        type_counts[t] = type_counts.get(t, 0) + 1
        last_time = r["Timestamp"]

    return pd.DataFrame(selected)

sel = (long
       .groupby(["SiteKey","Date"], group_keys=True, sort=True)
       .apply(select_for_group)
       .reset_index(drop=True))

# --------------- Synthesize logs ---------------
DATA_STEP = timedelta(minutes=DATA_FREQ_MINUTES)
lag_delta = timedelta(minutes=LAG_STEPS * DATA_FREQ_MINUTES)

# Monotonic Severity → Duration (min/max)
DUR_RANGE = {
    "Routine":  (15, 45),
    "Minor":    (45, 75),
    "Warning":  (75, 120),
    "Critical": (120, 180)
}

operators = ["Tech John", "Tech Lee", "AutoSensor", "Jane Doe", "Team Alpha"]
actions   = ["resetting", "panel cleaning", "replacing sensor", "reconnecting cable", "firmware update"]

logs = []
log_id = 1

# To fetch lagged numeric values for descriptions (all from t-Δ)
df_lag_lookup = df.set_index(["SiteKey","Timestamp_Aligned"])

for _, r in sel.iterrows():
    site = r["SiteKey"]
    t    = r["Timestamp"]
    t_det= t - lag_delta  # evidence time
    t_av = t              # availability time

    mtype = r["MaintenanceType"]
    comp  = r["Component"]
    sev   = r["Severity"]

    # pull lagged context safely
    sg_lag = at_lag_T = rh_lag = None
    key = (site, t)
    key_det = (site, t_det)
    if key_det in df_lag_lookup.index:
        row = df_lag_lookup.loc[key_det]
        sg_lag = row["SolarGeneration_Lag"] if "SolarGeneration_Lag" in row else np.nan
        at_lag = row["AirTemperature_Lag"] if "AirTemperature_Lag" in row else np.nan
        rh_lag = row["RelativeHumidity_Lag"] if "RelativeHumidity_Lag" in row else np.nan

    # description (no same-time numbers)
    if r["Trig"] == "Trig_gen":
        desc = (f"Inverter fault suspected: output dropped between (t-{LAG_STEPS+1}) and (t-{LAG_STEPS}). "
                f"Reset initiated.")
        impact = "Major"
    elif r["Trig"] == "Trig_temp":
        desc = (f"Sensor recalibration scheduled after high temperature observed at (t-{LAG_STEPS}).")
        impact = "Minor"
    elif r["Trig"] == "Trig_hum":
        desc = (f"Panel cleaning planned after elevated humidity at (t-{LAG_STEPS}).")
        impact = "Minor"
    else:
        op = random.choice(operators); act = random.choice(actions)
        desc = f"{mtype} performed by {op}; {act} applied."
        impact = random.choice(["None","Minor","Reduced by 5%"])

    dmin, dmax = DUR_RANGE[sev]
    duration = int(np.random.randint(dmin, dmax + 1))

    log = {
        "LogID": log_id,
        "Timestamp": t,
        "EventDetectedAt": t_det,
        "AvailableToModelAt": t_av,
        "HorizonSteps": HORIZON_STEPS,
        "LagSteps": LAG_STEPS,
        "SiteKey": site,
        "MaintenanceType": mtype,
        "Severity": sev,
        "Component": comp,
        "Duration": duration,
        "Operator": random.choice(operators),
        "Description": desc,
        "Resolved": "Yes",
        "ActionTaken": random.choice(actions),
        "SolarImpact": impact
    }
    logs.append(log)
    log_id += 1

logs_df = pd.DataFrame(logs)

# --------------- Save ---------------
out_dir = os.path.dirname(MERGED_FILE)
output_path = os.path.join(out_dir, OUTPUT_BASENAME)
logs_df.to_csv(output_path, index=False)

print(f"\n[SUCCESS] Synthetic logs generated: {output_path}")
print(f"Total logs: {len(logs_df)} | Sites: {logs_df['SiteKey'].nunique()}")

# --------------- Usage protocol (unchanged core) ---------------
usage_protocol = f"""
USAGE PROTOCOL (Leakage-free):
 • Δ = {LAG_STEPS} steps ({LAG_STEPS*DATA_FREQ_MINUTES} min), h = {HORIZON_STEPS} steps → enforced Δ ≥ h.
 • Features at time t aggregate logs with AvailableToModelAt ≤ t (counts/recency/embeddings over [t-W, t]).
 • Target = SolarGeneration shifted by -h (predict t+h). Drop last h rows.
 • Rolling-origin evaluation; fit encoders/scalers on TRAIN ONLY.

Rate & Balance (this generator):
 • Target events/day/site = {TARGET_EVENTS_PER_DAY_PER_SITE}, cooldown = {COOLDOWN_MIN} min.
 • Max type share/day/site = {int(TYPE_MAX_SHARE*100)}%.
 • Uniqueness on (Timestamp, SiteKey, MaintenanceType, Component).

Thresholds:
 • {'Fixed domain thresholds (GEN_REL_DROP_THRESH, HIGH_TEMP_THRESH, HIGH_HUM_THRESH).' if USE_FIXED_THRESHOLDS else f'Expanding past-only robust thresholds with K_MAD={K_MAD}.'}
"""
with open(os.path.join(out_dir, PROTOCOL_TXT), "w") as f:
    f.write(usage_protocol)
print(usage_protocol)
