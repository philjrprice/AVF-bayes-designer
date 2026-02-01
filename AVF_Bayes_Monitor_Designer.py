
# streamlit_app.py
# Bayesian Single-Arm Trial Design & Simulation Workbench (Streamlit)
# Author: M365 Copilot for Phil Price
# Last updated: 2026-02-01
#
# Enhancements in this version:
# - Major speedups via precomputed decision boundaries (efficacy/safety/futility)
#   and memoization. Uses exact Beta CDFs (SciPy) when available.
# - Flexible look schedule modes: run+interval, absolute Ns, evenly spaced #looks,
#   fixed % of remaining, absolute %s of N.
# - Rich readouts: per-look stop probabilities, cumulative stopping curves,
#   boundary visualization, detailed ESS quantiles, and conditional metrics.
# - Same four analysis modes + export bundle.

import io
import json
import math
from math import lgamma
import zipfile
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
except Exception:
    raise SystemExit("This script is a Streamlit app. Please run with: streamlit run streamlit_app.py")

# Optional SciPy for fast and exact Beta CDF
TRY_SCIPY = True
try:
    if TRY_SCIPY:
        from scipy.special import betainc  # regularized incomplete beta I_x(a,b)
        SCIPY_AVAILABLE = True
    else:
        SCIPY_AVAILABLE = False
except Exception:
    SCIPY_AVAILABLE = False

# ----------------------------- Numeric Utilities ----------------------------- #

def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy not available for exact Beta CDF.")
    return float(betainc(a, b, x))

@lru_cache(maxsize=200000)
def posterior_tail_prob_beta_cached(a: float, b: float, threshold: float) -> float:
    """Exact Beta tail 1 - I_x(a,b), cached. Requires SciPy."""
    cdf = _regularized_incomplete_beta(a, b, threshold)
    return max(0.0, min(1.0, 1.0 - cdf))

def posterior_tail_prob_beta(a: float, b: float, threshold: float, rng: Optional[np.random.Generator] = None,
                             mc_samples: int = 3000) -> float:
    """Compute Pr(p > threshold | Beta(a,b)). Prefer SciPy; else MC fallback."""
    if a <= 0 or b <= 0:
        return float('nan')
    threshold = float(np.clip(threshold, 0.0, 1.0))
    if SCIPY_AVAILABLE:
        return posterior_tail_prob_beta_cached(float(a), float(b), float(threshold))
    # Fallback MC
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.beta(a, b, size=mc_samples)
    return float(np.mean(samples > threshold))

# Predictive Beta-Binomial PMF

def log_beta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)

def beta_binomial_pmf_vec(k: np.ndarray, n: int, alpha: float, beta: float) -> np.ndarray:
    from numpy import log, exp
    lg_coef = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    val = np.exp(lg_coef + log_beta(alpha + k, beta + n - k) - log_beta(alpha, beta))
    return val

@lru_cache(maxsize=200000)
def predictive_prob_success_at_final_cached(r_eff: int, n_curr: int, N_final: int,
                                            alpha_e: float, beta_e: float,
                                            p0: float, eff_final_prob: float) -> float:
    """Exact predictive probability for final success. Cached by (r,t) state."""
    n_remain = N_final - n_curr
    if n_remain < 0:
        return 0.0
    # Vectorized over future successes
    ks = np.arange(0, n_remain + 1)
    r_tot = r_eff + ks
    a_post = alpha_e + r_tot
    b_post = beta_e + N_final - r_tot
    # tail at final
    tails = np.array([posterior_tail_prob_beta(a, b, p0) for a, b in zip(a_post, b_post)])
    ok = tails > eff_final_prob
    # predictive weights under posterior at (r_eff, n_curr)
    w = beta_binomial_pmf_vec(ks, n_remain, alpha_e + r_eff, beta_e + (n_curr - r_eff))
    total = float(np.sum(w[ok]))
    return max(0.0, min(1.0, total))

# --------------------------- Look Schedule Builders --------------------------- #

def schedule_run_interval(N: int, run_in: int, interval: int) -> List[int]:
    looks = set()
    if (interval or 0) <= 0 and (run_in or 0) <= 0:
        looks.add(N)
    else:
        start = max(1, run_in) if run_in and run_in > 0 else max(1, interval)
        x = start
        while x < N:
            looks.add(int(x))
            if interval <= 0:
                break
            x += interval
        looks.add(N)
    return sorted(looks)

def schedule_absolute_n(N: int, absolute_ns: List[int]) -> List[int]:
    looks = sorted({x for x in absolute_ns if 1 <= x <= N})
    if N not in looks:
        looks.append(N)
    return sorted(set(looks))

def schedule_evenly_spaced(N: int, num_looks: int) -> List[int]:
    """Evenly spaced **including final N**. num_looks >= 1."""
    num_looks = max(1, int(num_looks))
    pts = np.linspace(1, N, num=num_looks, endpoint=True)
    looks = sorted({int(round(x)) for x in pts})
    looks[-1] = N
    return looks

def schedule_fixed_pct_remaining(N: int, first: int, pct_remaining: float) -> List[int]:
    """After first look at 'first', next look is increased by ceil(pct_remaining * remaining)."""
    looks = []
    t = max(1, int(first)) if first > 0 else 1
    if t < N:
        looks.append(t)
    while t < N:
        remaining = N - t
        step = int(math.ceil(pct_remaining * remaining))
        if step <= 0:
            step = 1
        t = t + step
        if t < N:
            looks.append(t)
        else:
            break
    looks.append(N)
    return sorted(set(looks))

def schedule_absolute_pct(N: int, pct_list: List[float]) -> List[int]:
    looks = sorted({int(round(N * p/100.0)) for p in pct_list})
    looks = [max(1, x) for x in looks]
    looks.append(N)
    return sorted(set(looks))

# ----------------------- Boundary Precomputation (FAST) ----------------------- #

def compute_efficacy_min_r_by_t(N: int, alpha_e: float, beta_e: float, p0: float,
                                prob_interim: float, prob_final: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays eff_min_r_interim[t], eff_min_r_final[t] for t=1..N.
    Each entry is the **minimum r** such that Pr(p>p0) > threshold. If no r satisfies, set to t+1.
    """
    min_r_interim = np.full(N+1, N+1, dtype=int)
    min_r_final = np.full(N+1, N+1, dtype=int)
    for t in range(1, N+1):
        # scan r from 0..t
        for r in range(0, t+1):
            a = alpha_e + r
            b = beta_e + t - r
            tail = posterior_tail_prob_beta(a, b, p0)
            if tail > prob_interim and min_r_interim[t] == N+1:
                min_r_interim[t] = r
            if tail > prob_final and min_r_final[t] == N+1:
                min_r_final[t] = r
        if min_r_interim[t] == N+1:
            min_r_interim[t] = t+1  # impossible; will never stop early for efficacy
        if min_r_final[t] == N+1:
            min_r_final[t] = t+1
    return min_r_interim, min_r_final

def compute_safety_min_r_by_t(N: int, alpha_s: float, beta_s: float, p_tox_star: float, saf_prob_cut: float) -> np.ndarray:
    """Return array saf_min_r[t] = minimum r_tox to trigger safety stop (strict > cutoff). If none, set t+1."""
    saf_min_r = np.full(N+1, N+1, dtype=int)
    for t in range(1, N+1):
        for r in range(0, t+1):
            a = alpha_s + r
            b = beta_s + t - r
            tail = posterior_tail_prob_beta(a, b, p_tox_star)
            if tail > saf_prob_cut:
                saf_min_r[t] = r
                break
        if saf_min_r[t] == N+1:
            saf_min_r[t] = t+1
    return saf_min_r

def compute_futility_min_r_by_t(N: int, alpha_e: float, beta_e: float, p0: float, eff_final_prob: float, fut_prob_cut: float) -> np.ndarray:
    """Return array fut_min_r_needed[t] = minimal r such that PredictiveProb(final success) >= fut_prob_cut.
    If even r=t fails, set to t+1 (meaning futility would always trigger at that t).
    Only meaningful for t < N; for t==N we set to 0.
    """
    fut_min = np.zeros(N+1, dtype=int)
    for t in range(1, N):
        r_needed = t + 1
        for r in range(0, t+1):
            pp = predictive_prob_success_at_final_cached(r, t, N, alpha_e, beta_e, p0, eff_final_prob)
            if pp >= fut_prob_cut:
                r_needed = r
                break
        fut_min[t] = r_needed
    fut_min[N] = 0
    return fut_min

# ----------------------------- Simulation Engine ----------------------------- #

def simulate_single_trial_fast(N: int,
                               eff_looks: List[int], saf_looks: List[int], fut_looks: List[int],
                               eff_min_r_interim: np.ndarray, eff_min_r_final: np.ndarray,
                               saf_min_r: np.ndarray, fut_min_r_needed: np.ndarray,
                               true_p_eff: float, true_p_tox: float,
                               rng: np.random.Generator) -> Dict:
    """Ultra-fast simulation using precomputed integer boundaries only."""
    eff_looks_set = set(eff_looks)
    saf_looks_set = set(saf_looks)
    fut_looks_set = set(fut_looks)

    eff_outcomes = rng.binomial(1, true_p_eff, size=N)
    tox_outcomes = rng.binomial(1, true_p_tox, size=N)

    r_eff = 0
    r_tox = 0
    stop_reason = None
    n_enrolled = 0

    for t in range(1, N+1):
        r_eff += eff_outcomes[t-1]
        r_tox += tox_outcomes[t-1]
        n_enrolled = t

        # 1) Safety
        if t in saf_looks_set:
            if r_tox >= saf_min_r[t]:
                stop_reason = "safety"
                break

        # 2) Efficacy
        if t in eff_looks_set:
            if t < N:
                if r_eff >= eff_min_r_interim[t]:
                    stop_reason = "efficacy_early"
                    break
            else:
                if r_eff >= eff_min_r_final[t]:
                    stop_reason = "efficacy_final"
                else:
                    stop_reason = "no_effect"
                break

        # 3) Futility
        if t in fut_looks_set and t < N:
            if r_eff < fut_min_r_needed[t]:
                stop_reason = "futility"
                break

    success = stop_reason in {"efficacy_early", "efficacy_final"}

    return {
        "n": N,
        "n_enrolled": n_enrolled,
        "r_eff": int(r_eff),
        "r_tox": int(r_tox),
        "stop_reason": stop_reason,
        "success": bool(success),
        "stop_time": n_enrolled,
    }

# Fallback accurate but slower engine (uses probabilities on the fly)

def simulate_single_trial_slow(N: int,
                               eff_looks: List[int], saf_looks: List[int], fut_looks: List[int],
                               alpha_e: float, beta_e: float, alpha_s: float, beta_s: float,
                               p0: float, eff_prob_interim: float, eff_prob_final: float,
                               p_tox_star: float, saf_prob_cut: float, fut_prob_cut: float,
                               true_p_eff: float, true_p_tox: float,
                               rng: np.random.Generator) -> Dict:
    eff_looks_set = set(eff_looks)
    saf_looks_set = set(saf_looks)
    fut_looks_set = set(fut_looks)

    eff_outcomes = rng.binomial(1, true_p_eff, size=N)
    tox_outcomes = rng.binomial(1, true_p_tox, size=N)

    r_eff = 0
    r_tox = 0
    stop_reason = None
    n_enrolled = 0

    for t in range(1, N + 1):
        r_eff += eff_outcomes[t - 1]
        r_tox += tox_outcomes[t - 1]
        n_enrolled = t

        # Safety
        if t in saf_looks_set:
            a_s = alpha_s + r_tox
            b_s = beta_s + t - r_tox
            prob_tox_high = posterior_tail_prob_beta(a_s, b_s, p_tox_star)
            if prob_tox_high > saf_prob_cut:
                stop_reason = "safety"
                break

        # Efficacy
        if t in eff_looks_set:
            a_e = alpha_e + r_eff
            b_e = beta_e + t - r_eff
            prob_eff_above_p0 = posterior_tail_prob_beta(a_e, b_e, p0)
            if t < N:
                if prob_eff_above_p0 > eff_prob_interim:
                    stop_reason = "efficacy_early"
                    break
            else:
                if prob_eff_above_p0 > eff_prob_final:
                    stop_reason = "efficacy_final"
                else:
                    stop_reason = "no_effect"
                break

        # Futility
        if t in fut_looks_set and t < N:
            pp = predictive_prob_success_at_final_cached(r_eff, t, N, alpha_e, beta_e, p0, eff_prob_final)
            if pp < fut_prob_cut:
                stop_reason = "futility"
                break

    success = stop_reason in {"efficacy_early", "efficacy_final"}
    return {
        "n": N,
        "n_enrolled": n_enrolled,
        "r_eff": int(r_eff),
        "r_tox": int(r_tox),
        "stop_reason": stop_reason,
        "success": bool(success),
        "stop_time": n_enrolled,
    }

# Batch

def simulate_many_trials(num_sims: int, seed: Optional[int], engine: str,
                         # design & schedules
                         N: int, eff_looks: List[int], saf_looks: List[int], fut_looks: List[int],
                         # priors & thresholds
                         alpha_e: float, beta_e: float, alpha_s: float, beta_s: float,
                         p0: float, eff_prob_interim: float, eff_prob_final: float,
                         p_tox_star: float, saf_prob_cut: float, fut_prob_cut: float,
                         # boundaries (optional for FAST)
                         eff_min_r_interim: Optional[np.ndarray] = None,
                         eff_min_r_final: Optional[np.ndarray] = None,
                         saf_min_r: Optional[np.ndarray] = None,
                         fut_min_r_needed: Optional[np.ndarray] = None,
                         # truths
                         true_p_eff: float = 0.2, true_p_tox: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    if engine == 'FAST':
        assert eff_min_r_interim is not None and eff_min_r_final is not None and saf_min_r is not None and fut_min_r_needed is not None
        for _ in range(num_sims):
            res = simulate_single_trial_fast(
                N, eff_looks, saf_looks, fut_looks,
                eff_min_r_interim, eff_min_r_final,
                saf_min_r, fut_min_r_needed,
                true_p_eff, true_p_tox, rng
            )
            records.append(res)
    else:
        for _ in range(num_sims):
            res = simulate_single_trial_slow(
                N, eff_looks, saf_looks, fut_looks,
                alpha_e, beta_e, alpha_s, beta_s,
                p0, eff_prob_interim, eff_prob_final,
                p_tox_star, saf_prob_cut, fut_prob_cut,
                true_p_eff, true_p_tox, rng
            )
            records.append(res)
    return pd.DataFrame(records)

# ----------------------------- Streamlit UI ---------------------------------- #

st.set_page_config(page_title="Bayesian Single-Arm Trial Designer", layout="wide")

st.title("Bayesian Single-Arm Trial Designer & Simulator (v2)")

st.caption(
    "Fast, flexible Bayesian single-arm trial design with separate efficacy, safety, and futility monitoring."
)

with st.expander("⚙️ Performance Options", expanded=False):
    st.markdown(
        "- **FAST mode (recommended)**: compiles integer decision **boundaries** at load times and simulates using only counters. Very fast and exact (when SciPy is present).\n"
        "- **ACCURATE mode**: evaluates posterior/predictive probabilities on the fly (slower).\n"
        f"- SciPy available: **{SCIPY_AVAILABLE}**. If False, probability computations use MC approximations."
    )

# Sidebar: Global settings
st.sidebar.header("Design Inputs")

# Basic rates
p0 = st.sidebar.number_input("Null response rate (p0)", 0.0, 1.0, 0.20, 0.01, key="auto_key_0002")
p1 = st.sidebar.number_input("Target response rate (p1)", 0.0, 1.0, 0.40, 0.01, key="auto_key_0003")

# Priors
st.sidebar.subheader("Priors (Beta)")
colA, colB = st.sidebar.columns(2)
alpha_e = colA.number_input("Efficacy α", min_value=0.01, value=1.0, step=0.1, key="auto_key_0004")
beta_e  = colB.number_input("Efficacy β",  min_value=0.01, value=1.0, step=0.1, key="auto_key_0005")
colC, colD = st.sidebar.columns(2)
alpha_s = colC.number_input("Safety α", min_value=0.01, value=1.0, step=0.1, key="auto_key_0006")
beta_s  = colD.number_input("Safety β",  min_value=0.01, value=1.0, step=0.1, key="auto_key_0007")

# Thresholds
st.sidebar.subheader("Decision Thresholds")
colT1, colT2 = st.sidebar.columns(2)
eff_prob_interim = colT1.number_input("Efficacy interim Pr[p>p0]", 0.5, 1.0, 0.99, 0.01, key="auto_key_0008")
eff_prob_final   = colT2.number_input("Efficacy final Pr[p>p0]",   0.5, 1.0, 0.975, 0.005, key="auto_key_0009")
colT3, colT4 = st.sidebar.columns(2)
p_tox_star   = colT3.number_input("Max acceptable tox (p_tox*)", 0.0, 1.0, 0.30, 0.01, key="auto_key_0010")
saf_prob_cut = colT4.number_input("Safety cutoff prob.", 0.5, 1.0, 0.90, 0.01, key="auto_key_0011")

fut_prob_cut = st.sidebar.number_input("Futility cutoff (predictive)", 0.0, 0.5, 0.05, 0.01, key="auto_key_0012")

# Schedules
st.sidebar.subheader("Look Schedules (choose mode per rule)")

SCHED_MODES = [
    "Run+Interval",           # existing pattern
    "Absolute N list",        # comma-separated Ns
    "Evenly spaced (#looks)", # evenly spaced including final
    "Fixed % of remaining",   # step increases as % of remaining
    "Absolute %s of N"        # comma-separated percentages
]

# Helper to build UI for one schedule

def schedule_inputs(label_prefix: str):
    mode = st.sidebar.selectbox(f"{label_prefix} schedule mode", SCHED_MODES, index=0, key=f"mode_{label_prefix}")
    if mode == "Run+Interval":
        c1, c2 = st.sidebar.columns(2)
        run_in = c1.number_input(f"{label_prefix} run-in", min_value=0, value=10, step=1, key=f"{label_prefix}_run")
        interval = c2.number_input(f"{label_prefix} interval", min_value=0, value=10, step=1, key=f"{label_prefix}_int")
        params = {"run_in": run_in, "interval": interval}
    elif mode == "Absolute N list":
        txt = st.sidebar.text_input(f"{label_prefix} Ns (comma)", value="", key=f"{label_prefix}_abs")
        params = {"absolute_ns": [int(x.strip()) for x in txt.split(',') if x.strip().isdigit()]}
    elif mode == "Evenly spaced (#looks)":
        looks = st.sidebar.number_input(f"{label_prefix} number of looks", min_value=1, value=4, step=1, key=f"{label_prefix}_num")
        params = {"num_looks": int(looks)}
    elif mode == "Fixed % of remaining":
        first = st.sidebar.number_input(f"{label_prefix} first look (n)", min_value=1, value=10, step=1, key=f"{label_prefix}_first")
        pct  = st.sidebar.number_input(f"{label_prefix} % of remaining (0-1)", min_value=0.01, max_value=0.99, value=0.25, step=0.01, key=f"{label_prefix}_pctrem")
        params = {"first": int(first), "pct_remaining": float(pct)}
    else: # Absolute %s of N
        txt = st.sidebar.text_input(f"{label_prefix} %s (comma)", value="25, 50, 75, 100", key=f"{label_prefix}_pctabs")
        def parse_pct(s):
            try:
                return float(s.strip())
            except:
                return None
        params = {"pct_list": [p for p in (parse_pct(x) for x in txt.split(',')) if p is not None]}
    return mode, params

mode_eff, params_eff = schedule_inputs("Efficacy")
mode_saf, params_saf = schedule_inputs("Safety")
mode_fut, params_fut = schedule_inputs("Futility")

# OC targets
st.sidebar.subheader("OC Targets")
colO1, colO2, colO3 = st.sidebar.columns(3)
max_type1 = colO1.number_input("Max Type I", 0.01, 0.5, 0.10, 0.01, key="auto_key_0013")
min_power = colO2.number_input("Min Power", 0.5, 1.0, 0.80, 0.01, key="auto_key_0014")
flex_tol  = colO3.number_input("Tolerance (±)", 0.0, 0.2, 0.02, 0.01, key="auto_key_0015")

# Simulation settings
st.sidebar.subheader("Simulation Settings")
colS1, colS2 = st.sidebar.columns(2)
rand_seed = int(colS1.number_input("Random seed", min_value=0, value=12345, step=1, key="auto_key_0016"))
engine = colS2.selectbox("Engine", ["FAST", "ACCURATE"], index=0, key="auto_key_0017")

# Build schedule based on choice

def build_schedule(N: int, mode: str, params: Dict) -> List[int]:
    if mode == "Run+Interval":
        return schedule_run_interval(N, params.get("run_in", 10), params.get("interval", 10))
    elif mode == "Absolute N list":
        return schedule_absolute_n(N, params.get("absolute_ns", []))
    elif mode == "Evenly spaced (#looks)":
        return schedule_evenly_spaced(N, params.get("num_looks", 4))
    elif mode == "Fixed % of remaining":
        return schedule_fixed_pct_remaining(N, params.get("first", 10), params.get("pct_remaining", 0.25))
    else:
        return schedule_absolute_pct(N, params.get("pct_list", [25, 50, 75, 100]))

# Helper: compute boundaries for a given N

@st.cache_data(show_spinner=False)
def compile_boundaries(N: int,
                       alpha_e: float, beta_e: float, p0: float, eff_prob_interim: float, eff_prob_final: float,
                       alpha_s: float, beta_s: float, p_tox_star: float, saf_prob_cut: float,
                       fut_prob_cut: float) -> Dict[str, np.ndarray]:
    eff_min_r_interim, eff_min_r_final = compute_efficacy_min_r_by_t(N, alpha_e, beta_e, p0, eff_prob_interim, eff_prob_final)
    saf_min_r = compute_safety_min_r_by_t(N, alpha_s, beta_s, p_tox_star, saf_prob_cut)
    fut_min_r_needed = compute_futility_min_r_by_t(N, alpha_e, beta_e, p0, eff_prob_final, fut_prob_cut)
    return {
        "eff_min_r_interim": eff_min_r_interim,
        "eff_min_r_final": eff_min_r_final,
        "saf_min_r": saf_min_r,
        "fut_min_r_needed": fut_min_r_needed,
    }

# Tabs
main_tabs = st.tabs([
    "1) Quick Scan (N range)",
    "2) Compare Specific Ns",
    "3) Deep Dive (single N)",
    "4) OC Matrix (grid)",
    "Selected Design, Boundaries & Export",
])

# --- 1) Quick Scan ---
with main_tabs[0]:
    st.subheader("1) Quick Scan Across N")
    c1, c2, c3 = st.columns(3)
    N_min = int(c1.number_input("N min", min_value=5, value=40, step=1, key="auto_key_0018"))
    N_max = int(c2.number_input("N max", min_value=N_min, value=80, step=1, key="auto_key_0019"))
    N_step = int(c3.number_input("Step", min_value=1, value=5, step=1, key="auto_key_0020"))
    c4, c5 = st.columns(2)
    sims_fast = int(c4.number_input("Sims per N", min_value=500, value=3000, step=500, key="auto_key_0021"))
    ptox_eval = float(c5.number_input("True toxicity (acceptable)", 0.0, 1.0, max(0.01, min(p_tox_star - 0.05, 0.99)), 0.01, key="auto_key_0022"))
    run_btn = st.button("Run Quick Scan", type="primary", key="auto_key_0023")

    if run_btn:
        results = []
        Ns = list(range(N_min, N_max + 1, N_step))
        prog = st.progress(0.0)
        for i, N in enumerate(Ns, start=1):
            eff_looks = build_schedule(N, mode_eff, params_eff)
            saf_looks = build_schedule(N, mode_saf, params_saf)
            fut_looks = build_schedule(N, mode_fut, params_fut)
            if engine == 'FAST':
                b = compile_boundaries(N, alpha_e, beta_e, p0, eff_prob_interim, eff_prob_final,
                                       alpha_s, beta_s, p_tox_star, saf_prob_cut, fut_prob_cut)
                df_h0 = simulate_many_trials(sims_fast, rand_seed + 11*N, 'FAST', N, eff_looks, saf_looks, fut_looks,
                                             alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                             p_tox_star, saf_prob_cut, fut_prob_cut,
                                             b['eff_min_r_interim'], b['eff_min_r_final'], b['saf_min_r'], b['fut_min_r_needed'],
                                             true_p_eff=p0, true_p_tox=ptox_eval)
                df_h1 = simulate_many_trials(sims_fast, rand_seed + 13*N, 'FAST', N, eff_looks, saf_looks, fut_looks,
                                             alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                             p_tox_star, saf_prob_cut, fut_prob_cut,
                                             b['eff_min_r_interim'], b['eff_min_r_final'], b['saf_min_r'], b['fut_min_r_needed'],
                                             true_p_eff=p1, true_p_tox=ptox_eval)
            else:
                df_h0 = simulate_many_trials(sims_fast, rand_seed + 11*N, 'ACCURATE', N, eff_looks, saf_looks, fut_looks,
                                             alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                             p_tox_star, saf_prob_cut, fut_prob_cut,
                                             true_p_eff=p0, true_p_tox=ptox_eval)
                df_h1 = simulate_many_trials(sims_fast, rand_seed + 13*N, 'ACCURATE', N, eff_looks, saf_looks, fut_looks,
                                             alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                             p_tox_star, saf_prob_cut, fut_prob_cut,
                                             true_p_eff=p1, true_p_tox=ptox_eval)
            results.append({
                "N": N,
                "Type I": float(df_h0['success'].mean()),
                "Power": float(df_h1['success'].mean()),
                "ESS@H0": float(df_h0['n_enrolled'].mean()),
                "ESS@H1": float(df_h1['n_enrolled'].mean()),
                "Pr(EarlySucc)": float((df_h1['stop_reason'] == 'efficacy_early').mean()),
                "Pr(SafetyStop@H0)": float((df_h0['stop_reason'] == 'safety').mean()),
            })
            prog.progress(i/len(Ns))
        scan_df = pd.DataFrame(results)
        st.session_state['latest_scan_df'] = scan_df
        st.dataframe(scan_df.style.format({"Type I": "{:.3f}", "Power": "{:.3f}", "ESS@H0": "{:.1f}", "ESS@H1": "{:.1f}", "Pr(EarlySucc)": "{:.3f}", "Pr(SafetyStop@H0)": "{:.3f}"}), use_container_width=True)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(scan_df['N'], scan_df['Type I'], marker='o', label='Type I')
        ax.plot(scan_df['N'], scan_df['Power'], marker='o', label='Power')
        ax.axhline(max_type1, color='r', ls='--', alpha=0.5, label='Max Type I')
        ax.axhline(min_power, color='g', ls='--', alpha=0.5, label='Min Power')
        ax.set_xlabel('N'); ax.set_ylabel('Probability'); ax.set_title('Type I & Power vs N')
        ax.legend(); st.pyplot(fig)
        feas = scan_df[(scan_df['Type I'] <= max_type1 + flex_tol) & (scan_df['Power'] >= min_power - flex_tol)]
        st.markdown("**Feasible designs (within tolerance):**")
        st.dataframe(feas, use_container_width=True)

# --- 2) Compare Specific Ns ---
with main_tabs[1]:
    st.subheader("2) Compare Specific Ns")
    Ns_text = st.text_input("N values (comma)", value="40, 50, 60, 70, 80", key="auto_key_0024")
    sims_cmp = int(st.number_input("Sims per N", min_value=1000, value=5000, step=1000, key="auto_key_0025"))
    ptox_eval = float(st.number_input("True toxicity (acceptable)", 0.0, 1.0, max(0.01, min(p_tox_star - 0.05, 0.99)), 0.01, key="auto_key_0026"))
    cmp_btn = st.button("Run Comparison", key="auto_key_0027")

    if cmp_btn:
        Ns_list = [int(x.strip()) for x in Ns_text.split(',') if x.strip().isdigit()]
        rows = []
        prog = st.progress(0.0)
        for i, N in enumerate(Ns_list, start=1):
            eff_looks = build_schedule(N, mode_eff, params_eff)
            saf_looks = build_schedule(N, mode_saf, params_saf)
            fut_looks = build_schedule(N, mode_fut, params_fut)
            if engine == 'FAST':
                b = compile_boundaries(N, alpha_e, beta_e, p0, eff_prob_interim, eff_prob_final,
                                       alpha_s, beta_s, p_tox_star, saf_prob_cut, fut_prob_cut)
                df_h0 = simulate_many_trials(sims_cmp, rand_seed + 101*N, 'FAST', N, eff_looks, saf_looks, fut_looks,
                                             alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                             p_tox_star, saf_prob_cut, fut_prob_cut,
                                             b['eff_min_r_interim'], b['eff_min_r_final'], b['saf_min_r'], b['fut_min_r_needed'],
                                             true_p_eff=p0, true_p_tox=ptox_eval)
                df_h1 = simulate_many_trials(sims_cmp, rand_seed + 103*N, 'FAST', N, eff_looks, saf_looks, fut_looks,
                                             alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                             p_tox_star, saf_prob_cut, fut_prob_cut,
                                             b['eff_min_r_interim'], b['eff_min_r_final'], b['saf_min_r'], b['fut_min_r_needed'],
                                             true_p_eff=p1, true_p_tox=ptox_eval)
            else:
                df_h0 = simulate_many_trials(sims_cmp, rand_seed + 101*N, 'ACCURATE', N, eff_looks, saf_looks, fut_looks,
                                             alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                             p_tox_star, saf_prob_cut, fut_prob_cut,
                                             true_p_eff=p0, true_p_tox=ptox_eval)
                df_h1 = simulate_many_trials(sims_cmp, rand_seed + 103*N, 'ACCURATE', N, eff_looks, saf_looks, fut_looks,
                                             alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                             p_tox_star, saf_prob_cut, fut_prob_cut,
                                             true_p_eff=p1, true_p_tox=ptox_eval)
            rows.append({
                "N": N,
                "Type I": float(df_h0['success'].mean()),
                "Power": float(df_h1['success'].mean()),
                "ESS@H0": float(df_h0['n_enrolled'].mean()),
                "ESS@H1": float(df_h1['n_enrolled'].mean()),
                "Pr(EarlySucc|H1)": float((df_h1['stop_reason'] == 'efficacy_early').mean()),
            })
            prog.progress(i/len(Ns_list))
        cmp_df = pd.DataFrame(rows).sort_values("N")
        st.session_state['latest_compare_df'] = cmp_df
        st.dataframe(cmp_df.style.format({"Type I": "{:.3f}", "Power": "{:.3f}", "ESS@H0": "{:.1f}", "ESS@H1": "{:.1f}", "Pr(EarlySucc|H1)": "{:.3f}"}), use_container_width=True)
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.plot(cmp_df['N'], cmp_df['Type I'], marker='o', label='Type I')
        ax2.plot(cmp_df['N'], cmp_df['Power'], marker='o', label='Power')
        ax2.axhline(max_type1, color='r', ls='--', alpha=0.5, label='Max Type I')
        ax2.axhline(min_power, color='g', ls='--', alpha=0.5, label='Min Power')
        ax2.set_xlabel('N'); ax2.set_ylabel('Probability'); ax2.set_title('Type I & Power vs N (Comparison)'); ax2.legend()
        st.pyplot(fig2)

# --- 3) Deep Dive ---
with main_tabs[2]:
    st.subheader("3) Deep Dive on a Specific N")
    N_deep = int(st.number_input("Select N (final)", min_value=5, value=60, step=1, key="auto_key_0028"))
    sims_deep = int(st.number_input("Sims (deep dive)", min_value=5000, value=50000, step=5000, key="auto_key_0029"))
    ptox_deep = float(st.number_input("True toxicity (acceptable)", 0.0, 1.0, max(0.01, min(p_tox_star - 0.05, 0.99)), 0.01, key="auto_key_0030"))
    deep_btn = st.button("Run Deep Dive", key="auto_key_0031")

    if deep_btn:
        eff_looks = build_schedule(N_deep, mode_eff, params_eff)
        saf_looks = build_schedule(N_deep, mode_saf, params_saf)
        fut_looks = build_schedule(N_deep, mode_fut, params_fut)
        if engine == 'FAST':
            b = compile_boundaries(N_deep, alpha_e, beta_e, p0, eff_prob_interim, eff_prob_final,
                                   alpha_s, beta_s, p_tox_star, saf_prob_cut, fut_prob_cut)
            df_h0 = simulate_many_trials(sims_deep, rand_seed + 1001, 'FAST', N_deep, eff_looks, saf_looks, fut_looks,
                                         alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                         p_tox_star, saf_prob_cut, fut_prob_cut,
                                         b['eff_min_r_interim'], b['eff_min_r_final'], b['saf_min_r'], b['fut_min_r_needed'],
                                         true_p_eff=p0, true_p_tox=ptox_deep)
            df_h1 = simulate_many_trials(sims_deep, rand_seed + 1003, 'FAST', N_deep, eff_looks, saf_looks, fut_looks,
                                         alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                         p_tox_star, saf_prob_cut, fut_prob_cut,
                                         b['eff_min_r_interim'], b['eff_min_r_final'], b['saf_min_r'], b['fut_min_r_needed'],
                                         true_p_eff=p1, true_p_tox=ptox_deep)
        else:
            df_h0 = simulate_many_trials(sims_deep, rand_seed + 1001, 'ACCURATE', N_deep, eff_looks, saf_looks, fut_looks,
                                         alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                         p_tox_star, saf_prob_cut, fut_prob_cut,
                                         true_p_eff=p0, true_p_tox=ptox_deep)
            df_h1 = simulate_many_trials(sims_deep, rand_seed + 1003, 'ACCURATE', N_deep, eff_looks, saf_looks, fut_looks,
                                         alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                         p_tox_star, saf_prob_cut, fut_prob_cut,
                                         true_p_eff=p1, true_p_tox=ptox_deep)

        # Key metrics & CIs
        def binom_ci(p, n, alpha=0.05):
            # normal approx
            se = np.sqrt(p*(1-p)/max(1,n))
            z = 1.959963984540054
            return max(0.0, p - z*se), min(1.0, p + z*se)

        type1 = float(df_h0['success'].mean()); type1_ci = binom_ci(type1, len(df_h0))
        power = float(df_h1['success'].mean()); power_ci = binom_ci(power, len(df_h1))
        ess_h0 = float(df_h0['n_enrolled'].mean()); ess_h1 = float(df_h1['n_enrolled'].mean())
        st.markdown(f"**Type I** (p={p0:.2f}) = **{type1:.3f}** (95% CI {type1_ci[0]:.3f}-{type1_ci[1]:.3f}) | "
                    f"**Power** (p={p1:.2f}) = **{power:.3f}** (95% CI {power_ci[0]:.3f}-{power_ci[1]:.3f})")
        st.markdown(f"**ESS@H0** = {ess_h0:.1f} | **ESS@H1** = {ess_h1:.1f}")

        # Detailed readouts
        def detailed_readouts(df: pd.DataFrame, label: str):
            st.markdown(f"### Detailed readouts – {label}")
            # Stop reasons
            sr = df['stop_reason'].value_counts().reindex(['safety','efficacy_early','efficacy_final','futility','no_effect'], fill_value=0)
            st.write(pd.DataFrame({'count': sr, 'prob': (sr/len(df)).round(3)}))
            # n_enrolled quantiles
            q = df['n_enrolled'].quantile([0.1,0.25,0.5,0.75,0.9]).round(1)
            st.write(pd.DataFrame({'quantile': q.index, 'n_enrolled': q.values}))
            # Conditional metrics
            early = df[df['stop_reason']=='efficacy_early']
            fut = df[df['stop_reason']=='futility']
            st.write({
                'Pr(Early success)': float((df['stop_reason']=='efficacy_early').mean()),
                'Pr(Futility)': float((df['stop_reason']=='futility').mean()),
                'ESS | Early success': float(early['n_enrolled'].mean()) if not early.empty else np.nan,
                'ESS | Futility': float(fut['n_enrolled'].mean()) if not fut.empty else np.nan,
                'E[r_eff] at stop': float(df['r_eff'].mean()),
                'E[r_tox] at stop': float(df['r_tox'].mean()),
            })
            # Per-look stop probabilities (map time to look index per schedule)
            look_map = {t:i+1 for i,t in enumerate(sorted(set(eff_looks + saf_looks + fut_looks)))}
            df['look_idx'] = df['stop_time'].map(look_map).fillna(np.nan)
            by_look = df.groupby(['look_idx','stop_reason']).size().unstack(fill_value=0)
            by_look_prob = by_look.div(len(df))
            st.markdown("**Per-look stop probabilities**")
            st.dataframe(by_look_prob)
            # Cumulative stopping curve
            cum = df['n_enrolled'].value_counts().sort_index().cumsum()/len(df)
            fig, ax = plt.subplots(figsize=(6,3.5))
            ax.step(cum.index, cum.values, where='post')
            ax.set_xlabel('n (enrolled)'); ax.set_ylabel('Cumulative Pr[stop ≤ n]'); ax.set_title(f'Cumulative stopping: {label}')
            st.pyplot(fig)
            return by_look_prob

        bl_h0 = detailed_readouts(df_h0, 'H0')
        bl_h1 = detailed_readouts(df_h1, 'H1')

        # Boundary visualization
        if engine == 'FAST':
            b = compile_boundaries(N_deep, alpha_e, beta_e, p0, eff_prob_interim, eff_prob_final,
                                   alpha_s, beta_s, p_tox_star, saf_prob_cut, fut_prob_cut)
            figB, axB = plt.subplots(figsize=(7,4))
            ts = np.arange(1, N_deep+1)
            axB.plot(ts, b['eff_min_r_interim'][1:], label='Efficacy interim r_min', color='#1f77b4')
            axB.plot(ts, b['eff_min_r_final'][1:], label='Efficacy final r_min', color='#2ca02c')
            axB.plot(ts, b['saf_min_r'][1:], label='Safety r_tox min', color='#d62728')
            axB.plot(ts[:-1], b['fut_min_r_needed'][1:-1], label='Futility r_eff needed', color='#9467bd')
            axB.set_xlabel('n'); axB.set_ylabel('required count'); axB.set_title('Compiled decision boundaries')
            axB.legend(); st.pyplot(figB)

        # Save for export
        st.session_state['deep_dive_h0_df'] = df_h0
        st.session_state['deep_dive_h1_df'] = df_h1

# --- 4) OC Matrix ---
with main_tabs[3]:
    st.subheader("4) Operating Characteristics (OC) Matrix")
    c1, c2 = st.columns(2)
    p_eff_min = float(c1.number_input("Efficacy min", 0.0, 1.0, max(0.0, p0-0.1), 0.01, key="auto_key_0032"))
    p_eff_max = float(c1.number_input("Efficacy max", p_eff_min, 1.0, min(1.0, p1+0.2), 0.01, key="auto_key_0033"))
    p_eff_step= float(c1.number_input("Efficacy step", 0.01, 0.5, 0.05, 0.01, key="auto_key_0034"))
    p_tox_min = float(c2.number_input("Toxicity min", 0.0, 1.0, max(0.0, p_tox_star-0.2), 0.01, key="auto_key_0035"))
    p_tox_max = float(c2.number_input("Toxicity max", p_tox_min, 1.0, min(1.0, p_tox_star+0.2), 0.01, key="auto_key_0036"))
    p_tox_step= float(c2.number_input("Toxicity step", 0.01, 0.5, 0.05, 0.01, key="auto_key_0037"))

    N_oc = int(st.number_input("Select N for OC matrix", min_value=5, value=60, step=1, key="auto_key_0038"))
    sims_oc = int(st.number_input("Sims per cell", min_value=500, value=2000, step=500, key="auto_key_0039"))
    run_oc_btn = st.button("Run OC Matrix", key="auto_key_0040")

    if run_oc_btn:
        eff_looks = build_schedule(N_oc, mode_eff, params_eff)
        saf_looks = build_schedule(N_oc, mode_saf, params_saf)
        fut_looks = build_schedule(N_oc, mode_fut, params_fut)
        if engine == 'FAST':
            b = compile_boundaries(N_oc, alpha_e, beta_e, p0, eff_prob_interim, eff_prob_final,
                                   alpha_s, beta_s, p_tox_star, saf_prob_cut, fut_prob_cut)
        eff_vals = np.round(np.arange(p_eff_min, p_eff_max+1e-9, p_eff_step), 3)
        tox_vals = np.round(np.arange(p_tox_min, p_tox_max+1e-9, p_tox_step), 3)
        recs = []
        prog = st.progress(0.0)
        total = len(eff_vals)*len(tox_vals)
        idx=0
        for pe in eff_vals:
            for pt in tox_vals:
                if engine == 'FAST':
                    df = simulate_many_trials(sims_oc, rand_seed + 2001 + idx, 'FAST', N_oc, eff_looks, saf_looks, fut_looks,
                                              alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                              p_tox_star, saf_prob_cut, fut_prob_cut,
                                              b['eff_min_r_interim'], b['eff_min_r_final'], b['saf_min_r'], b['fut_min_r_needed'],
                                              true_p_eff=float(pe), true_p_tox=float(pt))
                else:
                    df = simulate_many_trials(sims_oc, rand_seed + 2001 + idx, 'ACCURATE', N_oc, eff_looks, saf_looks, fut_looks,
                                              alpha_e, beta_e, alpha_s, beta_s, p0, eff_prob_interim, eff_prob_final,
                                              p_tox_star, saf_prob_cut, fut_prob_cut,
                                              true_p_eff=float(pe), true_p_tox=float(pt))
                recs.append({
                    'p_eff': float(pe), 'p_tox': float(pt),
                    'Pr(Success)': float(df['success'].mean()),
                    'ESS': float(df['n_enrolled'].mean()),
                    'Pr(SafetyStop)': float((df['stop_reason']=='safety').mean()),
                    'Pr(Futility)': float((df['stop_reason']=='futility').mean()),
                })
                idx+=1
                prog.progress(idx/total)
        oc_df = pd.DataFrame(recs)
        st.session_state['oc_matrix_df'] = oc_df
        st.dataframe(oc_df, use_container_width=True)
        # Heatmaps
        def plot_heatmap(pivot_df: pd.DataFrame, title: str, cmap='viridis'):
            fig, ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(pivot_df.values, origin='lower', aspect='auto', cmap=cmap,
                           extent=[pivot_df.columns.min(), pivot_df.columns.max(), pivot_df.index.min(), pivot_df.index.max()])
            ax.set_xlabel('p_tox'); ax.set_ylabel('p_eff'); ax.set_title(title)
            fig.colorbar(im, ax=ax)
            return fig
        st.markdown("**Heatmaps**")
        piv1 = oc_df.pivot(index='p_eff', columns='p_tox', values='Pr(Success)')
        piv2 = oc_df.pivot(index='p_eff', columns='p_tox', values='ESS')
        piv3 = oc_df.pivot(index='p_eff', columns='p_tox', values='Pr(SafetyStop)')
        piv4 = oc_df.pivot(index='p_eff', columns='p_tox', values='Pr(Futility)')
        st.pyplot(plot_heatmap(piv1, 'Pr(Success)'))
        st.pyplot(plot_heatmap(piv2, 'Expected Sample Size', cmap='magma'))
        st.pyplot(plot_heatmap(piv3, 'Pr(Safety Stop)', cmap='Reds'))
        st.pyplot(plot_heatmap(piv4, 'Pr(Futility)', cmap='Blues'))

# --- 5) Selected Design, Boundaries & Export ---
with main_tabs[4]:
    st.subheader("Selected Design & Export")
    N_sel = int(st.number_input("Final N (selected)", min_value=5, value=60, step=1, key="auto_key_0041"))
    eff_looks_sel = build_schedule(N_sel, mode_eff, params_eff)
    saf_looks_sel = build_schedule(N_sel, mode_saf, params_saf)
    fut_looks_sel = build_schedule(N_sel, mode_fut, params_fut)

    if st.button("Compile & show decision boundaries", key="auto_key_0042"):
        b = compile_boundaries(N_sel, alpha_e, beta_e, p0, eff_prob_interim, eff_prob_final,
                               alpha_s, beta_s, p_tox_star, saf_prob_cut, fut_prob_cut)
        st.session_state['boundaries_sel'] = {k: v.tolist() for k,v in b.items()}
        st.success("Boundaries compiled.")
        ts = np.arange(1, N_sel+1)
        figB2, axB2 = plt.subplots(figsize=(7,4))
        axB2.plot(ts, b['eff_min_r_interim'][1:], label='Efficacy interim r_min')
        axB2.plot(ts, b['eff_min_r_final'][1:], label='Efficacy final r_min')
        axB2.plot(ts, b['saf_min_r'][1:], label='Safety r_tox min')
        axB2.plot(ts[:-1], b['fut_min_r_needed'][1:-1], label='Futility r_eff needed')
        axB2.set_xlabel('n'); axB2.set_ylabel('required count'); axB2.legend(); axB2.set_title('Decision boundaries (selected N)')
        st.pyplot(figB2)

    if st.button("Set as Selected Design", key="auto_key_0043"):
        selection = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "N": N_sel,
            "p0": p0, "p1": p1,
            "priors": {"eff": [alpha_e, beta_e], "saf": [alpha_s, beta_s]},
            "eff_schedule": {"mode": mode_eff, "params": params_eff, "looks": eff_looks_sel},
            "saf_schedule": {"mode": mode_saf, "params": params_saf, "looks": saf_looks_sel},
            "fut_schedule": {"mode": mode_fut, "params": params_fut, "looks": fut_looks_sel},
            "thresholds": {
                "eff_interim": eff_prob_interim, "eff_final": eff_prob_final,
                "p_tox_star": p_tox_star, "safety_cut": saf_prob_cut, "futility_cut": fut_prob_cut,
            },
            "oc_targets": {"max_type1": max_type1, "min_power": min_power, "tolerance": flex_tol},
            "engine": engine, "scipy": SCIPY_AVAILABLE,
            "boundaries": st.session_state.get('boundaries_sel')
        }
        st.session_state['final_design_selection'] = selection
        st.success("Selected design saved for export.")

    if 'final_design_selection' in st.session_state:
        st.markdown("### Final Design Summary")
        st.json(st.session_state['final_design_selection'])

    st.markdown("---")
    st.subheader("Export")

    def _fig_to_png_bytes(fig) -> bytes:
        buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=160); buf.seek(0); return buf.read()

    def build_export_zip() -> bytes:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            readme = (
                "Bayesian Single-Arm Trial Designer Export\n"
                f"Created: {datetime.utcnow().isoformat()}Z\n\n"
                "Contents:\n"
                "- final_design.json: JSON summary of selected design\n"
                "- boundaries.json: compiled boundaries for selected N (if compiled)\n"
                "- scan.csv, compare.csv: if available\n"
                "- deep_dive_h0.csv / deep_dive_h1.csv: raw outputs (if available)\n"
                "- oc_matrix.csv: OC matrix (if available)\n"
                "- figures/*.png: Plots\n"
            )
            zf.writestr("README.txt", readme)
            if 'final_design_selection' in st.session_state:
                zf.writestr("final_design.json", json.dumps(st.session_state['final_design_selection'], indent=2))
                if st.session_state['final_design_selection'].get('boundaries') is not None:
                    zf.writestr("boundaries.json", json.dumps(st.session_state['final_design_selection']['boundaries'], indent=2))
            if 'latest_scan_df' in st.session_state:
                df = st.session_state['latest_scan_df']
                zf.writestr("scan.csv", df.to_csv(index=False))
                try:
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.plot(df['N'], df['Type I'], marker='o', label='Type I')
                    ax.plot(df['N'], df['Power'], marker='o', label='Power')
                    ax.set_xlabel('N'); ax.set_ylabel('Probability'); ax.set_title('Quick Scan: Type I & Power vs N'); ax.legend()
                    zf.writestr("figures/quick_scan.png", _fig_to_png_bytes(fig)); plt.close(fig)
                except: pass
            if 'latest_compare_df' in st.session_state:
                zf.writestr("compare.csv", st.session_state['latest_compare_df'].to_csv(index=False))
            if 'deep_dive_h0_df' in st.session_state:
                zf.writestr("deep_dive_h0.csv", st.session_state['deep_dive_h0_df'].to_csv(index=False))
            if 'deep_dive_h1_df' in st.session_state:
                zf.writestr("deep_dive_h1.csv", st.session_state['deep_dive_h1_df'].to_csv(index=False))
            if 'oc_matrix_df' in st.session_state:
                oc_df = st.session_state['oc_matrix_df']
                zf.writestr("oc_matrix.csv", oc_df.to_csv(index=False))
                try:
                    piv_success = oc_df.pivot(index='p_eff', columns='p_tox', values='Pr(Success)')
                    piv_ess = oc_df.pivot(index='p_eff', columns='p_tox', values='ESS')
                    piv_saf = oc_df.pivot(index='p_eff', columns='p_tox', values='Pr(SafetyStop)')
                    piv_fut = oc_df.pivot(index='p_eff', columns='p_tox', values='Pr(Futility)')
                    def heat(pivot_df, title, cmap='viridis'):
                        fig, ax = plt.subplots(figsize=(6,5))
                        im = ax.imshow(pivot_df.values, origin='lower', aspect='auto', cmap=cmap,
                                       extent=[pivot_df.columns.min(), pivot_df.columns.max(), pivot_df.index.min(), pivot_df.index.max()])
                        ax.set_xlabel('p_tox'); ax.set_ylabel('p_eff'); ax.set_title(title); fig.colorbar(im, ax=ax)
                        return fig
                    zf.writestr("figures/oc_success.png", _fig_to_png_bytes(heat(piv_success, 'Pr(Success)')))
                    zf.writestr("figures/oc_ess.png", _fig_to_png_bytes(heat(piv_ess, 'Expected Sample Size', 'magma')))
                    zf.writestr("figures/oc_safety.png", _fig_to_png_bytes(heat(piv_saf, 'Pr(Safety Stop)', 'Reds')))
                    zf.writestr("figures/oc_futility.png", _fig_to_png_bytes(heat(piv_fut, 'Pr(Futility)', 'Blues')))
                    plt.close('all')
                except: pass
        zbuf.seek(0)
        return zbuf.read()

    if st.button("Build export zip", key="auto_key_0044"):
        try:
            zip_bytes = build_export_zip()
            st.download_button(label="Download Export (.zip)", data=zip_bytes, file_name="bayesian_single_arm_design_export.zip", mime="application/zip", key="auto_key_0045")
        except Exception as e:
            st.error(f"Export failed: {e}")

# Footer
st.markdown('---')
with st.expander('Methodology & Defaults', expanded=False):
    st.markdown(
        """
        **Model & Rules**
        - Efficacy & Safety ~ Bernoulli with Beta priors; posteriors are Beta(α + r, β + n − r).
        - Efficacy rule: Pr(p_eff > p0) > threshold (interim/final). Safety rule: Pr(p_tox > p_tox*) > cutoff.
        - Futility: predictive Pr(meeting *final* efficacy rule at N) < cutoff.
        - Stop priority: Safety → Efficacy success → Futility.

        **Performance**
        - FAST mode compiles integer decision boundaries for each look size t and uses only integer comparisons; with SciPy this is exact.
        - Without SciPy, Beta tail probabilities use a Monte Carlo fallback. Predictive probability remains exact via Beta–Binomial.

        **Defaults**
        - Priors Beta(1,1). Thresholds: interim 0.99, final 0.975; safety cutoff 0.90 at p_tox* = 0.30; futility 0.05.
        - OC targets: Type I ≤ 0.10; Power ≥ 0.80; tolerance ±0.02.
        """
    )
