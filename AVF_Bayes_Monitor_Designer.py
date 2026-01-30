 
# AVF_Bayes_Monitor_Designer_runin_safety_v3.py
# Streamlit app for single-arm Bayesian monitored design (binary endpoint)
# Enhancements over v2:
#  • FIXED selectbox bug: use labels directly (no index mapping)
#  • Threshold Tuner++:
#      - Stage 1: Bisection on θ_final to hit α target at p0
#      - Stage 2: Joint search over (θ_interim, c_futility) with coarse→refined grids
#  • Monte Carlo SE and 95% CI bands for key proportions and ESS
#  • Protocol-ready PDF export (design + key results)
#  • All previous features retained (split run-ins/schedules, OC Explorer, CSV/JSON export)

from __future__ import annotations
import io
import json
import zipfile
import math
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple
from scipy.stats import beta
from scipy.special import betaln, comb

# PDF export
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

SCHEMA_VERSION = "screen_v5_refined_tuner_mcse_pdf"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CORE BAYESIAN UTILITIES                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def beta_posterior_params(a0: float, b0: float, x: int, n: int) -> Tuple[float, float]:
    return a0 + x, b0 + (n - x)

def posterior_prob_p_greater_than(p_cut: float, a_post: float, b_post: float) -> float:
    return 1.0 - beta.cdf(p_cut, a_post, b_post)

def min_successes_for_posterior_threshold(a0: float, b0: float, N: int, p0: float, theta_final: float) -> Optional[int]:
    lo, hi = 0, N
    ans = N + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        a_post, b_post = beta_posterior_params(a0, b0, mid, N)
        prob = posterior_prob_p_greater_than(p0, a_post, b_post)
        if prob >= theta_final:
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return None if ans == N + 1 else int(ans)

def log_beta_binomial_pmf(y: int, m: int, a: float, b: float) -> float:
    if y < 0 or y > m:
        return -np.inf
    return math.log(comb(m, y)) + betaln(y + a, m - y + b) - betaln(a, b)

def beta_binomial_cdf_upper_tail(y_min: int, m: int, a: float, b: float) -> float:
    if y_min <= 0:
        return 1.0
    if y_min > m:
        return 0.0
    ys = np.arange(y_min, m + 1)
    logs = np.array([log_beta_binomial_pmf(int(y), m, a, b) for y in ys])
    mlog = np.max(logs)
    return float(np.exp(mlog) * np.sum(np.exp(logs - mlog)))

def predictive_prob_of_final_success(a0: float, b0: float, N_total: int, x_curr: int, n_curr: int, p0: float, theta_final: float) -> float:
    a_post, b_post = beta_posterior_params(a0, b0, x_curr, n_curr)
    m_remain = N_total - n_curr
    s_min = min_successes_for_posterior_threshold(a0, b0, N_total, p0, theta_final)
    if s_min is None:
        return 0.0
    y_needed = s_min - x_curr
    return beta_binomial_cdf_upper_tail(y_needed, m_remain, a_post, b_post)

def compute_interim_futility_cutoffs(a0: float, b0: float, N_total: int, looks_eff: List[int], p0: float, theta_final: float, c_futility: float) -> Dict[int, Optional[int]]:
    cutoffs: Dict[int, Optional[int]] = {}
    for n in looks_eff:
        lo, hi = 0, n
        ans = n + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            ppos = predictive_prob_of_final_success(a0, b0, N_total, mid, n, p0, theta_final)
            if ppos >= c_futility:
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        cutoffs[n] = None if ans == n + 1 else int(ans)
    return cutoffs

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ LOOK SCHEDULE UTILITIES                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_percent_list(s: str) -> List[float]:
    vals: List[float] = []
    if s:
        for tok in s.split(","):
            tok = tok.strip().replace("%", "")
            if tok:
                try:
                    v = float(tok) / 100.0
                    if np.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass
    return vals

def parse_n_list(s: str) -> List[int]:
    vals: List[int] = []
    if s:
        for tok in s.split(","):
            tok = tok.strip()
            if tok:
                try:
                    v = int(round(float(tok)))
                    vals.append(v)
                except Exception:
                    pass
    return vals

def build_looks_with_runin(N: int, run_in: int, mode_label: str, k_total: Optional[int] = None, perc_str: Optional[str] = None, ns_str: Optional[str] = None) -> List[int]:
    looks: List[int] = []
    run_in = int(max(0, min(run_in, N - 1)))
    if run_in > 0:
        looks.append(run_in)

    if mode_label == "None (final only)":
        return looks

    if mode_label == "Equal‑spaced (choose total looks incl. run‑in)":
        K = int(k_total or 0)
        K_rem = max(0, K - (1 if run_in > 0 else 0))
        if K_rem > 0 and (N - run_in) > 0:
            for i in range(1, K_rem + 1):
                n_i = run_in + int(np.floor(i * (N - run_in) / (K_rem + 1)))
                if 0 < n_i < N:
                    looks.append(n_i)
    elif mode_label == "Custom percentages of remaining":
        fracs = parse_percent_list(perc_str or "")
        for f in fracs:
            n_i = run_in + int(np.floor(f * (N - run_in)))
            if 0 < n_i < N:
                looks.append(n_i)
    elif mode_label == "Custom absolute Ns":
        ns = parse_n_list(ns_str or "")
        for n_i in ns:
            if run_in > 0 and n_i == run_in:
                pass
            if run_in < n_i < N:
                looks.append(int(n_i))

    looks = [int(x) for x in looks if 0 < x < N]
    looks = sorted(list(dict.fromkeys(looks)))
    return looks

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ SIMULATION ENGINES                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def simulate_design_eff_only(design: Dict, p: float, U: np.ndarray) -> Dict:
    N = design["N_total"]
    looks_eff = design["looks_eff"]
    run_in_eff = design.get("run_in_eff", 0)
    a0 = design["a0"]; b0 = design["b0"]
    p0 = design["p0"]; theta_final = design["theta_final"]
    theta_interim = design.get("theta_interim", theta_final)
    c_fut = design["c_futility"]
    allow_early = design["allow_early_success"]
    s_min_final = design["s_min_final"]
    x_min_to_continue = design["x_min_to_continue_by_look_eff"]

    n_sims = U.shape[0]
    X = (U[:, :N] < p).astype(np.int16)

    cum_x = np.zeros(n_sims, dtype=np.int32)
    n_curr = 0
    active = np.ones(n_sims, dtype=bool)
    success = np.zeros(n_sims, dtype=bool)
    final_n = np.zeros(n_sims, dtype=np.int32)

    early_succ_by_look = np.zeros(len(looks_eff), dtype=np.int64)
    early_fut_by_look = np.zeros(len(looks_eff), dtype=np.int64)

    if run_in_eff > 0:
        cum_x[active] += np.sum(X[active, :run_in_eff], axis=1)
        n_curr = run_in_eff

    for li, look_n in enumerate(looks_eff):
        add = look_n - n_curr
        if add > 0:
            cum_x[active] += np.sum(X[active, n_curr:look_n], axis=1)
            n_curr = look_n

        if allow_early and active.any():
            a_post, b_post = beta_posterior_params(a0, b0, cum_x[active], n_curr)
            post_probs = 1.0 - beta.cdf(p0, a_post, b_post)
            early_succ = (post_probs >= theta_interim)
            if np.any(early_succ):
                idx_active = np.where(active)[0]
                idx = idx_active[early_succ]
                success[idx] = True
                final_n[idx] = n_curr
                active[idx] = False
                early_succ_by_look[li] += early_succ.sum()
            if not active.any():
                break

        x_min = x_min_to_continue.get(look_n, None)
        if x_min is None:
            idx = np.where(active)[0]
            if idx.size > 0:
                final_n[idx] = n_curr
                active[idx] = False
                early_fut_by_look[li] += idx.size
        else:
            need_continue = cum_x[active] >= x_min
            idx_all_active = np.where(active)[0]
            idx_stop = idx_all_active[~need_continue]
            if idx_stop.size > 0:
                final_n[idx_stop] = n_curr
                active[idx_stop] = False
                early_fut_by_look[li] += idx_stop.size
        if not active.any():
            break

    if active.any():
        add = N - n_curr
        if add > 0:
            cum_x[active] += np.sum(X[active, n_curr:N], axis=1)
            n_curr = N
        succ_final = (cum_x[active] >= s_min_final)
        idx_active = np.where(active)[0]
        success[idx_active[succ_final]] = True
        final_n[idx_active] = N

    reject_rate = success.mean()
    ess = final_n.mean()
    early_succ_rate = early_succ_by_look.sum() / n_sims
    early_fut_rate = early_fut_by_look.sum() / n_sims
    early_stop_rate = early_succ_rate + early_fut_rate

    unique_ns, counts = np.unique(final_n, return_counts=True)
    stop_dist = pd.DataFrame({"N_stop": unique_ns, "Probability": counts / n_sims}).sort_values("N_stop")
    return {
        "reject_rate": float(reject_rate),
        "ess": float(ess),
        "stop_dist": stop_dist,
        "early_succ_by_look_eff": (early_succ_by_look / n_sims).tolist(),
        "early_fut_by_look_eff": (early_fut_by_look / n_sims).tolist(),
        "early_succ_rate": float(early_succ_rate),
        "early_fut_rate": float(early_fut_rate),
        "early_stop_rate": float(early_stop_rate),
    }


def simulate_design_joint(design: Dict, p_eff: float, p_tox: float, U_eff: np.ndarray, U_tox: np.ndarray) -> Dict:
    N = design["N_total"]
    looks_eff = design["looks_eff"]
    looks_saf = design.get("looks_saf", [])
    a0 = design["a0"]; b0 = design["b0"]
    p0 = design["p0"]; theta_final = design["theta_final"]
    theta_interim = design.get("theta_interim", theta_final)
    c_fut = design["c_futility"]
    allow_early = design["allow_early_success"]
    s_min_final = design["s_min_final"]
    x_min_to_continue = design["x_min_to_continue_by_look_eff"]
    tox = design.get("safety", None)

    n_sims = U_eff.shape[0]
    X_eff = (U_eff[:, :N] < p_eff).astype(np.int16)
    X_tox = (U_tox[:, :N] < p_tox).astype(np.int16) if tox is not None else None

    cum_x = np.zeros(n_sims, dtype=np.int32)
    cum_t = np.zeros(n_sims, dtype=np.int32) if tox is not None else None

    n_curr = 0
    active = np.ones(n_sims, dtype=bool)
    success = np.zeros(n_sims, dtype=bool)
    final_n = np.zeros(n_sims, dtype=np.int32)

    eff_early_succ_by_look = np.zeros(len(looks_eff), dtype=np.int64)
    eff_early_fut_by_look = np.zeros(len(looks_eff), dtype=np.int64)
    saf_stop_by_look = np.zeros(len(looks_saf) + 1, dtype=np.int64)

    events = sorted(list(dict.fromkeys(list(looks_eff) + list(looks_saf) + [N])))

    for ev in events:
        if ev > n_curr and active.any():
            cum_x[active] += np.sum(X_eff[active, n_curr:ev], axis=1)
            if tox is not None:
                cum_t[active] += np.sum(X_tox[active, n_curr:ev], axis=1)
            n_curr = ev

        if tox is not None and ev in looks_saf and active.any():
            a_t = tox["a_t0"] + (cum_t[active] if cum_t is not None else 0)
            b_t = tox["b_t0"] + (n_curr - (cum_t[active] if cum_t is not None else 0))
            prob_tox_high = 1.0 - beta.cdf(tox["q_max"], a_t, b_t)
            unsafe = (prob_tox_high >= tox["theta_tox"])
            if np.any(unsafe):
                idx_active = np.where(active)[0]
                idx = idx_active[unsafe]
                final_n[idx] = n_curr
                active[idx] = False
                li_s = looks_saf.index(ev)
                saf_stop_by_look[li_s] += unsafe.sum()
            if not active.any():
                break

        if ev in looks_eff and active.any():
            if allow_early:
                a_post, b_post = beta_posterior_params(a0, b0, cum_x[active], n_curr)
                post_probs = 1.0 - beta.cdf(p0, a_post, b_post)
                early_succ = (post_probs >= theta_interim)
                if np.any(early_succ):
                    idx_active = np.where(active)[0]
                    idx = idx_active[early_succ]
                    success[idx] = True
                    final_n[idx] = n_curr
                    active[idx] = False
                    li_e = looks_eff.index(ev)
                    eff_early_succ_by_look[li_e] += early_succ.sum()
                if not active.any():
                    break

            x_min = x_min_to_continue.get(ev, None)
            li_e = looks_eff.index(ev)
            if x_min is None:
                idx = np.where(active)[0]
                if idx.size > 0:
                    final_n[idx] = n_curr
                    active[idx] = False
                    eff_early_fut_by_look[li_e] += idx.size
            else:
                need_continue = cum_x[active] >= x_min
                idx_all_active = np.where(active)[0]
                idx_stop = idx_all_active[~need_continue]
                if idx_stop.size > 0:
                    final_n[idx_stop] = n_curr
                    active[idx_stop] = False
                    eff_early_fut_by_look[li_e] += idx_stop.size
            if not active.any():
                break

    if active.any():
        if n_curr < N:
            cum_x[active] += np.sum(X_eff[active, n_curr:N], axis=1)
            if tox is not None:
                cum_t[active] += np.sum(X_tox[active, n_curr:N], axis=1)
            n_curr = N

        if tox is not None and active.any():
            a_t = tox["a_t0"] + (cum_t[active] if cum_t is not None else 0)
            b_t = tox["b_t0"] + (n_curr - (cum_t[active] if cum_t is not None else 0))
            prob_tox_high = 1.0 - beta.cdf(tox["q_max"], a_t, b_t)
            unsafe = (prob_tox_high >= tox["theta_tox"])
            if np.any(unsafe):
                idx_active = np.where(active)[0]
                idx = idx_active[unsafe]
                final_n[idx] = n_curr
                active[idx] = False
                saf_stop_by_look[-1] += unsafe.sum()

        if active.any():
            succ_final = (cum_x[active] >= s_min_final)
            idx_active = np.where(active)[0]
            success[idx_active[succ_final]] = True
            final_n[idx_active] = N

    reject_rate = success.mean()
    ess = final_n.mean()

    n_sims = U_eff.shape[0]
    eff_early_succ_rate = eff_early_succ_by_look.sum() / n_sims
    eff_early_fut_rate = eff_early_fut_by_look.sum() / n_sims
    saf_early_rate = saf_stop_by_look[:-1].sum() / n_sims
    any_safety_rate = saf_stop_by_look.sum() / n_sims
    early_stop_rate = eff_early_succ_rate + eff_early_fut_rate + saf_early_rate

    unique_ns, counts = np.unique(final_n, return_counts=True)
    stop_dist = pd.DataFrame({"N_stop": unique_ns, "Probability": counts / n_sims}).sort_values("N_stop")

    return {
        "reject_rate": float(reject_rate),
        "ess": float(ess),
        "safety_stop_prob": float(any_safety_rate),
        "stop_dist": stop_dist,
        "eff_early_succ_by_look": (eff_early_succ_by_look / n_sims).tolist(),
        "eff_early_fut_by_look": (eff_early_fut_by_look / n_sims).tolist(),
        "saf_by_look": (saf_stop_by_look / n_sims).tolist(),
        "eff_early_succ_rate": float(eff_early_succ_rate),
        "eff_early_fut_rate": float(eff_early_fut_rate),
        "saf_early_rate": float(saf_early_rate),
        "early_stop_rate": float(early_stop_rate),
    }

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ SCREENING (FAST)                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def shortlist_designs(param_grid: List[Dict], n_sims_small: int, seed: int, U: Optional[np.ndarray] = None):
    rng = np.random.default_rng(seed)
    if U is None:
        Nmax = max([g["N_total"] for g in param_grid])
        U = rng.uniform(size=(n_sims_small, Nmax))
    rows = []
    designs_built = []

    for g in param_grid:
        N = g["N_total"]
        looks_eff = g["looks_eff"]
        a0 = g["a0"]; b0 = g["b0"]
        p0 = g["p0"]; p1 = g["p1"]
        theta_final = g["theta_final"]; c_futility = g["c_futility"]
        theta_interim = g.get("theta_interim", theta_final)
        allow_early = g["allow_early_success"]

        s_min = min_successes_for_posterior_threshold(a0, b0, N, p0, theta_final)
        if s_min is None:
            continue
        x_min_to_continue = compute_interim_futility_cutoffs(a0, b0, N, looks_eff, p0, theta_final, c_futility)
        design = dict(
            N_total=N, looks_eff=looks_eff, a0=a0, b0=b0, p0=p0,
            theta_final=theta_final, theta_interim=theta_interim,
            c_futility=c_futility, allow_early_success=allow_early,
            s_min_final=s_min, x_min_to_continue_by_look_eff=x_min_to_continue,
            p1=p1, run_in_eff=g.get("run_in_eff", 0)
        )
        res_p0 = simulate_design_eff_only(design, p0, U[:, :N])
        res_p1 = simulate_design_eff_only(design, p1, U[:, :N])

        rows.append({
            "N_total": N,
            "looks_eff": looks_eff,
            "run_in_eff": design["run_in_eff"],
            "theta_final": theta_final,
            "theta_interim": theta_interim,
            "c_futility": c_futility,
            "allow_early_success": allow_early,
            "Type I error @ p0": res_p0["reject_rate"],
            "Power @ p1": res_p1["reject_rate"],
            "ESS @ p0": res_p0["ess"],
            "ESS @ p1": res_p1["ess"],
            "Early stop @ p0 (any)": res_p0["early_stop_rate"],
            "Early success @ p0": res_p0["early_succ_rate"],
            "Early futility @ p0": res_p0["early_fut_rate"],
            "s_min_final": s_min,
            "x_min_to_continue_eff": x_min_to_continue,
        })
        designs_built.append(design)

    df = pd.DataFrame(rows)
    return df, designs_built

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ MONTE CARLO SE/CI HELPERS                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def se_prop(p_hat: float, n: int) -> float:
    p_hat = max(0.0, min(1.0, float(p_hat)))
    return math.sqrt(p_hat * (1 - p_hat) / max(1, n))

def ci95_prop(p_hat: float, n: int) -> Tuple[float, float]:
    se = se_prop(p_hat, n)
    return max(0.0, p_hat - 1.96 * se), min(1.0, p_hat + 1.96 * se)

def ci95_mean(x: np.ndarray) -> Tuple[float, float]:
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
    return m - 1.96 * se, m + 1.96 * se

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ TUNER: BISECTION ON θ_final + JOINT (θ_interim, c_futility)             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def tune_theta_final_bisect(N: int, looks_eff: List[int], a0: float, b0: float, p0: float, p1: float, allow_early_success: bool, c_fut: float, theta_interim: float, run_in_eff: int, alpha_target: float, n_sims: int, seed: int, tol: float = 0.005, max_iter: int = 22) -> Optional[float]:
    """
    Find θ_final so that Type I error at p0 ~= alpha_target using efficacy-only sims.
    Monotonic in θ_final (higher threshold → lower type I).
    Returns None if infeasible in [0.5, 0.999].
    """
    rng = np.random.default_rng(seed)
    U = rng.uniform(size=(n_sims, N))

    lo, hi = 0.50, 0.999
    best = None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        s_min = min_successes_for_posterior_threshold(a0, b0, N, p0, mid)
        if s_min is None:
            # impossible final rule → lower threshold
            hi = mid
            continue
        x_min_map = compute_interim_futility_cutoffs(a0, b0, N, looks_eff, p0, mid, c_fut)
        design = dict(N_total=N, looks_eff=looks_eff, a0=a0, b0=b0, p0=p0, theta_final=mid, theta_interim=float(theta_interim), c_futility=c_fut, allow_early_success=allow_early_success, s_min_final=s_min, x_min_to_continue_by_look_eff=x_min_map, run_in_eff=run_in_eff)
        r0 = simulate_design_eff_only(design, p0, U)
        diff = r0["reject_rate"] - alpha_target
        best = (mid, r0["reject_rate"]) if best is None or abs(diff) < abs(best[1] - alpha_target) else best
        if abs(diff) <= tol:
            return mid
        if diff > 0:  # too liberal → raise threshold
            lo = mid
        else:
            hi = mid
    return best[0] if best else None


def joint_search_theta_interim_cf(N: int, looks_eff: List[int], a0: float, b0: float, p0: float, p1: float, theta_final: float, allow_early_success: bool, run_in_eff: int, objective: str, alpha_cap: float, power_floor: float, n_sims: int, seed: int, coarse_steps: Tuple[float, float] = (0.02, 0.02), refine_steps: Tuple[float, float] = (0.01, 0.01)) -> Optional[Dict]:
    """
    Coordinate-style grid search over (θ_interim, c_futility) with coarse→refine stages.
    Returns best dict with keys: theta_interim, c_futility, metrics.
    """
    rng = np.random.default_rng(seed)
    U = rng.uniform(size=(n_sims, N))

    def eval_pair(ti, cf):
        s_min = min_successes_for_posterior_threshold(a0, b0, N, p0, theta_final)
        if s_min is None:
            return None
        x_min_map = compute_interim_futility_cutoffs(a0, b0, N, looks_eff, p0, theta_final, cf)
        design = dict(N_total=N, looks_eff=looks_eff, a0=a0, b0=b0, p0=p0, theta_final=theta_final, theta_interim=float(ti), c_futility=cf, allow_early_success=allow_early_success, s_min_final=s_min, x_min_to_continue_by_look_eff=x_min_map, run_in_eff=run_in_eff)
        r0 = simulate_design_eff_only(design, p0, U)
        r1 = simulate_design_eff_only(design, p1, U)
        if r0["reject_rate"] <= alpha_cap and r1["reject_rate"] >= power_floor:
            obj = {
                "Min ESS @ p0": r0["ess"],
                "Min ESS @ p1": r1["ess"],
                "Max Power @ p1": -r1["reject_rate"],
                "Max Early stops @ p0": -r0["early_stop_rate"],
            }[objective]
            return dict(theta_interim=ti, c_futility=cf, r0=r0, r1=r1, objective=obj)
        return None

    # Coarse grid region
    ti_vals = np.arange(0.80, 0.995 + 1e-9, coarse_steps[0])
    cf_vals = np.arange(0.00, 0.50 + 1e-9, coarse_steps[1])

    best = None
    for ti in ti_vals:
        for cf in cf_vals:
            cand = eval_pair(ti, cf)
            if cand is None:
                continue
            if (best is None) or (cand["objective"] < best["objective"]):
                best = cand
    if best is None:
        return None

    # Refine around best
    ti0, cf0 = best["theta_interim"], best["c_futility"]
    ti_vals_ref = np.clip(np.arange(ti0 - 2*refine_steps[0], ti0 + 2*refine_steps[0] + 1e-9, refine_steps[0]), 0.5, 0.999)
    cf_vals_ref = np.clip(np.arange(cf0 - 2*refine_steps[1], cf0 + 2*refine_steps[1] + 1e-9, refine_steps[1]), 0.0, 0.5)

    for ti in ti_vals_ref:
        for cf in cf_vals_ref:
            cand = eval_pair(ti, cf)
            if cand is None:
                continue
            if (cand["objective"] < best["objective"]):
                best = cand
    return best

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ PLOTTING                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_lines(df: pd.DataFrame, x: str, y: str, title: str, yerr: Optional[str] = None):
    if _HAS_PLOTLY:
        if yerr is not None and yerr in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[x], y=df[y], mode='lines+markers', name=y))
            if yerr:
                fig.update_traces(error_y=dict(type='data', array=df[yerr], visible=True))
            fig.update_layout(title=title, xaxis_title=x, yaxis_title=y)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.line(df, x=x, y=y, title=title)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader(title)
        st.line_chart(df[[x, y]].set_index(x))

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ PDF EXPORT                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_design_pdf(design: Dict, deep_results: Optional[Dict], compare_df: Optional[pd.DataFrame], tuner_df: Optional[pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    x0, y = 2*cm, H - 2*cm

    def line(text: str, dy: float = 0.6*cm, bold: bool = False):
        nonlocal y
        if y < 3*cm:
            c.showPage(); y = H - 2*cm
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11 if bold else 10)
        c.drawString(x0, y, text)
        y -= dy

    line("Bayesian Single-Arm Design – Summary", bold=True)
    line("")
    line("Design settings", bold=True)
    line(f"N_total: {design.get('N_total')}")
    line(f"Efficacy looks: {design.get('looks_eff')}")
    if design.get('looks_saf') is not None:
        line(f"Safety looks: {design.get('looks_saf')}")
    line(f"Run-in (eff/saf): {design.get('run_in_eff',0)} / {design.get('run_in_saf',0)}")
    line(f"Efficacy prior Beta(a0,b0): {design.get('a0')}, {design.get('b0')}")
    if 'safety' in design:
        s = design['safety']
        line(f"Safety prior Beta(a_t0,b_t0): {s.get('a_t0')}, {s.get('b_t0')}")
        line(f"q_max / θ_tox: {s.get('q_max')} / {s.get('theta_tox')}")
    line(f"p0 / p1: {design.get('p0')} / {design.get('p1')}")
    line(f"θ_final / θ_interim / c_futility: {design.get('theta_final')} / {design.get('theta_interim')} / {design.get('c_futility')}")
    line(f"s_min_final: {design.get('s_min_final')}")

    if deep_results is not None:
        line("")
        line("Deep-dive key metrics", bold=True)
        for k, r in deep_results.items():
            if isinstance(r, dict):
                line(f"{k}: Power/TypeI={r.get('reject_rate'):.3f}, ESS={r.get('ess'):.1f}, Early(any)={r.get('early_stop_rate', float('nan')):.3f}")

    if tuner_df is not None and isinstance(tuner_df, pd.DataFrame) and not tuner_df.empty:
        line("")
        line("Best threshold candidates (top 10)", bold=True)
        for _, row in tuner_df.head(10).iterrows():
            line(f"θf={row['theta_final']:.3f}, θi={row['theta_interim']:.3f}, α={row['type1']:.3f}, Pow={row['power']:.3f}, ESS0={row['ESS_p0']:.1f}")

    if compare_df is not None and isinstance(compare_df, pd.DataFrame) and not compare_df.empty:
        line("")
        line("Compare by N (first 10)", bold=True)
        for _, row in compare_df.head(10).iterrows():
            line(f"N={row['N']}, Pow={row['Power @p1,q_good']:.3f}, ESS1={row['ESS @p1']:.1f}")

    c.showPage()
    c.save()
    return buf.getvalue()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ STREAMLIT UI                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.set_page_config(page_title="Bayesian Single‑Arm Designer (Binary) — v3", layout="wide")
st.title("Bayesian Single‑Arm Monitored Study Designer (Binary Endpoint) — v3")
st.caption("Split efficacy/safety schedules & run‑ins, enhanced tuner (bisection + joint), OC explorer, MC‑SE bands, CSV/JSON/PDF export.")

with st.expander("What this tool does (in simple terms)"):
    st.markdown("Run‑in is treated as Look #1; all run‑in patients count in interim and final analyses. The Tuner first bisections θ_final to match α, then jointly tunes θ_interim & c_futility to optimize your chosen objective while respecting α/power constraints.")

# ── Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("1) Efficacy targets")
col1, col2 = st.sidebar.columns(2)
with col1:
    p0 = st.number_input("Null rate (p₀)", 0.0, 1.0, 0.20, 0.01, format="%.2f")
    a0 = st.number_input("Prior a₀ (eff)", 0.0, None, 1.0, 0.5)
    theta_final = st.number_input("θ_final", 0.5, 0.999, 0.95, 0.01, format="%.3f")
with col2:
    p1 = st.number_input("Target rate (p₁)", 0.0, 1.0, 0.40, 0.01, format="%.2f")
    b0 = st.number_input("Prior b₀ (eff)", 0.0, None, 1.0, 0.5)
    c_futility = st.number_input("c_futility (PPoS)", 0.0, 0.5, 0.05, 0.01, format="%.3f")

with st.sidebar.expander("Prior notes"):
    if (a0 + b0) > 0:
        st.write(f"Prior mean={a0/(a0+b0):.3f}; ESS≈{a0+b0:.1f}")
    else:
        st.write("Use Beta(1,1) for uninformative prior.")

allow_early_success = st.sidebar.checkbox("Allow early success?", value=False)

theta_interim = st.sidebar.number_input("θ_interim", 0.5, 0.999, float(theta_final), 0.01, format="%.3f")

# Safety
st.sidebar.header("2) Safety monitoring")
enable_safety = st.sidebar.checkbox("Enable safety?", True)
if enable_safety:
    c1, c2 = st.sidebar.columns(2)
    with c1:
        a_t0 = st.number_input("Safety a_t0", 0.0, None, 1.0, 0.5)
        q_max = st.number_input("q_max", 0.0, 1.0, 0.15, 0.01, format="%.2f")
    with c2:
        b_t0 = st.number_input("Safety b_t0", 0.0, None, 9.0, 0.5)
        theta_tox = st.number_input("θ_tox", 0.5, 0.999, 0.90, 0.01, format="%.3f")
else:
    a_t0 = b_t0 = q_max = theta_tox = None

# Schedules
st.sidebar.header("3) Efficacy look schedule")
run_in_eff = st.sidebar.number_input("Run‑in eff", 0, 400, 0, 1)
looks_eff_mode_label = st.sidebar.selectbox("Efficacy look timing", ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"], index=1)

k_looks_eff = perc_eff_str = ns_eff_str = None
if looks_eff_mode_label == "Equal‑spaced (choose total looks incl. run‑in)":
    k_looks_eff = st.sidebar.slider("Total efficacy looks (incl. run‑in)", 1, 8, 2, 1)
elif looks_eff_mode_label == "Custom percentages of remaining":
    perc_eff_str = st.sidebar.text_input("Efficacy % of remaining", "33,67")
elif looks_eff_mode_label == "Custom absolute Ns":
    ns_eff_str = st.sidebar.text_input("Efficacy Ns (comma)", "")

st.sidebar.header("4) Safety look schedule")
run_in_saf = st.sidebar.number_input("Run‑in safety", 0, 400, 0, 1)
looks_saf_mode_label = st.sidebar.selectbox("Safety look timing", ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"], index=1)

k_looks_saf = perc_saf_str = ns_saf_str = None
if looks_saf_mode_label == "Equal‑spaced (choose total looks incl. run‑in)":
    k_looks_saf = st.sidebar.slider("Total safety looks (incl. run‑in)", 1, 8, 2, 1)
elif looks_saf_mode_label == "Custom percentages of remaining":
    perc_saf_str = st.sidebar.text_input("Safety % of remaining", "33,67")
elif looks_saf_mode_label == "Custom absolute Ns":
    ns_saf_str = st.sidebar.text_input("Safety Ns (comma)", "")

# Screener settings
st.sidebar.header("5) Screener settings")
N_min, N_max = st.sidebar.slider("N range", 10, 400, (30, 120), 1)
N_step = st.sidebar.number_input("N step", 1, 50, 5, 1)
n_sims_small = st.sidebar.number_input("Sims/design", 100, 200000, 5000, 500)
alpha_max = st.sidebar.number_input("Max α", 0.0, 0.5, 0.10, 0.01, format="%.2f")
power_min = st.sidebar.number_input("Min power", 0.0, 1.0, 0.80, 0.01, format="%.2f")
seed = st.sidebar.number_input("Seed", 1, None, 2026, 1)

# Build candidate grid
Ns = list(range(N_min, N_max + 1, N_step))
param_grid = []
for N in Ns:
    looks_eff = build_looks_with_runin(N, run_in_eff, looks_eff_mode_label, k_total=k_looks_eff, perc_str=perc_eff_str, ns_str=ns_eff_str)
    param_grid.append(dict(N_total=N, looks_eff=looks_eff, a0=a0, b0=b0, p0=p0, p1=p1, theta_final=theta_final, theta_interim=float(theta_interim), c_futility=c_futility, allow_early_success=allow_early_success, run_in_eff=int(run_in_eff)))

# Screener
st.write("### 1) Rapid Screener (efficacy-only)")
@st.cache_data(show_spinner=False)
def _screen(param_grid, n_sims_small, seed, schema_version: str):
    rng = np.random.default_rng(seed)
    Nmax = max([g["N_total"] for g in param_grid])
    U = rng.uniform(size=(n_sims_small, Nmax))
    df, designs = shortlist_designs(param_grid, n_sims_small, seed, U)
    _ = schema_version
    return df, designs

df_screen, designs_built = _screen(param_grid, n_sims_small, seed, SCHEMA_VERSION)
if df_screen.empty:
    st.warning("No viable designs found. Consider adjusting θ_final or increasing N.")
else:
    df_ok = df_screen[(df_screen["Type I error @ p0"] <= alpha_max) & (df_screen["Power @ p1"] >= power_min)].copy()
    cols = ["N_total", "run_in_eff", "looks_eff", "theta_final", "c_futility", "Type I error @ p0", "Power @ p1", "ESS @ p0", "Early stop @ p0 (any)"]
    table_df = df_ok if not df_ok.empty else df_screen
    st.dataframe(table_df[cols].sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True))

# Deep dive build-by-N
st.write("### 2) Deep Dive (enter N)")
N_select = st.number_input("Enter N to deep‑dive", Ns[0], Ns[-1], max(60, Ns[0]), 1)
looks_eff_sel = build_looks_with_runin(N_select, run_in_eff, looks_eff_mode_label, k_total=k_looks_eff, perc_str=perc_eff_str, ns_str=ns_eff_str)
looks_saf_sel = build_looks_with_runin(N_select, run_in_saf, looks_saf_mode_label, k_total=k_looks_saf, perc_str=perc_saf_str, ns_str=ns_saf_str)

s_min_sel = min_successes_for_posterior_threshold(a0, b0, N_select, p0, theta_final)
if s_min_sel is None:
    st.error("Final rule infeasible at this N. Relax θ_final / adjust prior / increase N.")
else:
    x_min_to_continue_sel = compute_interim_futility_cutoffs(a0, b0, N_select, looks_eff_sel, p0, theta_final, c_futility)
    design_sel = dict(N_total=N_select, a0=a0, b0=b0, p0=p0, p1=p1, theta_final=theta_final, theta_interim=float(theta_interim), c_futility=c_futility, allow_early_success=allow_early_success, s_min_final=s_min_sel, looks_eff=looks_eff_sel, looks_saf=looks_saf_sel, run_in_eff=int(run_in_eff), run_in_saf=int(run_in_saf), x_min_to_continue_by_look_eff=x_min_to_continue_sel)
    if enable_safety:
        design_sel["safety"] = dict(a_t0=a_t0, b_t0=b_t0, q_max=q_max, theta_tox=theta_tox)

    with st.expander("Chosen Design Summary", expanded=True):
        st.write({
            "N_total": design_sel["N_total"],
            "looks_eff": design_sel["looks_eff"],
            "looks_saf": design_sel.get("looks_saf", []),
            "run_in_eff": design_sel.get("run_in_eff", 0),
            "run_in_saf": design_sel.get("run_in_saf", 0),
            "theta_final": design_sel["theta_final"],
            "theta_interim": design_sel["theta_interim"],
            "c_futility": design_sel["c_futility"],
            "s_min_final": design_sel["s_min_final"],
        })
        st.caption("Interim continue thresholds (need ≥ x cures to continue):")
        st.dataframe(pd.DataFrame.from_dict(design_sel["x_min_to_continue_by_look_eff"], orient="index", columns=["x ≥ to continue"]))

    st.write("#### Deep‑dive simulation settings")
    colD1, colD2, colD3 = st.columns(3)
    with colD1:
        n_sims_deep = st.number_input("Sims (deep dive)", 2000, 800000, 150000, 5000)
    with colD2:
        seed_deep = st.number_input("Seed (deep)", 1, None, seed + 1, 1)
    with colD3:
        show_ci = st.checkbox("Show 95% CI bands", value=True)

    colQ1, colQ2, colQ3 = st.columns(3)
    with colQ1:
        q_good = st.number_input("q_good", 0.0, 1.0, 0.10, 0.01)
    with colQ2:
        q_bad = st.number_input("q_bad", 0.0, 1.0, 0.20, 0.01)
    with colQ3:
        q_for_OC = st.number_input("q for OC curves", 0.0, 1.0, 0.10, 0.01)

    if st.button("Run DEEP‑DIVE (eff + safety)"):
        rng = np.random.default_rng(seed_deep)
        Ueff = rng.uniform(size=(n_sims_deep, design_sel["N_total"]))
        Utox = rng.uniform(size=(n_sims_deep, design_sel["N_total"]))
        res_p0_qgood = simulate_design_joint(design_sel, p_eff=p0, p_tox=q_good, U_eff=Ueff, U_tox=Utox)
        res_p1_qgood = simulate_design_joint(design_sel, p_eff=p1, p_tox=q_good, U_eff=Ueff, U_tox=Utox)
        res_p1_qbad  = simulate_design_joint(design_sel, p_eff=p1, p_tox=q_bad,  U_eff=Ueff, U_tox=Utox)
        st.session_state["deep_results"] = dict(res_p0_qgood=res_p0_qgood, res_p1_qgood=res_p1_qgood, res_p1_qbad=res_p1_qbad)
        st.session_state["deep_design"] = design_sel

        # Show metrics + MC SE bands
        def se_p(p):
            return se_prop(p, n_sims_deep)
        cols = st.columns(6 if enable_safety else 5)
        t1 = res_p0_qgood['reject_rate']; cols[0].metric("Type I @ p₀,q_good", f"{t1:.3f}", f"±{1.96*se_p(t1):.3f}")
        pwr = res_p1_qgood['reject_rate']; cols[1].metric("Power @ p₁,q_good", f"{pwr:.3f}", f"±{1.96*se_p(pwr):.3f}")
        # ESS CI via stop-dist
        sd = res_p1_qgood['stop_dist']
        if isinstance(sd, pd.DataFrame) and not sd.empty:
            Nvals = sd['N_stop'].to_numpy(); probs = sd['Probability'].to_numpy()
            meanN = (Nvals*probs).sum(); varN = ((Nvals-meanN)**2*probs).sum(); seN = math.sqrt(varN/max(1,n_sims_deep))
        else:
            seN = 0.0
        cols[3].metric("ESS @ p₁,q_good", f"{res_p1_qgood['ess']:.1f}", f"±{1.96*seN:.1f}")
        cols[2].metric("ESS @ p₀,q_good", f"{res_p0_qgood['ess']:.1f}")
        estop = res_p1_qgood['early_stop_rate']; cols[4].metric("Early stop @ p₁ (any)", f"{estop:.3f}", f"±{1.96*se_p(estop):.3f}")
        if enable_safety:
            sprob = res_p1_qbad['safety_stop_prob']; cols[5].metric("P(Safety stop) @ p₁,q_bad", f"{sprob:.3f}", f"±{1.96*se_p(sprob):.3f}")

        # OC & ESS vs p with CIs
        p_grid_min = st.number_input("OC p‑min", 0.0, 1.0, max(0.0, p0-0.15), 0.01)
        p_grid_max = st.number_input("OC p‑max", 0.0, 1.0, min(1.0, p1+0.20), 0.01)
        n_grid = st.slider("# grid points", 5, 40, 15, 1)
        ps = np.linspace(p_grid_min, p_grid_max, n_grid)
        oc, ess_curve, oc_err = [], [], []
        for pp in ps:
            r = simulate_design_joint(design_sel, p_eff=pp, p_tox=q_for_OC, U_eff=Ueff, U_tox=Utox)
            oc.append(r['reject_rate'])
            oc_err.append(1.96*se_prop(r['reject_rate'], n_sims_deep) if show_ci else 0.0)
            ess_curve.append(r['ess'])
        df_oc = pd.DataFrame({"p": ps, "Reject_Prob": oc, "Reject_Prob_err": oc_err, "ESS": ess_curve})
        plot_lines(df_oc, x="p", y="Reject_Prob", title=f"OC vs p (q={q_for_OC:.2f})", yerr=("Reject_Prob_err" if show_ci else None))
        plot_lines(df_oc, x="p", y="ESS", title=f"ESS vs p (q={q_for_OC:.2f})")

# ── 3) Compare panel ───────────────────────────────────────────────────────
st.write("### 3) Compare multiple Ns (deep‑dive, joint with safety)")
with st.expander("Open compare panel"):
    ns_str_compare = st.text_input("Enter Ns (comma)", "60,70,80")
    n_sims_compare = st.number_input("Sims/design (compare)", 5000, 400000, 80000, 5000)
    seed_compare = st.number_input("Seed (compare)", 1, None, seed + 2, 1)
    q_good_cmp = st.number_input("q_good (compare)", 0.0, 1.0, 0.10, 0.01)
    q_bad_cmp = st.number_input("q_bad (compare)", 0.0, 1.0, 0.20, 0.01)

    def _build_eff(N):
        return build_looks_with_runin(N, run_in_eff, looks_eff_mode_label, k_total=k_looks_eff, perc_str=perc_eff_str, ns_str=ns_eff_str)
    def _build_saf(N):
        return build_looks_with_runin(N, run_in_saf, looks_saf_mode_label, k_total=k_looks_saf, perc_str=perc_saf_str, ns_str=ns_saf_str)

    if st.button("Run compare"):
        Ns_cmp = []
        for tok in ns_str_compare.split(','):
            tok = tok.strip()
            if tok:
                try:
                    Ns_cmp.append(int(round(float(tok))))
                except Exception:
                    pass
        Ns_cmp = [n for n in Ns_cmp if n >= 5]
        if not Ns_cmp:
            st.warning("Enter valid integers for N.")
        else:
            rng = np.random.default_rng(seed_compare)
            rows = []
            for Ncmp in Ns_cmp:
                looks_eff_cmp = _build_eff(Ncmp)
                looks_saf_cmp = _build_saf(Ncmp)
                smin_cmp = min_successes_for_posterior_threshold(a0, b0, Ncmp, p0, theta_final)
                if smin_cmp is None:
                    continue
                x_min_cmp = compute_interim_futility_cutoffs(a0, b0, Ncmp, looks_eff_cmp, p0, theta_final, c_futility)
                design_cmp = dict(N_total=Ncmp, looks_eff=looks_eff_cmp, looks_saf=looks_saf_cmp, run_in_eff=int(run_in_eff), run_in_saf=int(run_in_saf), a0=a0, b0=b0, p0=p0, theta_final=theta_final, theta_interim=float(theta_interim), c_futility=c_futility, allow_early_success=allow_early_success, s_min_final=smin_cmp, x_min_to_continue_by_look_eff=x_min_cmp)
                if enable_safety:
                    design_cmp["safety"] = dict(a_t0=a_t0, b_t0=b_t0, q_max=q_max, theta_tox=theta_tox)
                Ueff = rng.uniform(size=(n_sims_compare, Ncmp))
                Utox = rng.uniform(size=(n_sims_compare, Ncmp))
                r_p0 = simulate_design_joint(design_cmp, p_eff=p0, p_tox=q_good_cmp, U_eff=Ueff, U_tox=Utox)
                r_p1 = simulate_design_joint(design_cmp, p_eff=p1, p_tox=q_good_cmp, U_eff=Ueff, U_tox=Utox)
                r_p1_bad = simulate_design_joint(design_cmp, p_eff=p1, p_tox=q_bad_cmp, U_eff=Ueff, U_tox=Utox)
                rows.append(dict(N=Ncmp, looks_eff=looks_eff_cmp, looks_saf=looks_saf_cmp, s_min_final=smin_cmp,
                                 **{"Type I @p0,q_good": r_p0["reject_rate"],
                                    "Power @p1,q_good": r_p1["reject_rate"],
                                    "ESS @p0": r_p0["ess"],
                                    "ESS @p1": r_p1["ess"],
                                    "Early stop (any) @p1": r_p1["early_stop_rate"],
                                    "Early succ @p1": r_p1["eff_early_succ_rate"],
                                    "Early fut @p1": r_p1["eff_early_fut_rate"],
                                    "Early safety @p1": r_p1.get("saf_early_rate", 0.0),
                                    "Safety stop @p1,q_bad (any stage)": r_p1_bad.get("safety_stop_prob", 0.0)}))
            if not rows:
                st.warning("No feasible N produced.")
            else:
                df_cmp = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)
                st.dataframe(df_cmp)
                st.session_state['compare_df'] = df_cmp

# ── 4) Threshold Tuner++ (bisection + joint search) ───────────────────────
st.write("### 4) Threshold Tuner++")
with st.expander("Open Threshold Tuner++"):
    colT1, colT2 = st.columns(2)
    with colT1:
        n_sims_tuner = st.number_input("Sims per eval", 2000, 200000, 40000, 2000)
        seed_tuner = st.number_input("Seed (tuner)", 1, None, seed + 3, 1)
        alpha_target = st.number_input("Target α at p₀ (for θ_final bisection)", 0.0, 0.5, alpha_max, 0.01, format="%.2f")
    with colT2:
        objective = st.selectbox("Optimize for", ["Min ESS @ p0", "Min ESS @ p1", "Max Power @ p1", "Max Early stops @ p0"], index=0)
        power_floor = st.number_input("Constraint: min power", 0.0, 1.0, power_min, 0.01, format="%.2f")
        alpha_cap = st.number_input("Constraint: max α", 0.0, 0.5, alpha_max, 0.01, format="%.2f")

    if st.button("Run Tuner++"):
        looks_eff_tune = build_looks_with_runin(N_select, run_in_eff, looks_eff_mode_label, k_total=k_looks_eff, perc_str=perc_eff_str, ns_str=ns_eff_str)
        # Stage 1: bisection on θ_final
        theta_final_star = tune_theta_final_bisect(N_select, looks_eff_tune, a0, b0, p0, p1, allow_early_success, c_futility, theta_interim, run_in_eff, alpha_target, n_sims_tuner, seed_tuner)
        if theta_final_star is None:
            st.error("Could not find a feasible θ_final that meets the α target. Try relaxing target or increasing N.")
        else:
            st.success(f"Bisection θ_final ≈ {theta_final_star:.3f}")
            # Stage 2: joint search over (θ_interim, c_futility)
            best = joint_search_theta_interim_cf(N_select, looks_eff_tune, a0, b0, p0, p1, theta_final_star, allow_early_success, run_in_eff, objective, alpha_cap, power_floor, n_sims_tuner, seed_tuner)
            if best is None:
                st.warning("No (θ_interim, c_futility) pair met constraints; try expanding ranges or relax constraints.")
            else:
                r0, r1 = best['r0'], best['r1']
                st.write({
                    'theta_final*': float(theta_final_star),
                    'theta_interim*': float(best['theta_interim']),
                    'c_futility*': float(best['c_futility']),
                    'Type I @ p0': r0['reject_rate'],
                    'Power @ p1': r1['reject_rate'],
                    'ESS @ p0': r0['ess'],
                    'ESS @ p1': r1['ess'],
                })
                if st.button("Apply tuned thresholds to sidebar"):
                    st.session_state['theta_final'] = float(theta_final_star)
                    st.session_state['theta_interim'] = float(best['theta_interim'])
                    st.session_state['c_futility'] = float(best['c_futility'])
                    st.success("Applied tuned θ_final, θ_interim, c_futility to sidebar.")
                st.session_state['tuner_df'] = pd.DataFrame([{
                    'theta_final': float(theta_final_star),
                    'theta_interim': float(best['theta_interim']),
                    'c_futility': float(best['c_futility']),
                    'type1': r0['reject_rate'], 'power': r1['reject_rate'],
                    'ESS_p0': r0['ess'], 'ESS_p1': r1['ess'], 'objective': best['objective']
                }])

# ── 5) OC Explorer ─────────────────────────────────────────────────────────
st.write("### 5) OC Explorer")
with st.expander("Open OC Explorer"):
    if 'deep_design' not in st.session_state:
        st.info("Run a deep‑dive first to lock a design.")
    else:
        design_locked = st.session_state['deep_design']
        colO1, colO2 = st.columns(2)
        with colO1:
            p_min = st.number_input("p_min", 0.0, 1.0, max(0.0, p0-0.20), 0.01)
            p_max = st.number_input("p_max", 0.0, 1.0, min(1.0, p1+0.25), 0.01)
            p_points = st.slider("# p points", 3, 25, 9, 1)
        with colO2:
            q_min = st.number_input("q_min", 0.0, 1.0, max(0.0, (q_max or 0.10)-0.10), 0.01)
            q_max_g = st.number_input("q_max (grid)", 0.0, 1.0, min(1.0, (q_max or 0.10)+0.20), 0.01)
            q_points = st.slider("# q points", 3, 25, 7, 1)
        n_sims_oc = st.number_input("Sims per cell", 2000, 200000, 50000, 2000)
        seed_oc = st.number_input("Seed (OC)", 1, None, seed+4, 1)
        if st.button("Run OC Explorer"):
            rng = np.random.default_rng(seed_oc)
            Ueff = rng.uniform(size=(n_sims_oc, design_locked['N_total']))
            Utox = rng.uniform(size=(n_sims_oc, design_locked['N_total']))
            ps = np.linspace(p_min, p_max, p_points)
            qs = np.linspace(q_min, q_max_g, q_points)
            oc_rows = []
            for pp in ps:
                for qq in qs:
                    r = simulate_design_joint(design_locked, p_eff=pp, p_tox=qq, U_eff=Ueff, U_tox=Utox)
                    oc_rows.append(dict(p=pp, q=qq, reject_rate=r['reject_rate'], ess=r['ess'], safety_stop_any=r['safety_stop_prob'], early_any=r['early_stop_rate']))
            df_grid = pd.DataFrame(oc_rows)
            st.dataframe(df_grid)
            st.session_state['oc_grid_df'] = df_grid
            if _HAS_PLOTLY:
                piv = df_grid.pivot(index='q', columns='p', values='reject_rate')
                fig = px.imshow(piv.values, x=piv.columns.round(3), y=piv.index.round(3), aspect='auto', origin='lower', color_continuous_scale='Viridis', title='Pr(Declare efficacy) over (p,q)')
                st.plotly_chart(fig, use_container_width=True)

# ── 6) Export: JSON, CSV ZIP, and PDF ──────────────────────────────────────
st.write("### 6) Export settings and results (JSON / CSVs / PDF)")
export_bundle = {
    "design": st.session_state.get('deep_design', {}),
    "screening_table": None if df_screen.empty else df_screen.to_dict(orient='list'),
    "deep_dive": st.session_state.get('deep_results', None),
    "compare": st.session_state.get('compare_df', None),
    "tuner": None if 'tuner_df' not in st.session_state else st.session_state['tuner_df'].to_dict(orient='list'),
    "oc_grid": None if 'oc_grid_df' not in st.session_state else st.session_state['oc_grid_df'].to_dict(orient='list'),
}
json_bytes = json.dumps(export_bundle, default=lambda o: o if isinstance(o, (int,float,str,bool,type(None))) else str(o)).encode('utf-8')
st.download_button("Download JSON bundle", data=json_bytes, file_name="design_and_results_v3.json", mime="application/json")

buf = io.BytesIO()
with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
    if not df_screen.empty:
        zf.writestr("screening.csv", df_screen.to_csv(index=False))
    deep = st.session_state.get('deep_results', None)
    if isinstance(deep, dict):
        rows = []
        for k,v in deep.items():
            if isinstance(v, dict) and 'stop_dist' in v and isinstance(v['stop_dist'], pd.DataFrame):
                zf.writestr(f"deep_{k}_stopdist.csv", v['stop_dist'].to_csv(index=False))
            if isinstance(v, dict):
                rows.append({"scenario": k, **{kk: vv for kk, vv in v.items() if kk != 'stop_dist'}})
        if rows:
            zf.writestr("deep_summary.csv", pd.DataFrame(rows).to_csv(index=False))
    if 'compare_df' in st.session_state:
        zf.writestr("compare.csv", st.session_state['compare_df'].to_csv(index=False))
    if 'tuner_df' in st.session_state:
        zf.writestr("tuner_candidates.csv", st.session_state['tuner_df'].to_csv(index=False))
    if 'oc_grid_df' in st.session_state:
        zf.writestr("oc_grid.csv", st.session_state['oc_grid_df'].to_csv(index=False))

st.download_button("Download ZIP (CSVs)", data=buf.getvalue(), file_name="design_results_v3_csv.zip", mime="application/zip")

if st.button("Download protocol-ready PDF"):
    design_pdf = st.session_state.get('deep_design', None)
    pdf_bytes = make_design_pdf(design_pdf or {}, st.session_state.get('deep_results', None), st.session_state.get('compare_df', None), st.session_state.get('tuner_df', None))
    st.download_button("Click to download PDF", data=pdf_bytes, file_name="design_summary_v3.pdf", mime="application/pdf")
