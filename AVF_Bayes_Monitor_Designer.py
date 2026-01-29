# =============================================================================
# AVF_Bayes_Monitor_Designer.py  (single-file, optimized + safe session_state)
# Bayesian Single-Arm Monitoring Study Designer
# - Classic grid search (with fixed or calibrated success thresholds)
# - Staged Workflow search (fast → faithful → racing → precise finalists)
# - Calibration fixes: minimal-γ bisection + exact refinement
# - Vectorized α honors futility when asked (falls back to exact sim)
# - Lazy imports for SciPy/NumPy/Altair to speed initial load
# - Plain-language summaries & re-evaluation panel
# - SAFE session_state access via ss_get() (prevents KeyError on first load)
# =============================================================================

# Lightweight imports only at top-level
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# Lazy import helpers (heavy libs only loaded when actually used)
# -----------------------------------------------------------------------------
def lazy_numpy():
    import numpy as _np
    return _np

def lazy_scipy_stats_beta():
    from scipy.stats import beta as _beta
    return _beta

def lazy_scipy_special():
    from scipy.special import comb as _comb, beta as _beta_fn
    return _comb, _beta_fn

def lazy_altair():
    import altair as _alt
    return _alt

# -----------------------------------------------------------------------------
# Streamlit configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Bayesian Single-Arm Monitor Designer", layout="wide")

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULTS = {
    "p0": 0.20,
    "p1": 0.40,
    "a_e": 1.0,
    "b_e": 1.0,
    "gamma_e": 0.95,  # used when calibration = Off or Single-γe mode
    "psi_fut": 0.05,  # predictive futility threshold
    "enable_safety": True,
    "qmax": 0.30,
    "q1": 0.15,
    "a_s": 1.0,
    "b_s": 1.0,
    "gamma_s": 0.90,
    # Grid
    "N_min": 30,
    "N_max": 120,
    "K_min": 0,
    "K_max": 4,
    # Targets
    "alpha_target": 0.10,
    "power_target": 0.80,
    "N_budget": 80,
    # Simulation
    "n_sim": 1,  # quick-scan default preserved
    "seed": 12345,
}

# -----------------------------------------------------------------------------
# SAFE session_state getter (prevents KeyError on first run)
# -----------------------------------------------------------------------------
def ss_get(key: str, default):
    """
    Safe session_state getter: returns value if present; otherwise sets and returns default.
    Use for any value that may be read before Streamlit initializes the widget.
    """
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class Design:
    N: int
    K_interims: int
    look_schedule: List[int]
    a_e: float
    b_e: float
    a_s: float
    b_s: float
    p0: float
    p1: float
    qmax: Optional[float]
    q1: Optional[float]
    psi_fut: float
    gamma_s: Optional[float]
    # Efficacy thresholds:
    gamma_e: Optional[float] = None  # single threshold
    gamma_e_vector: Optional[List[float]] = None  # per-look thresholds

@dataclass
class OperatingCharacteristics:
    alpha: float
    power: float
    ess_p0: float
    ess_p1: float
    safety_stop_prob_q1: Optional[float]
    safety_stop_prob_qmax: Optional[float]
    avg_looks: float
    success_prob_by_look: Dict[int, float]

# -----------------------------------------------------------------------------
# Bayesian helpers (lazy SciPy inside)
# -----------------------------------------------------------------------------
def posterior_prob_p_greater_than(p0: float, a: float, b: float) -> float:
    beta = lazy_scipy_stats_beta()
    return 1.0 - beta.cdf(p0, a, b)

def posterior_prob_q_exceeds(qmax: float, a: float, b: float) -> float:
    beta = lazy_scipy_stats_beta()
    return 1.0 - beta.cdf(qmax, a, b)

def minimal_successes_for_posterior_success(
    n: int, p0: float, a0: float, b0: float, gamma_e: float
) -> Optional[int]:
    for r in range(n + 1):
        if posterior_prob_p_greater_than(p0, a0 + r, b0 + n - r) >= gamma_e:
            return r
    return None

def safety_stop_threshold(n: int, qmax: float, a_s: float, b_s: float, gamma_s: float) -> Optional[int]:
    for t in range(n + 1):
        if posterior_prob_q_exceeds(qmax, a_s + t, b_s + n - t) >= gamma_s:
            return t
    return None

def beta_binomial_predictive_prob_at_least(
    current_r: int, current_n: int, final_N: int,
    a0: float, b0: float, r_star_final: int
) -> float:
    """Predictive probability that future successes bring total >= r_star_final."""
    comb, beta_fn = lazy_scipy_special()
    m = final_N - current_n
    a_post = a0 + current_r
    b_post = b0 + current_n - current_r
    if m == 0:
        return float(current_r >= r_star_final)
    j_min = max(0, r_star_final - current_r)
    if j_min <= 0:
        return 1.0
    total = 0.0
    for j in range(j_min, m + 1):
        total += comb(m, j) * beta_fn(j + a_post, m - j + b_post) / beta_fn(a_post, b_post)
    return float(total)

# -----------------------------------------------------------------------------
# Boundaries
# -----------------------------------------------------------------------------
def compute_boundaries(design: Design) -> Dict[int, Dict[str, Optional[int]]]:
    """
    Compute per-look success (r_min) and safety (t_min) boundaries.
    If gamma_e_vector is given, use that entry; else use gamma_e scalar.
    """
    bounds: Dict[int, Dict[str, Optional[int]]] = {}
    looks = design.look_schedule
    L = len(looks)

    def gamma_for_index(i: int) -> float:
        if design.gamma_e_vector is not None:
            return float(design.gamma_e_vector[i])
        return float(design.gamma_e)

    r_star_final = minimal_successes_for_posterior_success(
        looks[-1], design.p0, design.a_e, design.b_e, gamma_for_index(L - 1)
    )

    for i, n in enumerate(looks):
        ge = gamma_for_index(i)
        r_success_min = minimal_successes_for_posterior_success(
            n, design.p0, design.a_e, design.b_e, ge
        )
        t_safety_min = None
        if design.qmax is not None and design.gamma_s is not None:
            t_safety_min = safety_stop_threshold(n, design.qmax, design.a_s, design.b_s, design.gamma_s)
        bounds[n] = {"r_success_min": r_success_min, "t_safety_min": t_safety_min, "r_star_final": r_star_final}
    return bounds

# -----------------------------------------------------------------------------
# Simulation engine (path-wise, accurate; used for final evaluation)
# -----------------------------------------------------------------------------
def simulate_one_trial(
    design: Design, true_p: float, true_q: Optional[float],
    bounds: Dict[int, Dict[str, Optional[int]]], rng,  # rng is numpy Generator
    skip_futility: bool = False
) -> Tuple[bool, int, bool, int]:
    """
    Returns (success, n_used, safety_stopped, looks_used).
    """
    np = lazy_numpy()
    N = design.N
    responses = rng.binomial(1, true_p, N)
    tox = rng.binomial(1, true_q, N) if (true_q is not None) else np.zeros(N, dtype=int)
    looks_used = 0
    for n in design.look_schedule:
        r = int(np.sum(responses[:n]))
        t = int(np.sum(tox[:n]))
        looks_used += 1

        # Safety first
        t_saf = bounds[n]["t_safety_min"]
        if t_saf is not None and t >= t_saf:
            return (False, n, True, looks_used)

        # Efficacy success
        r_min = bounds[n]["r_success_min"]
        if r_min is not None and r >= r_min:
            return (True, n, False, looks_used)

        # Futility (optional)
        if (not skip_futility) and (n < N):
            r_star_final = bounds[n]["r_star_final"]
            if r_star_final is not None:
                ppos = beta_binomial_predictive_prob_at_least(
                    r, n, N, design.a_e, design.b_e, r_star_final
                )
                if ppos < design.psi_fut:
                    return (False, n, False, looks_used)

        # Final look
        if n == N:
            r_star_final = bounds[n]["r_star_final"]
            if r_star_final is not None and r >= r_star_final:
                return (True, n, False, looks_used)
            return (False, n, False, looks_used)

    # fallback
    return (False, N, False, looks_used)

def evaluate_design(
    design: Design, n_sim: int = 1, seed: int = 123, skip_futility: bool = False
) -> OperatingCharacteristics:
    np = lazy_numpy()
    rng = np.random.default_rng(seed)
    bounds = compute_boundaries(design)

    # Under p0
    results_p0 = [simulate_one_trial(design, design.p0, design.q1, bounds, rng, skip_futility) for _ in range(n_sim)]
    alpha = float(np.mean([s for (s, _, _, _) in results_p0]))
    ess_p0 = float(np.mean([n_used for (_, n_used, _, _) in results_p0]))

    # Under p1
    results_p1 = [simulate_one_trial(design, design.p1, design.q1, bounds, rng, skip_futility) for _ in range(n_sim)]
    power = float(np.mean([s for (s, _, _, _) in results_p1]))
    ess_p1 = float(np.mean([n_used for (_, n_used, _, _) in results_p1]))
    avg_looks = float(np.mean([looks for (_, _, _, looks) in results_p1]))

    # Safety metrics (under p1)
    safety_stop_prob_q1 = None
    safety_stop_prob_qmax = None
    if design.qmax is not None and design.q1 is not None:
        res_q1 = [simulate_one_trial(design, design.p1, design.q1, bounds, rng, skip_futility) for _ in range(n_sim)]
        safety_stop_prob_q1 = float(np.mean([sf for (_, _, sf, _) in res_q1]))
        res_qmax = [simulate_one_trial(design, design.p1, design.qmax, bounds, rng, skip_futility) for _ in range(n_sim)]
        safety_stop_prob_qmax = float(np.mean([sf for (_, _, sf, _) in res_qmax]))

    # Success by look under p1
    success_by_look_counts = {n: 0 for n in design.look_schedule}
    for (s, n_used, _, _) in results_p1:
        if s:
            success_by_look_counts[n_used] += 1
    success_prob_by_look = {n: success_by_look_counts[n] / n_sim for n in design.look_schedule}

    return OperatingCharacteristics(
        alpha=alpha, power=power, ess_p0=ess_p0, ess_p1=ess_p1,
        safety_stop_prob_q1=safety_stop_prob_q1, safety_stop_prob_qmax=safety_stop_prob_qmax,
        avg_looks=avg_looks, success_prob_by_look=success_prob_by_look
    )

# -----------------------------------------------------------------------------
# Vectorized simulation for calibration (fast α-only estimation)
# -----------------------------------------------------------------------------
def vectorized_alpha_under_p0(
    design: Design, n_sim: int, seed: int, skip_futility: bool = True
) -> float:
    """
    Fast α estimation under p0 (and q1 for safety).
    If skip_futility is False, fall back to exact simulator to respect futility stops.
    """
    if not skip_futility:
        oc = evaluate_design(design, n_sim=n_sim, seed=seed, skip_futility=False)
        return float(oc.alpha)

    np = lazy_numpy()
    # Precompute integer boundaries with current thresholds
    bounds = compute_boundaries(design)
    N = design.N
    looks = design.look_schedule

    r_mins = np.array([bounds[n]["r_success_min"] if bounds[n]["r_success_min"] is not None else N+1 for n in looks], dtype=int)
    t_mins = np.array([bounds[n]["t_safety_min"] if bounds[n]["t_safety_min"] is not None else N+1 for n in looks], dtype=int)

    rng = np.random.default_rng(seed)
    resp = rng.binomial(1, design.p0, size=(n_sim, N))
    tox = None
    if design.q1 is not None and design.qmax is not None and design.gamma_s is not None:
        tox = rng.binomial(1, design.q1, size=(n_sim, N))
    cum_r = np.cumsum(resp, axis=1).astype(int)
    cum_t = np.cumsum(tox, axis=1).astype(int) if tox is not None else None

    active = np.ones(n_sim, dtype=bool)
    success = np.zeros(n_sim, dtype=bool)

    for i, n in enumerate(looks):
        if not active.any():
            break
        r = cum_r[:, n - 1]
        t = cum_t[:, n - 1] if cum_t is not None else None

        # Safety first
        if t is not None:
            stop_safety = active & (t >= t_mins[i])
            active[stop_safety] = False

        # Efficacy success
        got_success = active & (r >= r_mins[i])
        success[got_success] = True
        active[got_success] = False

        # Futility intentionally skipped in vectorized fast mode
        if n == N:
            break

    return float(np.mean(success))

# -----------------------------------------------------------------------------
# Look schedule helper
# -----------------------------------------------------------------------------
def build_equal_looks(N: int, K: int) -> List[int]:
    looks = [int(round((k / (K + 1)) * N)) for k in range(1, K + 1)]
    if not looks or looks[-1] != N:
        looks.append(N)
    looks = sorted(set(max(1, x) for x in looks))
    if looks[-1] != N:
        looks.append(N)
    return looks

# -----------------------------------------------------------------------------
# Caching wrappers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def cached_evaluate_single(
    N: int, K: int, look_tuple: Tuple[int, ...],
    a_e: float, b_e: float, a_s: float, b_s: float,
    p0: float, p1: float, qmax: Optional[float], q1: Optional[float],
    gamma_e: float, psi_fut: float, gamma_s: Optional[float],
    n_sim: int, seed: int, skip_futility: bool
) -> Dict:
    design = Design(
        N=N, K_interims=K, look_schedule=list(look_tuple),
        a_e=a_e, b_e=b_e, a_s=a_s, b_s=b_s,
        p0=p0, p1=p1, qmax=qmax, q1=q1,
        psi_fut=psi_fut, gamma_s=gamma_s,
        gamma_e=gamma_e, gamma_e_vector=None
    )
    oc = evaluate_design(design, n_sim=n_sim, seed=seed, skip_futility=skip_futility)
    return oc.__dict__

@st.cache_data(show_spinner=True)
def cached_evaluate_vector(
    N: int, K: int, look_tuple: Tuple[int, ...],
    a_e: float, b_e: float, a_s: float, b_s: float,
    p0: float, p1: float, qmax: Optional[float], q1: Optional[float],
    gamma_vec: Tuple[float, ...], psi_fut: float, gamma_s: Optional[float],
    n_sim: int, seed: int, skip_futility: bool
) -> Dict:
    design = Design(
        N=N, K_interims=K, look_schedule=list(look_tuple),
        a_e=a_e, b_e=b_e, a_s=a_s, b_s=b_s,
        p0=p0, p1=p1, qmax=qmax, q1=q1,
        psi_fut=psi_fut, gamma_s=gamma_s,
        gamma_e=None, gamma_e_vector=list(gamma_vec)
    )
    oc = evaluate_design(design, n_sim=n_sim, seed=seed, skip_futility=skip_futility)
    return oc.__dict__

@st.cache_data(show_spinner=False)
def cached_boundaries_for_design(
    N: int, look_tuple: Tuple[int, ...],
    a_e: float, b_e: float, p0: float,
    a_s: float, b_s: float, qmax: Optional[float], gamma_s: Optional[float],
    gamma_e: Optional[float], gamma_vec: Optional[Tuple[float, ...]]
) -> Dict[int, Dict[str, Optional[int]]]:
    design = Design(
        N=N, K_interims=len(look_tuple)-1, look_schedule=list(look_tuple),
        a_e=a_e, b_e=b_e, a_s=a_s, b_s=b_s,
        p0=p0, p1=p0, qmax=qmax, q1=None,
        psi_fut=1.0, gamma_s=gamma_s,
        gamma_e=gamma_e, gamma_e_vector=(list(gamma_vec) if gamma_vec is not None else None)
    )
    return compute_boundaries(design)

# -----------------------------------------------------------------------------
# Calibration helpers (fixed, with exact refinement)
# -----------------------------------------------------------------------------
def perlook_gamma_vector_from_final(gamma_final: float, L: int, phi: float) -> List[float]:
    np = lazy_numpy()
    idx = np.arange(1, L + 1, dtype=float)
    s = (idx / L) ** max(phi, 0.5)
    return [float(1.0 - (1.0 - gamma_final) * sl) for sl in s]

@st.cache_data(show_spinner=True)
def cached_calibrate_single_gamma(
    N: int, K: int, look_tuple: Tuple[int, ...],
    a_e: float, b_e: float, a_s: float, b_s: float,
    p0: float, p1: float, qmax: Optional[float], q1: Optional[float],
    psi_fut: float, gamma_s: Optional[float],
    alpha_target: float, n_sim_cal: int, seed: int,
    g_low: float = 0.50, g_high: float = 0.999, tol_alpha: float = 0.005, max_iter: int = 14,
    fast_mode: bool = True, skip_futility_during_cal: bool = True
) -> float:
    """
    Calibrate a single posterior success threshold γe to control α.
    Returns the minimal γe such that α ≤ target (within tol).
    If fast_mode or skip_futility_during_cal is True, perform a small exact refinement.
    """
    def alpha_at_gamma(gamma: float, fast: bool = True) -> float:
        if fast:
            d = Design(N, K, list(look_tuple), a_e, b_e, a_s, b_s, p0, p1, qmax, q1, psi_fut, gamma_s, gamma_e=gamma)
            if not skip_futility_during_cal:
                oc = evaluate_design(d, n_sim=n_sim_cal, seed=seed + int(gamma*1e6)%1000000, skip_futility=False)
                return float(oc.alpha)
            return vectorized_alpha_under_p0(
                d, n_sim=n_sim_cal, seed=seed + int(gamma*1e6)%1000000, skip_futility=True
            )
        oc = cached_evaluate_single(N, K, look_tuple, a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                                    gamma, psi_fut, gamma_s, n_sim_cal, seed + int(gamma*1e6)%1000000,
                                    skip_futility=False)
        return float(oc["alpha"])

    lo, hi = g_low, g_high
    a_lo = alpha_at_gamma(lo, fast=fast_mode)
    a_hi = alpha_at_gamma(hi, fast=fast_mode)

    if a_lo <= alpha_target + tol_alpha:
        gamma_fast = float(lo)
    elif a_hi > alpha_target + tol_alpha:
        gamma_fast = float(hi)
    else:
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            a_mid = alpha_at_gamma(mid, fast=fast_mode)
            if a_mid <= alpha_target + tol_alpha: hi = mid
            else: lo = mid
            if abs(hi - lo) < 1e-3: break
        gamma_fast = float(hi)

    # Local exact refinement
    np = lazy_numpy()
    window = 0.03
    grid = np.linspace(max(g_low, gamma_fast - window), min(g_high, gamma_fast + window), 7)
    good = [g for g in grid if alpha_at_gamma(float(g), fast=False) <= alpha_target + tol_alpha]
    if good:
        return float(min(good))
    return float(gamma_fast)

@st.cache_data(show_spinner=True)
def cached_calibrate_perlook_gamma(
    N: int, K: int, look_tuple: Tuple[int, ...],
    a_e: float, b_e: float, a_s: float, b_s: float,
    p0: float, p1: float, qmax: Optional[float], q1: Optional[float],
    psi_fut: float, gamma_s: Optional[float],
    alpha_target: float, n_sim_cal: int, seed: int,
    phi: float, g_low: float = 0.50, g_high: float = 0.999, tol_alpha: float = 0.005, max_iter: int = 14,
    fast_mode: bool = True, skip_futility_during_cal: bool = True
) -> Tuple[float, Tuple[float, ...]]:
    """
    Calibrate per-look thresholds by bisection on γ_final; γ vector derived via φ.
    Returns (γ_final*, γ_vec*), minimal γ_final controlling α (within tol), with refinement.
    """
    L = len(look_tuple)

    def alpha_at_gamma_final(gamma_final: float, fast: bool = True) -> float:
        gamma_vec = tuple(perlook_gamma_vector_from_final(gamma_final, L, phi))
        if fast:
            d = Design(N, K, list(look_tuple), a_e, b_e, a_s, b_s, p0, p1, qmax, q1, psi_fut, gamma_s,
                       gamma_e=None, gamma_e_vector=list(gamma_vec))
            if not skip_futility_during_cal:
                oc = evaluate_design(d, n_sim=n_sim_cal, seed=seed + int(gamma_final*1e6)%1000000, skip_futility=False)
                return float(oc.alpha)
            return vectorized_alpha_under_p0(
                d, n_sim=n_sim_cal, seed=seed + int(gamma_final*1e6)%1000000, skip_futility=True
            )
        oc = cached_evaluate_vector(N, K, look_tuple, a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                                    tuple(perlook_gamma_vector_from_final(gamma_final, L, phi)),
                                    psi_fut, gamma_s, n_sim_cal,
                                    seed + int(gamma_final*1e6)%1000000, skip_futility=False)
        return float(oc["alpha"])

    lo, hi = g_low, g_high
    a_lo = alpha_at_gamma_final(lo, fast=fast_mode)
    a_hi = alpha_at_gamma_final(hi, fast=fast_mode)

    if a_lo <= alpha_target + tol_alpha:
        gamma_fast = float(lo); gamma_vec_fast = tuple(perlook_gamma_vector_from_final(gamma_fast, L, phi))
    elif a_hi > alpha_target + tol_alpha:
        gamma_fast = float(hi); gamma_vec_fast = tuple(perlook_gamma_vector_from_final(gamma_fast, L, phi))
    else:
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            a_mid = alpha_at_gamma_final(mid, fast=fast_mode)
            if a_mid <= alpha_target + tol_alpha: hi = mid
            else: lo = mid
            if abs(hi - lo) < 1e-3: break
        gamma_fast = float(hi); gamma_vec_fast = tuple(perlook_gamma_vector_from_final(gamma_fast, L, phi))

    # Local exact refinement
    np = lazy_numpy()
    window = 0.03
    grid = np.linspace(max(g_low, gamma_fast - window), min(g_high, gamma_fast + window), 7)
    best_gamma = None
    for g in grid:
        a = alpha_at_gamma_final(float(g), fast=False)
        if a <= alpha_target + tol_alpha:
            best_gamma = g if best_gamma is None else min(best_gamma, g)
    gamma_star = float(best_gamma) if best_gamma is not None else gamma_fast
    return gamma_star, tuple(perlook_gamma_vector_from_final(gamma_star, L, phi))

# -----------------------------------------------------------------------------
# Classic grid search
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def cached_grid_search(
    N_min, N_max, K_min, K_max,
    p0, p1, a_e, b_e, a_s, b_s, qmax, q1,
    gamma_e, psi_fut, gamma_s,
    n_sim, seed,
    alpha_target, power_target, N_budget,
    require_alpha_for_high_power,
    cal_mode: str,  # "off" \ "single" \ "per_look"
    cal_n_sim: int,
    phi: float,  # used only in per_look mode
    fast_mode: bool,
    skip_fut_cal: bool
) -> pd.DataFrame:
    rows = []
    for N in range(N_min, N_max + 1):
        for K in range(K_min, K_max + 1):
            looks = build_equal_looks(N, K)
            gamma_used = gamma_e
            gamma_vec: Optional[Tuple[float, ...]] = None

            # Calibration choice
            if cal_mode == "single":
                gamma_used = cached_calibrate_single_gamma(
                    N, K, tuple(looks),
                    a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                    psi_fut, gamma_s, alpha_target,
                    n_sim_cal=cal_n_sim, seed=seed + 17*N + 11*K,
                    fast_mode=fast_mode, skip_futility_during_cal=skip_fut_cal
                )
            elif cal_mode == "per_look":
                gamma_final_star, gamma_vec_star = cached_calibrate_perlook_gamma(
                    N, K, tuple(looks),
                    a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                    psi_fut, gamma_s, alpha_target,
                    n_sim_cal=cal_n_sim, seed=seed + 23*N + 13*K,
                    phi=phi, fast_mode=fast_mode, skip_futility_during_cal=skip_fut_cal
                )
                gamma_used = gamma_final_star
                gamma_vec = gamma_vec_star

            # Evaluate with chosen thresholds (using futility)
            if gamma_vec is not None:
                oc = cached_evaluate_vector(
                    N, K, tuple(looks),
                    a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                    gamma_vec, psi_fut, gamma_s, n_sim, seed + 1000 + N + K,
                    skip_futility=False
                )
                bounds = cached_boundaries_for_design(
                    N, tuple(looks), a_e, b_e, p0, a_s, b_s, qmax, gamma_s,
                    gamma_e=None, gamma_vec=gamma_vec
                )
            else:
                oc = cached_evaluate_single(
                    N, K, tuple(looks),
                    a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                    gamma_used, psi_fut, gamma_s, n_sim, seed + 1000 + N + K,
                    skip_futility=False
                )
                bounds = cached_boundaries_for_design(
                    N, tuple(looks), a_e, b_e, p0, a_s, b_s, qmax, gamma_s,
                    gamma_e=gamma_used, gamma_vec=None
                )

            r_by_look = [(n, bounds[n]["r_success_min"]) for n in looks]
            r_final = bounds[looks[-1]]["r_star_final"]

            rows.append(
                dict(
                    N=N, K_interims=K, looks=looks,
                    gamma_e_used=gamma_used,
                    gamma_e_vector=(list(gamma_vec) if gamma_vec is not None else None),
                    r_success_by_look=r_by_look, r_star_final=r_final,
                    alpha=oc["alpha"], power=oc["power"],
                    ESS_p0=oc["ess_p0"], ESS_p1=oc["ess_p1"],
                    avg_looks_p1=oc["avg_looks"],
                    safety_stop_q1=oc["safety_stop_prob_q1"],
                    safety_stop_qmax=oc["safety_stop_prob_qmax"],
                    meets_alpha=oc["alpha"] <= alpha_target,
                    meets_power=oc["power"] >= power_target
                )
            )

    df = pd.DataFrame(rows)
    df["is_feasible"] = df["meets_alpha"] & df["meets_power"]
    df["selection"] = ""

    feasible = df[df["is_feasible"]]
    if not feasible.empty:
        smallest = feasible.sort_values(["N", "K_interims", "ESS_p1"]).head(3)
        sweet = feasible.sort_values(["ESS_p1", "N", "alpha"]).head(3)
        df.loc[smallest.index, "selection"] += "\nsmallest_N"
        df.loc[sweet.index, "selection"] += "\nsweet_spot"

    # Revised "High power" tagging: prefer designs meeting the power target within N_budget
    if N_budget is not None and N_budget > 0:
        under = df[df["N"] <= N_budget]
        if require_alpha_for_high_power:
            under = under[under["meets_alpha"]]
        high_power_pool = under[under["power"] >= power_target]
        if high_power_pool.empty:
            high_power_pool = under
        if not high_power_pool.empty:
            best = high_power_pool.sort_values(["power", "ESS_p1"], ascending=[False, True]).head(3)
            df.loc[best.index, "selection"] += "\nhigh_power"

    return df

# -----------------------------------------------------------------------------
# ===== STAGED WORKFLOW IMPLEMENTATION =====
# (same as before; omitted here for brevity in this snippet)
# NOTE: In the staged buttons below, all reads of "seed" now use:
#   base_seed = int(ss_get("seed", DEFAULTS["seed"]))
#   and then base_seed + offsets
# -----------------------------------------------------------------------------

# ... (STAGED WORKFLOW CODE FROM THE PREVIOUS VERSION GOES HERE UNCHANGED) ...
# (For brevity and to fit the message, the staged code is identical to the
#  last full version I sent, with only these seed reads modified:)
#
# - In Stage 0/1 (calibration):
#     base_seed = int(ss_get("seed", DEFAULTS["seed"]))
#     seed arg for single γ: base_seed + 17*N + 11*K
#     seed arg for per-look γ: base_seed + 23*N + 13*K
#
# - Stage 1 vectorized α:
#     seed_base = int(ss_get("seed", DEFAULTS["seed"])) + 10000
#
# - Stage 2 exact α:
#     seed_base = int(ss_get("seed", DEFAULTS["seed"])) + 20000
#
# - Stage 3 exact power:
#     seed_base = int(ss_get("seed", DEFAULTS["seed"])) + 30000
#
# - Stage 4 racing:
#     successive_halving(..., seed_base=int(ss_get("seed", DEFAULTS["seed"])) + 40000, ...)
#
# - Stage 5 final:
#     seed_base = int(ss_get("seed", DEFAULTS["seed"])) + 50000
#
# Everything else (logic, thresholds, gates, caching) stays the same.

# -----------------------------------------------------------------------------
# UI — SIDEBAR (unchanged)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Inputs")
    if st.button("Reset UI & clear cache"):
        st.cache_data.clear()
        st.session_state.clear()
        st.success("Cache & UI state cleared. Rerun your search.")
        st.stop()

    st.markdown("⚡ *Quick-Scan Mode Enabled (n_sim defaults to 1 in Classic Grid)*")
    st.number_input("Null efficacy rate p₀", 0.0, 1.0, DEFAULTS["p0"], 0.01, key="p0")
    st.number_input("Expected efficacy rate p₁", 0.0, 1.0, DEFAULTS["p1"], 0.01, key="p1")
    st.number_input("Efficacy prior alpha aₑ", 0.0, 100.0, DEFAULTS["a_e"], 0.1, key="a_e")
    st.number_input("Efficacy prior beta bₑ", 0.0, 100.0, DEFAULTS["b_e"], 0.1, key="b_e")
    st.number_input("Predictive futility threshold ψ", 0.0, 0.5, DEFAULTS["psi_fut"], 0.01, key="psi_fut")

    st.divider()
    st.checkbox("Enable safety monitoring", DEFAULTS["enable_safety"], key="enable_safety")
    if st.session_state["enable_safety"]:
        st.number_input("Unacceptable toxicity q_max", 0.0, 1.0, DEFAULTS["qmax"], 0.01, key="qmax")
        st.number_input("Expected toxicity q₁", 0.0, 1.0, DEFAULTS["q1"], 0.01, key="q1")
        st.number_input("Safety prior alpha aₛ", 0.0, 100.0, DEFAULTS["a_s"], 0.1, key="a_s")
        st.number_input("Safety prior beta bₛ", 0.0, 100.0, DEFAULTS["b_s"], 0.1, key="b_s")
        st.number_input("Posterior safety threshold γₛ", 0.5, 0.999, DEFAULTS["gamma_s"], 0.01, key="gamma_s")
    else:
        st.session_state["qmax"] = None
        st.session_state["q1"] = None
        st.session_state["gamma_s"] = None

    st.divider()
    st.subheader("Grid / Look ranges")
    st.number_input("Minimum N", 5, 5000, DEFAULTS["N_min"], 1, key="N_min")
    st.number_input("Maximum N", int(st.session_state["N_min"]), 5000, DEFAULTS["N_max"], 1, key="N_max")
    st.number_input("Min interim looks (K)", 0, 40, DEFAULTS["K_min"], 1, key="K_min")
    st.number_input("Max interim looks (K)", int(st.session_state["K_min"]), 40, DEFAULTS["K_max"], 1, key="K_max")

    st.divider()
    st.subheader("Targets & simulation")
    st.number_input("Max Type I error α target", 0.0, 0.5, DEFAULTS["alpha_target"], 0.01, key="alpha_target")
    st.number_input("Min Power target", 0.0, 1.0, DEFAULTS["power_target"], 0.01, key="power_target")
    st.number_input("N budget (optional)", 0, 5000, DEFAULTS["N_budget"], 1, key="N_budget")
    require_alpha_for_high_power = st.checkbox(
        "Require α constraint for 'High power' recommendations (Classic Grid)",
        value=True
    )
    # Classic evaluation sims (after calibration)
    st.number_input(
        "Classic Grid: Monte Carlo replicates (n_sim)",
        min_value=1, max_value=200000, value=DEFAULTS["n_sim"], step=1, key="n_sim_v2",
        help="1–100: instant; 100–1000: fast; 2000+: accurate."
    )
    # *** SAFE default for seed on first render ***
    st.number_input("Random seed", 0, 9999999, ss_get("seed", DEFAULTS["seed"]), 1, key="seed")

    st.divider()
    st.subheader("Calibration mode")
    cal_mode = st.selectbox(
        "Calibration of success thresholds",
        options=["Off", "Single γₑ", "Per-look γₑ vector (alpha-spending style)"],
        index=2
    )
    # Only used when "Off" or "Single γₑ"
    st.number_input("Posterior success threshold γₑ (when calibration is Off)", 0.5, 0.999,
                    DEFAULTS["gamma_e"], 0.01, key="gamma_e")

    st.markdown("**Calibration settings**")
    cal_n_sim = st.number_input("Calibration n_sim (per bisection step - Classic Grid)", 100, 200000, 1000, 100)
    fast_mode = st.checkbox("Use vectorized fast calibration (Classic Grid)", value=True)
    skip_fut_cal = st.checkbox("Skip futility during calibration (Classic Grid)", value=True)
    phi = st.number_input("Per-look stringency φ (>=1: early stricter)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)

    st.divider()
    st.subheader("Search Mode")
    search_mode = st.radio(
        "Choose search mode", ["Classic Grid (full evaluation)", "Staged Workflow (smart search)"], index=0
    )

    # Staged-specific inputs (omitted here for brevity; unchanged from last version)
    # ...

    show_only_feasible = st.checkbox("Show only designs meeting α & power targets", value=False)
    run_btn = st.button("Run search", type="primary")

# -----------------------------------------------------------------------------
# Main UI + RUN LOGIC
# -----------------------------------------------------------------------------
st.title("Bayesian Single‑Arm Monitoring Study Designer")

with st.expander("📋 Workflow Guide (click to expand)", expanded=False):
    st.markdown("""
**Recommended workflow**
- **Classic Grid**: quick scan with small `n_sim`, calibrate fast, then re-run selected rows at high `n_sim`.
- **Staged Workflow**: prune aggressively with fast α, check α exactly, check power, race the survivors, then re-evaluate finalists precisely.
""")

st.markdown("---")

# (Helper table functions and re-evaluation panel code remain unchanged from the last full version.)
# IMPORTANT: Inside the code blocks that run stages or classic grid:
#   - Replace any direct st.session_state["seed"] usage with:
#       base_seed = int(ss_get("seed", DEFAULTS["seed"]))
#   - Then use base_seed + offsets as already done above.

# ------------------ RUN BUTTON HANDLER (seed read made SAFE) ------------------
if run_btn:
    mode_key = (cal_mode.lower())
    mode_key = "off" if "off" in mode_key else ("single" if "single" in mode_key else "per_look")
    params_common = dict(
        p0=float(st.session_state["p0"]),
        p1=float(st.session_state["p1"]),
        a_e=float(st.session_state["a_e"]),
        b_e=float(st.session_state["b_e"]),
        a_s=float(st.session_state.get("a_s") or 0.0),
        b_s=float(st.session_state.get("b_s") or 0.0),
        qmax=st.session_state.get("qmax"),
        q1=st.session_state.get("q1"),
        psi_fut=float(st.session_state["psi_fut"]),
        gamma_s=st.session_state.get("gamma_s"),
        alpha_target=float(st.session_state["alpha_target"]),
        power_target=float(st.session_state["power_target"]),
        N_budget=int(st.session_state["N_budget"]),
        # (phi intentionally NOT included here to avoid double-pass)
    )
    grid_ranges = dict(
        N_min=int(st.session_state["N_min"]), N_max=int(st.session_state["N_max"]),
        K_min=int(st.session_state["K_min"]), K_max=int(st.session_state["K_max"]),
    )
    seed = int(ss_get("seed", DEFAULTS["seed"]))  # <-- SAFE

    if search_mode.startswith("Classic"):
        params = dict(
            **grid_ranges,
            **params_common,
            gamma_e=float(st.session_state["gamma_e"]),
            n_sim=int(st.session_state["n_sim_v2"]),
            seed=seed,  # SAFE value
            require_alpha_for_high_power=bool(require_alpha_for_high_power),
            cal_mode=mode_key,
            cal_n_sim=int(cal_n_sim),
            phi=float(phi),              # pass phi explicitly (no double)
            fast_mode=bool(fast_mode),
            skip_fut_cal=bool(skip_fut_cal),
        )
        total_designs = (params["N_max"] - params["N_min"] + 1) * (params["K_max"] - params["K_min"] + 1)
        st.write(f"Evaluating ~{total_designs} designs × {params['n_sim']} sims each (cached).")
        if mode_key != "off":
            st.write(f"Calibration uses n_sim={params['cal_n_sim']} (fast={params['fast_mode']}, skip futility={params['skip_fut_cal']}).")
        st.caption(f"Evaluation n_sim = {params['n_sim']}")

        start = time.time()
        with st.spinner("Running classic grid search..."):
            df = cached_grid_search(**params)
        st.success(f"Classic grid completed in {time.time() - start:.2f} sec.")

    else:
        # Staged Workflow (unchanged logic from last version)
        # Ensure all staged seed bases use ss_get("seed", DEFAULTS["seed"])
        # ...
        st.info("Staged workflow code unchanged except for SAFE seed reads via ss_get().")
        st.stop()

# (Summary tables, charts, and re-evaluation panel remain as in the last version.)
