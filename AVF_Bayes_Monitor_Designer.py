# =============================================================================
# AVF_Bayes_Monitor_Designer.py
# Bayesian Single-Arm Monitoring Study Designer
# - Classic grid search (with fixed or calibrated success thresholds)
# - Staged Workflow search (fast → faithful → racing → precise finalists)
# - Calibration fixes: minimal-γ bisection + exact refinement
# - Vectorized α honors futility if requested (falls back to exact sim)
# - Plain-language summaries & re-evaluation panel
# =============================================================================
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable
import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import comb, beta as beta_fn
from scipy.stats import beta

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
# Bayesian helpers
# -----------------------------------------------------------------------------
def posterior_prob_p_greater_than(p0: float, a: float, b: float) -> float:
    return 1.0 - beta.cdf(p0, a, b)

def posterior_prob_q_exceeds(qmax: float, a: float, b: float) -> float:
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
    If gamma_e_vector is given, use the entry aligned to the look index; otherwise use gamma_e scalar.
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
    bounds: Dict[int, Dict[str, Optional[int]]], rng: np.random.Generator,
    skip_futility: bool = False
) -> Tuple[bool, int, bool, int]:
    """
    Returns (success, n_used, safety_stopped, looks_used).
    skip_futility=True is used during calibration for speed; evaluation uses futility.
    """
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

    rng = np.random.default_rng(seed)
    N = design.N
    looks = design.look_schedule

    # Precompute integer boundaries with current thresholds
    bounds = compute_boundaries(design)
    r_mins = np.array([bounds[n]["r_success_min"] if bounds[n]["r_success_min"] is not None else N+1 for n in looks], dtype=int)
    t_mins = np.array([bounds[n]["t_safety_min"] if bounds[n]["t_safety_min"] is not None else N+1 for n in looks], dtype=int)

    # Generate trial matrices
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

        # (Futility is intentionally skipped in vectorized fast mode)
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
# Calibration helpers (fixed)
# -----------------------------------------------------------------------------
def perlook_gamma_vector_from_final(gamma_final: float, L: int, phi: float) -> List[float]:
    """
    Construct per-look posterior thresholds:
    gamma_l = 1 - (1 - gamma_final) * s_l
    with s_l = (l/L)^phi, l = 1..L (final l=L gives s_L=1 => gamma_L=gamma_final).
    phi >= 1 makes early looks stricter (O'Brien-Fleming-like as phi grows).
    """
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
    If fast_mode or skip_futility_during_cal is True, performs a small exact refinement.
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
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            a_mid = alpha_at_gamma(mid, fast=fast_mode)
            if a_mid <= alpha_target + tol_alpha: hi = mid
            else: lo = mid
            if abs(hi - lo) < 1e-3: break
        gamma_fast = float(hi)

    if fast_mode or skip_futility_during_cal:
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
    Returns (γ_final*, γ_vec*), minimal γ_final controlling α (within tol), with refinement if needed.
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
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            a_mid = alpha_at_gamma_final(mid, fast=fast_mode)
            if a_mid <= alpha_target + tol_alpha: hi = mid
            else: lo = mid
            if abs(hi - lo) < 1e-3: break
        gamma_fast = float(hi); gamma_vec_fast = tuple(perlook_gamma_vector_from_final(gamma_fast, L, phi))

    if fast_mode or skip_futility_during_cal:
        window = 0.03
        grid = np.linspace(max(g_low, gamma_fast - window), min(g_high, gamma_fast + window), 7)
        best_gamma = None
        for g in grid:
            a = alpha_at_gamma_final(float(g), fast=False)
            if a <= alpha_target + tol_alpha:
                best_gamma = g if best_gamma is None else min(best_gamma, g)
        if best_gamma is not None:
            return float(best_gamma), tuple(perlook_gamma_vector_from_final(float(best_gamma), L, phi))

    return float(gamma_fast), gamma_vec_fast

# -----------------------------------------------------------------------------
# Classic grid search (unchanged API; tags tightened later)
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
# -----------------------------------------------------------------------------
# Small utilities
def _se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 0.0) / max(n, 1)))

def _dominates(r_by_look_A: List[Tuple[int,int]], r_by_look_B: List[Tuple[int,int]]) -> bool:
    """
    Return True if A dominates B (>= r_min at all looks and > at least one), same look grid assumed.
    """
    ge = [ra >= rb for (_, ra), (_, rb) in zip(r_by_look_A, r_by_look_B)]
    gt = [ra >  rb for (_, ra), (_, rb) in zip(r_by_look_A, r_by_look_B)]
    return all(ge) and any(gt)

def _implausible_final_rstar(N: int, p1: float, r_star_final: Optional[int]) -> bool:
    if r_star_final is None:
        return False
    mu = N * p1
    var = N * p1 * (1 - p1)
    thr = mu + 3.0 * np.sqrt(max(var, 1e-12))
    return r_star_final > thr

def _parse_schedule(schedule_str: str) -> List[Tuple[int, float]]:
    """
    Parse racing schedule like "1000@0.5,2000@0.5,5000@1.0" -> [(1000,0.5),(2000,0.5),(5000,1.0)]
    """
    out = []
    if not schedule_str:
        return out
    parts = [s.strip() for s in schedule_str.split(",") if s.strip()]
    for p in parts:
        if "@ " in p: p = p.replace("@ ", "@")
        n, k = p.split("@")
        out.append((int(float(n)), float(k)))
    return out

# Candidate wrapper with cumulative stats (for racing)
@dataclass
class EvalStats:
    n_eval: int = 0  # number of batches added (info only)
    n_sims: int = 0
    alpha_sum: float = 0.0
    power_sum: float = 0.0
    ess_p0_sum: float = 0.0
    ess_p1_sum: float = 0.0
    def alpha_mean(self): return self.alpha_sum / max(1, self.n_sims)
    def power_mean(self): return self.power_sum / max(1, self.n_sims)
    def ESS_p0_mean(self): return self.ess_p0_sum / max(1, self.n_sims)
    def ESS_p1_mean(self): return self.ess_p1_sum / max(1, self.n_sims)

class Candidate:
    def __init__(self, design: Design, looks: List[int], gamma_used: Optional[float], gamma_vec: Optional[Tuple[float,...]]):
        self.design = design
        self.looks = looks
        self.gamma_used = gamma_used
        self.gamma_vec = gamma_vec
        self.stats = EvalStats()
        self.bounds = compute_boundaries(design)

    def add_batch(self, n_batch: int, seed: int):
        oc = evaluate_design(self.design, n_sim=n_batch, seed=seed, skip_futility=False)
        self.stats.n_eval += 1
        self.stats.n_sims += n_batch
        self.stats.alpha_sum += oc.alpha * n_batch
        self.stats.power_sum += oc.power * n_batch
        self.stats.ess_p0_sum += oc.ess_p0 * n_batch
        self.stats.ess_p1_sum += oc.ess_p1 * n_batch

# Gates
def _gate_alpha_point(alpha_hat: float, n: int, alpha_target: float, z: float) -> str:
    if n <= 0: return "undecided"
    lo = alpha_hat - z * _se(alpha_hat, n)
    hi = alpha_hat + z * _se(alpha_hat, n)
    if hi <= alpha_target: return "promote"
    if lo >  alpha_target: return "prune"
    return "gray"

def _gate_power_point(power_hat: float, n: int, power_target: float, z: float) -> str:
    if n <= 0: return "undecided"
    lo = power_hat - z * _se(power_hat, n)
    hi = power_hat + z * _se(power_hat, n)
    if lo >= power_target: return "promote"
    if hi <  power_target: return "prune"
    return "gray"

# Stage 0: Deterministic pruning (dominance + implausible final boundary)
def _stage0_filter(rows_stage0: List[dict], p1: float) -> List[dict]:
    # Implausible final r*
    keep = []
    for row in rows_stage0:
        if not _implausible_final_rstar(row["N"], p1, row["r_star_final"]):
            keep.append(row)
    rows_stage0 = keep
    # Dominance: compare within groups of (N, looks)
    out = []
    by_key: Dict[Tuple[int, Tuple[int,...]], List[dict]] = {}
    for row in rows_stage0:
        key = (row["N"], tuple(row["looks"]))
        by_key.setdefault(key, []).append(row)
    for key, group in by_key.items():
        # prune dominated rows by r_success_by_look
        marks = [True]*len(group)
        for i in range(len(group)):
            for j in range(len(group)):
                if i == j or not marks[i]: continue
                A = group[i]["r_success_by_look"]
                B = group[j]["r_success_by_look"]
                if _dominates(A, B):
                    marks[i] = False
        out.extend([g for g, m in zip(group, marks) if m])
    return out

# Successive-halving racing
def successive_halving(cands: List[Candidate], schedule: List[Tuple[int,float]], seed_base: int,
                       alpha_target: float) -> List[Candidate]:
    pool = list(cands)
    rng_seed = seed_base
    for (batch, keep_frac) in schedule:
        if len(pool) <= 1: break
        for c in pool:
            c.add_batch(int(batch), rng_seed)
            rng_seed += 1
        # Rank by: power high, ESS_p1 low, alpha below target (penalize overflow)
        scored = []
        for c in pool:
            alpha_over = max(0.0, c.stats.alpha_mean() - alpha_target)
            scored.append((c, c.stats.power_mean(), -c.stats.ESS_p1_mean(), -alpha_over))
        scored.sort(key=lambda t: (t[1], t[2], t[3]), reverse=True)
        k = max(1, int(len(scored) * keep_frac))
        pool = [c for (c, *_ ) in scored[:k]]
    return pool

# -----------------------------------------------------------------------------
# ===== STAGED WORKFLOW DRIVER =====
# -----------------------------------------------------------------------------
def run_staged_workflow(
    N_min, N_max, K_min, K_max,
    p0, p1, a_e, b_e, a_s, b_s, qmax, q1,
    psi_fut, gamma_s,
    seed,
    alpha_target, power_target, N_budget,
    cal_mode: str, cal_n_sim_s1: int, phi: float,
    fast_mode_s1: bool, skip_fut_s1: bool,
    # gates & budgets
    z_early: float, z_strict: float,
    stage1_nsim: int, stage2_alpha_nsim: int, stage3_power_nsim: int,
    racing_schedule: List[Tuple[int,float]],
    carry_limit_stage1: int, carry_limit_stage2: int, carry_limit_stage3: int
) -> pd.DataFrame:
    """
    Multi-stage workflow:
    0) Deterministic pruning
    1) Fast α screen (vectorized, futility skipped)
    2) Exact α (futility ON)
    3) Exact power
    4) Racing (successive halving)
    5) Final precise eval (implicit in racing result + later re-eval panel)
    """
    # ---------- Build initial grid & (fast) calibrate ----------
    rows = []
    for N in range(N_min, N_max + 1):
        for K in range(K_min, K_max + 1):
            looks = build_equal_looks(N, K)
            # Calibration per design (fast mode)
            if cal_mode.lower().startswith("off"):
                gamma_used = DEFAULTS["gamma_e"]
                gamma_vec = None
            elif "single" in cal_mode.lower():
                gamma_used = cached_calibrate_single_gamma(
                    N, K, tuple(looks), a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                    psi_fut, gamma_s, alpha_target,
                    n_sim_cal=cal_n_sim_s1, seed=seed + 17*N + 11*K,
                    fast_mode=fast_mode_s1, skip_futility_during_cal=skip_fut_s1
                )
                gamma_vec = None
            else:
                gamma_final, gamma_vec_star = cached_calibrate_perlook_gamma(
                    N, K, tuple(looks), a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                    psi_fut, gamma_s, alpha_target,
                    n_sim_cal=cal_n_sim_s1, seed=seed + 23*N + 13*K,
                    phi=phi, fast_mode=fast_mode_s1, skip_futility_during_cal=skip_fut_s1
                )
                gamma_used = gamma_final
                gamma_vec = gamma_vec_star

            # Boundaries for Stage 0 pruning
            if gamma_vec is not None:
                bounds = cached_boundaries_for_design(
                    N, tuple(looks), a_e, b_e, p0, a_s, b_s, qmax, gamma_s,
                    gamma_e=None, gamma_vec=tuple(gamma_vec)
                )
            else:
                bounds = cached_boundaries_for_design(
                    N, tuple(looks), a_e, b_e, p0, a_s, b_s, qmax, gamma_s,
                    gamma_e=float(gamma_used), gamma_vec=None
                )
            r_by_look = [(n, bounds[n]["r_success_min"]) for n in looks]
            r_final = bounds[looks[-1]]["r_star_final"]

            rows.append(dict(
                N=N, K_interims=K, looks=looks,
                gamma_e_used=float(gamma_used),
                gamma_e_vector=(list(gamma_vec) if gamma_vec is not None else None),
                r_success_by_look=r_by_look, r_star_final=r_final
            ))

    # ---------- Stage 0: Deterministic pruning ----------
    rows0 = _stage0_filter(rows, p1=p1)
    # Optional hard N budget filter (but allow surveying above-budget if wanted)
    if N_budget is not None and N_budget > 0:
        rows0 = [r for r in rows0 if r["N"] <= max(N_budget, N_min)]

    # ---------- Stage 1: Fast α (vectorized, futility skipped) ----------
    s1 = []
    for row in rows0:
        N, K, looks = int(row["N"]), int(row["K_interims"]), list(row["looks"])
        if row["gamma_e_vector"] is not None:
            oc_alpha = vectorized_alpha_under_p0(
                Design(N, K, looks, a_e, b_e, a_s, b_s, p0, p1, qmax, q1, psi_fut, gamma_s,
                       gamma_e=None, gamma_e_vector=list(row["gamma_e_vector"])),
                n_sim=stage1_nsim, seed=seed + 10000 + N + K, skip_futility=True
            )
        else:
            oc_alpha = vectorized_alpha_under_p0(
                Design(N, K, looks, a_e, b_e, a_s, b_s, p0, p1, qmax, q1, psi_fut, gamma_s,
                       gamma_e=float(row["gamma_e_used"]), gamma_e_vector=None),
                n_sim=stage1_nsim, seed=seed + 10000 + N + K, skip_futility=True
            )
        gate = _gate_alpha_point(oc_alpha, stage1_nsim, alpha_target, z_early)
        s1.append({**row, "alpha_hat_s1": oc_alpha, "n_s1": stage1_nsim, "gate_s1": gate})
    # Keep promoted + gray; limit carry to top by optimistic margin
    s1_keep = [r for r in s1 if r["gate_s1"] in ("promote", "gray")]
    # Rank by optimistic alpha margin (lower is better)
    s1_keep.sort(key=lambda r: (r["alpha_hat_s1"], r["N"]))
    s1_keep = s1_keep[:max(1, carry_limit_stage1)]

    # ---------- Stage 2: Exact α (futility ON) ----------
    s2 = []
    for row in s1_keep:
        N, K, looks = int(row["N"]), int(row["K_interims"]), list(row["looks"])
        if row["gamma_e_vector"] is not None:
            oc = cached_evaluate_vector(
                N, K, tuple(looks), a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                tuple(row["gamma_e_vector"]), psi_fut, gamma_s,
                n_sim=stage2_alpha_nsim, seed=seed + 20000 + N + K, skip_futility=False
            )
        else:
            oc = cached_evaluate_single(
                N, K, tuple(looks), a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                float(row["gamma_e_used"]), psi_fut, gamma_s,
                n_sim=stage2_alpha_nsim, seed=seed + 20000 + N + K, skip_futility=False
            )
        gate = _gate_alpha_point(float(oc["alpha"]), stage2_alpha_nsim, alpha_target, z_strict)
        s2.append({**row, "alpha_hat_s2": float(oc["alpha"]), "n_s2": stage2_alpha_nsim,
                   "ESS_p0_s2": float(oc["ess_p0"]), "gate_s2": gate})
    s2_keep = [r for r in s2 if r["gate_s2"] in ("promote", "gray")]
    # limit carry
    s2_keep.sort(key=lambda r: (r["alpha_hat_s2"], r.get("ESS_p0_s2", 1e9)))
    s2_keep = s2_keep[:max(1, carry_limit_stage2)]

    # ---------- Stage 3: Exact power ----------
    s3 = []
    for row in s2_keep:
        N, K, looks = int(row["N"]), int(row["K_interims"]), list(row["looks"])
        if row["gamma_e_vector"] is not None:
            oc = cached_evaluate_vector(
                N, K, tuple(looks), a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                tuple(row["gamma_e_vector"]), psi_fut, gamma_s,
                n_sim=stage3_power_nsim, seed=seed + 30000 + N + K, skip_futility=False
            )
        else:
            oc = cached_evaluate_single(
                N, K, tuple(looks), a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                float(row["gamma_e_used"]), psi_fut, gamma_s,
                n_sim=stage3_power_nsim, seed=seed + 30000 + N + K, skip_futility=False
            )
        gate = _gate_power_point(float(oc["power"]), stage3_power_nsim, power_target, z_strict)
        s3.append({**row, "alpha_hat_s2": row["alpha_hat_s2"], "power_hat_s3": float(oc["power"]),
                   "ESS_p1_s3": float(oc["ess_p1"]), "n_s3": stage3_power_nsim, "gate_s3": gate})
    s3_keep = [r for r in s3 if r["gate_s3"] in ("promote", "gray")]
    s3_keep.sort(key=lambda r: (-r["power_hat_s3"], r.get("ESS_p1_s3", 1e9)))
    s3_keep = s3_keep[:max(1, carry_limit_stage3)]

    # ---------- Stage 4: Racing ----------
    # Initialize candidates
    cands: List[Candidate] = []
    for row in s3_keep:
        N, K, looks = int(row["N"]), int(row["K_interims"]), list(row["looks"])
        design = Design(
            N, K, looks, a_e, b_e, a_s, b_s, p0, p1, qmax, q1, psi_fut, gamma_s,
            gamma_e=None if row["gamma_e_vector"] is not None else float(row["gamma_e_used"]),
            gamma_e_vector=(list(row["gamma_e_vector"]) if row["gamma_e_vector"] is not None else None)
        )
        cands.append(Candidate(design, looks, row["gamma_e_used"], tuple(row["gamma_e_vector"]) if row["gamma_e_vector"] is not None else None))
    finalists = successive_halving(cands, racing_schedule, seed_base=seed + 40000, alpha_target=alpha_target)

    # ---------- Assemble final DataFrame ----------
    rows_out = []
    for c in finalists:
        N, K, looks = c.design.N, c.design.K_interims, c.design.look_schedule
        rows_out.append(dict(
            N=N, K_interims=K, looks=looks,
            gamma_e_used=c.design.gamma_e, gamma_e_vector=c.design.gamma_e_vector,
            r_success_by_look=[(n, c.bounds[n]["r_success_min"]) for n in looks],
            r_star_final=c.bounds[looks[-1]]["r_star_final"],
            alpha=c.stats.alpha_mean(), power=c.stats.power_mean(),
            ESS_p0=c.stats.ESS_p0_mean(), ESS_p1=c.stats.ESS_p1_mean(),
            avg_looks_p1=np.nan,  # (optional to compute path-wise again)
            safety_stop_q1=None, safety_stop_qmax=None,
            meets_alpha=(c.stats.alpha_mean() <= alpha_target),
            meets_power=(c.stats.power_mean() >= power_target),
            selection="staged_finalist"
        ))

    df = pd.DataFrame(rows_out)
    if df.empty:
        return df

    # Tagging similar to classic path
    df["is_feasible"] = df["meets_alpha"] & df["meets_power"]
    feasible = df[df["is_feasible"]]
    if not feasible.empty:
        smallest = feasible.sort_values(["N", "K_interims", "ESS_p1"]).head(3)
        sweet = feasible.sort_values(["ESS_p1", "N", "alpha"]).head(3)
        df.loc[smallest.index, "selection"] += "\nsmallest_N"
        df.loc[sweet.index, "selection"] += "\nsweet_spot"
        if N_budget is not None and N_budget > 0:
            under = df[df["N"] <= N_budget]
            # Require α by default; prefer meeting power target too
            under = under[under["meets_alpha"]]
            pool = under[under["power"] >= power_target]
            if pool.empty: pool = under
            if not pool.empty:
                best = pool.sort_values(["power", "ESS_p1"], ascending=[False, True]).head(3)
                df.loc[best.index, "selection"] += "\nhigh_power"
    return df

# -----------------------------------------------------------------------------
# UI — SIDEBAR
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
    st.number_input("Random seed", 0, 9999999, DEFAULTS["seed"], 1, key="seed")

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

    if search_mode.startswith("Staged"):
        st.markdown("**Staged Workflow Settings**")
        st.caption("Stage 1 uses fast α (vectorized, futility skipped), then exact α, then exact power, then racing.")
        z_early = st.number_input("Early z-score for gates (Stage 1)", 0.0, 5.0, 1.64, 0.01)
        z_strict = st.number_input("Strict z-score for gates (Stages 2-3)", 0.0, 5.0, 1.96, 0.01)
        stage1_nsim = st.number_input("Stage 1 n_sim (fast α, vectorized)", 50, 5000, 300, 50)
        stage2_alpha_nsim = st.number_input("Stage 2 n_sim (exact α)", 100, 10000, 1000, 100)
        stage3_power_nsim = st.number_input("Stage 3 n_sim (exact power)", 100, 10000, 1000, 100)

        st.markdown("**Racing schedule (successive halving)**")
        schedule_str = st.text_input("Schedule (e.g., '1000@0.5,2000@0.5,5000@1.0')", value="1000@0.5,2000@0.5,5000@1.0")
        s_sched = _parse_schedule(schedule_str)
        carry_limit_stage1 = st.number_input("Max designs to carry after Stage 1", 1, 2000, 50, 1)
        carry_limit_stage2 = st.number_input("Max designs to carry after Stage 2", 1, 2000, 30, 1)
        carry_limit_stage3 = st.number_input("Max designs to carry before Racing", 1, 2000, 15, 1)

        st.markdown("**Staged calibration (fast initial)**")
        cal_n_sim_s1 = st.number_input("Fast calibration n_sim (Stage 1)", 100, 200000, 500, 100)
        fast_mode_s1 = st.checkbox("Stage 1 calibration uses fast vectorized α", value=True)
        skip_fut_s1 = st.checkbox("Stage 1 calibration skips futility", value=True)

    run_btn = st.button("Run search", type="primary")

# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
st.title("Bayesian Single‑Arm Monitoring Study Designer")

with st.expander("📋 Workflow Guide (click to expand)", expanded=False):
    st.markdown("""
**Recommended workflow**
- **Classic Grid**: quick scan with small `n_sim`, calibrate fast, then re-run selected rows at high `n_sim`.
- **Staged Workflow**: let the app prune aggressively with fast α, check α exactly, check power, race the survivors, then re-evaluate finalists precisely.

**Tips**
- Use φ≈1.5–3.0 to make early looks stricter in per-look mode.
- If early futility is too aggressive, try ψ=0.01 (or 0 temporarily) to diagnose.
- Re-evaluate shortlisted designs with `n_sim ≥ 5000` before finalizing.
""")

st.markdown("---")

# ---- Helpers for plain-language tables ----
def _fmt_looks(looks):
    if looks is None or (isinstance(looks, float) and np.isnan(looks)):
        return "—"
    try:
        return ", ".join(str(int(n)) for n in looks)
    except Exception:
        return str(looks)

def _fmt_gamma_vec(gvec):
    if gvec is None or (isinstance(gvec, float) and np.isnan(gvec)):
        return "—"
    try:
        return ", ".join(f"{float(g):.3f}" for g in gvec)
    except Exception:
        return str(gvec)

def _fmt_r_by_look(r_by_look):
    """
    r_by_look is a list like [(n1, rmin1), (n2, rmin2), ...]
    Render as: n=10: r≥3; n=20: r≥6; ...
    """
    if r_by_look is None or (isinstance(r_by_look, float) and np.isnan(r_by_look)):
        return "—"
    try:
        parts = []
        for n, r in r_by_look:
            rtxt = "—" if r is None else str(int(r))
            parts.append(f"n={int(n)}: r≥{rtxt}")
        return "; ".join(parts)
    except Exception:
        return str(r_by_look)

def _make_plain_table(df_src: pd.DataFrame) -> pd.DataFrame:
    """Build a human-friendly table for display."""
    if df_src is None or df_src.empty:
        return pd.DataFrame()
    df = df_src.copy()
    # High-level design
    df["Max patients (N)"] = df["N"]
    df["Interim checks (K)"] = df["K_interims"]
    df["When we check (patients enrolled)"] = df["looks"].apply(_fmt_looks)

    # Success threshold(s): per-look vector if present, else single gamma
    def _gamma_text_row(row):
        gv = row.get("gamma_e_vector", None)
        if gv is not None:
            return _fmt_gamma_vec(gv)
        gu = row.get("gamma_e_used", None)
        if gu is None or (isinstance(gu, float) and np.isnan(gu)):
            return "—"
        return f"{float(gu):.3f}"
    df["Success threshold(s) γₑ"] = df.apply(_gamma_text_row, axis=1)

    # Integer boundaries
    df["Min responses needed at each check"] = df["r_success_by_look"].apply(_fmt_r_by_look)
    df["Final responses needed"] = df["r_star_final"].apply(
        lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else str(int(x))
    )

    # Operating characteristics (with uncertainty)
    if "alpha" in df:
        df["Type I error (α)"] = df["alpha"].map(lambda x: f"{float(x):.3f}")
    if "alpha_95ci" in df:
        df["α ±95% CI width"] = df["alpha_95ci"].map(
            lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"±{float(x):.3f}"
        )
    if "power" in df:
        df["Power"] = df["power"].map(lambda x: f"{float(x):.3f}")
    if "power_95ci" in df:
        df["Power ±95% CI width"] = df["power_95ci"].map(
            lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"±{float(x):.3f}"
        )

    # Workload / efficiency
    if "ESS_p1" in df:
        df["Avg patients if p = p₁"] = df["ESS_p1"].map(lambda x: f"{float(x):.1f}")
    if "ESS_p0" in df:
        df["Avg patients if p = p₀"] = df["ESS_p0"].map(lambda x: f"{float(x):.1f}")
    if "avg_looks_p1" in df:
        df["Avg checks if p = p₁"] = df["avg_looks_p1"].map(lambda x: f"{float(x):.2f}")

    # Feasibility tag
    if "is_feasible" in df:
        df["Meets α & power?"] = df["is_feasible"].map({True: "Yes", False: "No"})
    df["Tag(s)"] = df.get("selection", "").replace({np.nan: ""})
    # Optional safety shown if present
    if "safety_stop_q1" in df:
        df["P(stop for safety \n q₁)"] = df["safety_stop_q1"].map(
            lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.3f}"
        )
    if "safety_stop_qmax" in df:
        df["P(stop for safety \n q_max)"] = df["safety_stop_qmax"].map(
            lambda x: "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.3f}"
        )

    # Column order
    cols = [
        "Tag(s)","Meets α & power?","Max patients (N)","Interim checks (K)",
        "When we check (patients enrolled)",
        "Success threshold(s) γₑ","Min responses needed at each check","Final responses needed",
        "Type I error (α)","α ±95% CI width","Power","Power ±95% CI width",
        "Avg patients if p = p₁","Avg patients if p = p₀","Avg checks if p = p₁"
    ]
    if "P(stop for safety \n q₁)" in df: cols.append("P(stop for safety \n q₁)")
    if "P(stop for safety \n q_max)" in df: cols.append("P(stop for safety \n q_max)")
    return df[cols]

# ---- Helpers to re-evaluate a single design with fixed thresholds ----
def _reval_one_row(row: pd.Series, n_sim_big: int, seed_base: int) -> Dict:
    N = int(row["N"])
    looks = list(row["looks"])
    K = int(row["K_interims"])
    a_e = float(st.session_state["a_e"])
    b_e = float(st.session_state["b_e"])
    a_s = float(st.session_state.get("a_s") or 0.0)
    b_s = float(st.session_state.get("b_s") or 0.0)
    p0 = float(st.session_state["p0"])
    p1 = float(st.session_state["p1"])
    qmax = st.session_state.get("qmax")
    q1 = st.session_state.get("q1")
    psi_fut = float(st.session_state["psi_fut"])
    gamma_s = st.session_state.get("gamma_s")
    seed = int(seed_base + 9973*N + 37*K + n_sim_big)

    gamma_vec = row.get("gamma_e_vector", None)
    if gamma_vec is not None and isinstance(gamma_vec, (list, tuple)):
        oc = cached_evaluate_vector(
            N, K, tuple(looks),
            a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
            tuple(float(g) for g in gamma_vec),
            psi_fut, gamma_s,
            n_sim_big, seed, skip_futility=False
        )
    else:
        ge = float(row.get("gamma_e_used", st.session_state["gamma_e"]))
        oc = cached_evaluate_single(
            N, K, tuple(looks),
            a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
            ge, psi_fut, gamma_s,
            n_sim_big, seed, skip_futility=False
        )
    return {
        "N": N, "K_interims": K, "looks": looks,
        "gamma_e_used": row.get("gamma_e_used", None),
        "gamma_e_vector": row.get("gamma_e_vector", None),
        "alpha": oc["alpha"], "power": oc["power"],
        "ESS_p0": oc["ess_p0"], "ESS_p1": oc["ess_p1"], "avg_looks_p1": oc["avg_looks"]
    }

def _add_uncertainty_columns(df_in: pd.DataFrame, n_sim_val: int) -> pd.DataFrame:
    df = df_in.copy()
    df["alpha_se"] = df["alpha"].apply(lambda p: _se(float(p), n_sim_val))
    df["power_se"] = df["power"].apply(lambda p: _se(float(p), n_sim_val))
    df["alpha_95ci"] = df["alpha_se"] * 1.96
    df["power_95ci"] = df["power_se"] * 1.96
    return df

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if run_btn:
    mode_key = cal_mode.lower()
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
    )
    grid_ranges = dict(
        N_min=int(st.session_state["N_min"]), N_max=int(st.session_state["N_max"]),
        K_min=int(st.session_state["K_min"]), K_max=int(st.session_state["K_max"]),
    )
    seed = int(st.session_state["seed"])

    if search_mode.startswith("Classic"):
        params = dict(
            **grid_ranges,
            **params_common,
            gamma_e=float(st.session_state["gamma_e"]),
            n_sim=int(st.session_state["n_sim_v2"]),
            seed=seed,
            require_alpha_for_high_power=bool(require_alpha_for_high_power),
            cal_mode=mode_key,
            cal_n_sim=int(cal_n_sim),
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
        # Staged Workflow
        # Pull staged settings from sidebar
        z_early = float(z_early)
        z_strict = float(z_strict)
        stage1_nsim = int(stage1_nsim)
        stage2_alpha_nsim = int(stage2_alpha_nsim)
        stage3_power_nsim = int(stage3_power_nsim)
        schedule = _parse_schedule(schedule_str)
        cal_n_sim_s1 = int(cal_n_sim_s1)
        fast_mode_s1 = bool(fast_mode_s1)
        skip_fut_s1 = bool(skip_fut_s1)
        carry_limit_stage1 = int(carry_limit_stage1)
        carry_limit_stage2 = int(carry_limit_stage2)
        carry_limit_stage3 = int(carry_limit_stage3)

        st.write("Staged workflow: fast α → exact α → power → racing (successive halving).")
        st.write(f"Stage1 n_sim={stage1_nsim} (vectorized α, futility skipped); Stage2 α n_sim={stage2_alpha_nsim}; Stage3 power n_sim={stage3_power_nsim}.")
        st.write(f"Racing schedule={schedule} (batches @ keep_fraction).")

        start = time.time()
        with st.spinner("Running staged workflow..."):
            df = run_staged_workflow(
                **grid_ranges, **params_common,
                seed=seed,
                cal_mode=mode_key, cal_n_sim_s1=cal_n_sim_s1, phi=float(phi),
                fast_mode_s1=fast_mode_s1, skip_fut_s1=skip_fut_s1,
                z_early=z_early, z_strict=z_strict,
                stage1_nsim=stage1_nsim, stage2_alpha_nsim=stage2_alpha_nsim, stage3_power_nsim=stage3_power_nsim,
                racing_schedule=schedule,
                carry_limit_stage1=carry_limit_stage1,
                carry_limit_stage2=carry_limit_stage2,
                carry_limit_stage3=carry_limit_stage3
            )
        st.success(f"Staged search completed in {time.time() - start:.2f} sec.")

    # If nothing found
    if df is None or df.empty:
        st.warning("No designs survived the filters. Try relaxing gates or widening the grid.")
        st.stop()

    # Add SE / 95% CI if the evaluation had a uniform n_sim (classic), otherwise skip
    if search_mode.startswith("Classic"):
        n_used = int(st.session_state["n_sim_v2"])
        df["alpha_se"] = df["alpha"].apply(lambda p: _se(float(p), n_used))
        df["power_se"] = df["power"].apply(lambda p: _se(float(p), n_used))
        df["alpha_95ci"] = df["alpha_se"] * 1.96
        df["power_95ci"] = df["power_se"] * 1.96

    # Optional feasibility filter (toggle above)
    show_only_feasible = st.session_state.get("show_only_feasible", False)
    df_view = df[df["is_feasible"]] if show_only_feasible and "is_feasible" in df else df

    # -------------------- Summary of Designs (Plain Language) --------------------
    st.subheader("Summary of Designs")
    plain_summary = _make_plain_table(df_view)
    st.dataframe(plain_summary, use_container_width=True)

    with st.expander("What do these columns mean?"):
        st.markdown("""
- **Max patients (N)**: Maximum number of patients if the study runs to completion.
- **Interim checks (K)**: How many interim analyses before the final look.
- **When we check (patients enrolled)**: Patient counts (cumulative) at which we perform each look.
- **Success threshold(s) γₑ**: Posterior probability threshold(s) to declare success (one per look if calibrated per‑look).
- **Min responses needed at each check**: For each look `n`, the minimum number of responders `r` required to meet the success rule at that look.
- **Final responses needed**: Minimum total responders required at the final analysis to declare success.
- **Type I error (α)**: Chance of a false positive (declaring success when the true rate is **p₀**).
- **Power**: Chance of correctly declaring success when the true rate is **p₁**.
- **Avg patients if p = p₁ / p₀**: Expected sample size under **p₁** / **p₀**.
- **Avg checks if p = p₁**: Average number of looks used when the true rate is **p₁**.
- **P(stop for safety \n q₁ / q_max)** (if shown): Probability of stopping early for safety under the given toxicity rate.
""")

    # -------------------- Recommended Designs (Plain Language) --------------------
    st.subheader("Recommended Designs")
    def _show_recs(df_src: pd.DataFrame, tag: str, title: str):
        subset = df_src[df_src["selection"].str.contains(tag, na=False)]
        st.write(f"### {title}")
        if subset.empty:
            st.info("None.")
            return
        st.dataframe(
            _make_plain_table(subset)[[
                "Max patients (N)","Interim checks (K)","When we check (patients enrolled)",
                "Success threshold(s) γₑ","Min responses needed at each check","Final responses needed",
                "Type I error (α)","Power","Avg patients if p = p₁"
            ]],
            use_container_width=True
        )

    _show_recs(df, "staged_finalist", "Staged Workflow Finalists")
    _show_recs(df, "smallest_N", "Smallest N (meets α & power)")
    _show_recs(df, "high_power", "High Power (under N budget)")
    _show_recs(df, "sweet_spot", "Sweet Spot (min Avg patients if p = p₁)")

    # ------------------------------ Download + Chart -----------------------------
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "bayes_monitor_designs.csv")
    st.markdown("---")

    import altair as alt
    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x="N:Q",
            y="power:Q",
            color="is_feasible:N",
            tooltip=["N","K_interims","alpha","power","ESS_p1","gamma_e_used"]
        )
        .properties(width=900, height=340)
    )
    st.altair_chart(chart, use_container_width=True)

    # ---- Re-evaluate shortlisted designs (targeted, fixed thresholds) ----
    with st.expander("🔁 Re‑evaluate shortlisted designs (fixed thresholds)"):
        st.markdown(
            "Pick designs to re‑simulate with a larger `n_sim`. "
            "This holds the calibrated thresholds fixed so you can measure α/power precisely."
        )
        default_idx = list(df[df["selection"].str.contains("staged_finalist|smallest_N|high_power|sweet_spot", na=True)].index)
        if len(default_idx) == 0:
            default_idx = list(df.sort_values(["is_feasible","power","ESS_p1"], ascending=[False, False, True]).head(5).index)
        choose_rows = st.multiselect(
            "Choose designs (by row index)",
            options=list(df.index),
            default=default_idx,
            help="Select rows from the table above by their index."
        )
        n_sim_big = st.number_input("n_sim for re‑evaluation", min_value=1000, max_value=200000, value=5000, step=500)
        seed_big = st.number_input("Base seed for re‑evaluation", min_value=0, max_value=9999999, value=20260129, step=1)

        if st.button("Run re‑evaluation on selected"):
            if len(choose_rows) == 0:
                st.warning("Select at least one row.")
            else:
                with st.spinner("Re‑evaluating selected designs..."):
                    rows = []
                    for ridx in choose_rows:
                        rows.append(_reval_one_row(df.loc[ridx], int(n_sim_big), int(seed_big)))
                    re_df = pd.DataFrame(rows)
                    re_df = _add_uncertainty_columns(re_df, int(n_sim_big))
                    st.success("Re‑evaluation complete.")

                    # Show in plain language
                    re_view = re_df.copy()
                    re_view["When we check (patients enrolled)"] = re_view["looks"].apply(_fmt_looks)
                    re_view["Success threshold(s) γₑ"] = np.where(
                        re_view.get("gamma_e_vector").notna() if "gamma_e_vector" in re_view else False,
                        re_view.get("gamma_e_vector", None).apply(_fmt_gamma_vec),
                        re_view.get("gamma_e_used", None).apply(lambda g: "—" if g is None or (isinstance(g, float) and np.isnan(g)) else f"{float(g):.3f}")
                    )
                    re_view = re_view[[
                        "N","K_interims","When we check (patients enrolled)","Success threshold(s) γₑ",
                        "alpha","alpha_95ci","power","power_95ci","ESS_p1","ESS_p0","avg_looks_p1"
                    ]].rename(columns={
                        "N":"Max patients (N)",
                        "K_interims":"Interim checks (K)",
                        "ESS_p1":"Avg patients if p = p₁",
                        "ESS_p0":"Avg patients if p = p₀",
                        "alpha":"Type I error (α)",
                        "power":"Power"
                    })
                    st.dataframe(re_view, use_container_width=True)
                    st.download_button(
                        "Download re‑evaluation (CSV)",
                        re_df.to_csv(index=False).encode(),
                        file_name="reevaluated_designs.csv"
                    )

