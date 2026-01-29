# =============================================================================
# AVF_Bayes_Monitor_Designer.py
# Streamlit app with per-look posterior success thresholds (alpha-spending style),
# calibration to Type I error target, and faster vectorized calibration.
# =============================================================================

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

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
    "gamma_e": 0.95,          # used when calibration = Off or Single-γe mode
    "psi_fut": 0.05,          # predictive futility threshold
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
    "n_sim": 1,               # quick-scan by default
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
    gamma_e: Optional[float] = None                  # single threshold (legacy)
    gamma_e_vector: Optional[List[float]] = None     # per-look thresholds


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
    """Predictive probability that future successes bring total ≥ r_star_final."""
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
    Fast approximate α estimation under p0 (and q1 for safety).
    Uses vectorized arrays; futility can be skipped for speed (conservative for calibration).
    """
    rng = np.random.default_rng(seed)
    N = design.N
    looks = design.look_schedule
    L = len(looks)

    # Precompute integer boundaries with current thresholds
    bounds = compute_boundaries(design)
    r_mins = np.array([bounds[n]["r_success_min"] if bounds[n]["r_success_min"] is not None else N+1 for n in looks], dtype=int)
    t_mins = np.array([bounds[n]["t_safety_min"] if bounds[n]["t_safety_min"] is not None else N+1 for n in looks], dtype=int)
    r_star_final = bounds[looks[-1]]["r_star_final"]

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
            active[stop_safety] = False  # stop, but not success

        # Efficacy success
        got_success = active & (r >= r_mins[i])
        success[got_success] = True
        active[got_success] = False

        # Futility (optional) — skipped by default during calibration for speed
        if (not skip_futility) and (n < N) and (r_star_final is not None):
            # Predictive prob is expensive; leave off in calibration.
            pass

        # Final look — those still active are evaluated at final implicitly by r_mins[-1]
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
# Calibration helpers
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
    """Bisection on a single posterior threshold gamma_e to achieve alpha <= target."""
    def alpha_at_gamma(gamma: float) -> float:
        if fast_mode:
            # use vectorized alpha eval
            d = Design(N, K, list(look_tuple), a_e, b_e, a_s, b_s, p0, p1, qmax, q1, psi_fut, gamma_s, gamma_e=gamma)
            return vectorized_alpha_under_p0(d, n_sim=n_sim_cal, seed=seed + int(gamma * 1e6) % 1000000,
                                             skip_futility=skip_futility_during_cal)
        oc = cached_evaluate_single(N, K, look_tuple, a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                                    gamma, psi_fut, gamma_s, n_sim_cal, seed + int(gamma*1e6)%1000000,
                                    skip_futility_during_cal)
        return float(oc["alpha"])

    lo, hi = g_low, g_high
    # Quick check at hi
    if alpha_at_gamma(hi) <= alpha_target + tol_alpha:
        return hi

    gamma_star = hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        a_mid = alpha_at_gamma(mid)
        if a_mid <= alpha_target + tol_alpha:
            gamma_star = mid
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < 1e-3:
            break
    return float(gamma_star)

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
    Calibrate per-look thresholds by bisection on gamma_final, gamma_vec derived via phi.
    Returns (gamma_final_star, gamma_vec_star).
    """
    L = len(look_tuple)

    def alpha_at_gamma_final(gamma_final: float) -> float:
        gamma_vec = tuple(perlook_gamma_vector_from_final(gamma_final, L, phi))
        if fast_mode:
            d = Design(N, K, list(look_tuple), a_e, b_e, a_s, b_s, p0, p1, qmax, q1, psi_fut, gamma_s,
                       gamma_e=None, gamma_e_vector=list(gamma_vec))
            return vectorized_alpha_under_p0(d, n_sim=n_sim_cal, seed=seed + int(gamma_final*1e6)%1000000,
                                             skip_futility=skip_futility_during_cal)
        oc = cached_evaluate_vector(N, K, look_tuple, a_e, b_e, a_s, b_s, p0, p1, qmax, q1,
                                    gamma_vec, psi_fut, gamma_s, n_sim_cal,
                                    seed + int(gamma_final*1e6)%1000000, skip_futility_during_cal)
        return float(oc["alpha"])

    lo, hi = g_low, g_high
    if alpha_at_gamma_final(hi) <= alpha_target + tol_alpha:
        return hi, tuple(perlook_gamma_vector_from_final(hi, L, phi))

    gamma_star = hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        a_mid = alpha_at_gamma_final(mid)
        if a_mid <= alpha_target + tol_alpha:
            gamma_star = mid
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < 1e-3:
            break

    return float(gamma_star), tuple(perlook_gamma_vector_from_final(float(gamma_star), L, phi))

# -----------------------------------------------------------------------------
# Grid search (supports Off / Single-γe / Per-look-γe modes)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def cached_grid_search(
    N_min, N_max, K_min, K_max,
    p0, p1, a_e, b_e, a_s, b_s, qmax, q1,
    gamma_e, psi_fut, gamma_s,
    n_sim, seed,
    alpha_target, power_target, N_budget,
    require_alpha_for_high_power,
    cal_mode: str,            # "off" | "single" | "per_look"
    cal_n_sim: int,
    phi: float,               # used only in per_look mode
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

            # Evaluate with chosen thresholds (using futility as configured)
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
        df.loc[smallest.index, "selection"] += "|smallest_N"
        df.loc[sweet.index, "selection"] += "|sweet_spot"

    if N_budget is not None and N_budget > 0:
        under = df[df["N"] <= N_budget]
        if require_alpha_for_high_power:
            under = under[under["meets_alpha"]]
        if not under.empty:
            best = under.sort_values(["power", "ESS_p1"], ascending=[False, True]).head(3)
            df.loc[best.index, "selection"] += "|high_power"

    return df

# -----------------------------------------------------------------------------
# UI — SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Inputs")

    if st.button("Reset UI & clear cache"):
        st.cache_data.clear()
        st.session_state.clear()
        st.success("Cache & UI state cleared. Rerun your grid search.")
        st.stop()

    st.markdown("⚡ *Quick-Scan Mode Enabled (n_sim defaults to 1)*")

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
    st.subheader("Grid search")
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
        "Require α constraint for 'High power' recommendations", value=True
    )

    # Evaluation sims (after calibration)
    st.number_input(
        "Monte Carlo replicates (n_sim)",
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
    cal_n_sim = st.number_input("Calibration n_sim (per bisection step)", 100, 200000, 1000, 100)
    fast_mode = st.checkbox("Use vectorized fast calibration (recommended)", value=True)
    skip_fut_cal = st.checkbox("Skip futility during calibration (faster; conservative)", value=True)

    phi = st.number_input("Per-look stringency φ (>=1: early stricter)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)

    show_only_feasible = st.checkbox("Show only designs meeting α & power targets", value=False)

    run_btn = st.button("Run grid search", type="primary")

# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
st.title("Bayesian Single‑Arm Monitoring Study Designer")
st.info(
    "Use **Per-look γₑ vector** to make early interims stricter and keep overall α under control. "
    "Calibrate with a modest n_sim (e.g., 500–2000), then re-evaluate shortlisted designs with larger n_sim."
)
st.markdown("---")

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if run_btn:
    mode_key = cal_mode.lower()
    mode_key = "off" if "off" in mode_key else ("single" if "single" in mode_key else "per_look")

    params = dict(
        N_min=int(st.session_state["N_min"]),
        N_max=int(st.session_state["N_max"]),
        K_min=int(st.session_state["K_min"]),
        K_max=int(st.session_state["K_max"]),
        p0=float(st.session_state["p0"]),
        p1=float(st.session_state["p1"]),
        a_e=float(st.session_state["a_e"]),
        b_e=float(st.session_state["b_e"]),
        a_s=float(st.session_state.get("a_s") or 0.0),
        b_s=float(st.session_state.get("b_s") or 0.0),
        qmax=st.session_state.get("qmax"),
        q1=st.session_state.get("q1"),
        gamma_e=float(st.session_state["gamma_e"]),
        psi_fut=float(st.session_state["psi_fut"]),
        gamma_s=st.session_state.get("gamma_s"),
        n_sim=int(st.session_state["n_sim_v2"]),
        seed=int(st.session_state["seed"]),
        alpha_target=float(st.session_state["alpha_target"]),
        power_target=float(st.session_state["power_target"]),
        N_budget=int(st.session_state["N_budget"]),
        require_alpha_for_high_power=bool(require_alpha_for_high_power),
        cal_mode=mode_key,
        cal_n_sim=int(cal_n_sim),
        phi=float(phi),
        fast_mode=bool(fast_mode),
        skip_fut_cal=bool(skip_fut_cal),
    )

    total_designs = (params["N_max"] - params["N_min"] + 1) * (params["K_max"] - params["K_min"] + 1)
    st.write(f"Evaluating ~{total_designs} designs × {params['n_sim']} sims each (cached).")
    if mode_key != "off":
        st.write(f"Calibration uses n_sim={params['cal_n_sim']} per bisection step (fast mode={params['fast_mode']}, skip futility={params['skip_fut_cal']}).")
    st.caption(f"Evaluation n_sim = {params['n_sim']}")

    start = time.time()
    with st.spinner("Running grid search..."):
        df = cached_grid_search(**params)
    st.success(f"Grid search completed in {time.time() - start:.2f} sec.")

    # SE / 95% CI
    def binom_se(p: float, n: int) -> float:
        return float(np.sqrt(max(p * (1 - p), 0.0) / max(n, 1)))

    n_used = int(st.session_state["n_sim_v2"])
    df["alpha_se"] = df["alpha"].apply(lambda p: binom_se(float(p), n_used))
    df["power_se"] = df["power"].apply(lambda p: binom_se(float(p), n_used))
    df["alpha_95ci"] = df["alpha_se"] * 1.96
    df["power_95ci"] = df["power_se"] * 1.96

    # Optional feasibility filter
    df_view = df[df["is_feasible"]] if show_only_feasible else df

    # Display
    if df[df["is_feasible"]].empty:
        st.warning("No feasible designs met α & power targets with the current inputs.")

    st.subheader("Summary of Designs")
    st.dataframe(
        df_view[
            [
                "selection","is_feasible","N","K_interims","looks",
                "gamma_e_used","gamma_e_vector","r_star_final","r_success_by_look",
                "alpha","alpha_se","alpha_95ci",
                "power","power_se","power_95ci",
                "ESS_p0","ESS_p1","avg_looks_p1",
                "safety_stop_q1","safety_stop_qmax"
            ]
        ],
        use_container_width=True
    )

    st.subheader("Recommended Designs")
    for label, tag in [
        ("Smallest N", "smallest_N"),
        ("High Power (under N budget)", "high_power"),
        ("Sweet Spot (min ESS_p1)", "sweet_spot"),
    ]:
        subset = df[df["selection"].str.contains(tag, na=False)]
        st.write(f"### {label}")
        if subset.empty:
            st.info("None.")
        else:
            st.dataframe(
                subset[
                    [
                        "N","K_interims","looks",
                        "gamma_e_used","gamma_e_vector","r_star_final","r_success_by_look",
                        "alpha","alpha_se","alpha_95ci",
                        "power","power_se","power_95ci","ESS_p1"
                    ]
                ],
                use_container_width=True
            )

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
