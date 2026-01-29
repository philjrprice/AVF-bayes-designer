# =============================================================================
# AVF_Staged_Monitor_Designer.py  (standalone, high-performance staged workflow)
# Bayesian Single-Arm Monitoring Study Designer — Staged Workflow only
# - Stage 0: Deterministic pruning (dominance + implausible final boundary)
# - Stage 1: Fast α screen (vectorized, futility skipped) + small CI gate
# - Stage 2: Exact α (path-wise, futility ON) — incremental sims
# - Stage 3: Exact power (path-wise) — incremental sims
# - Stage 4: Racing (successive halving) — incremental sims w/ CRN
# - Stage 5: Final precise re-evaluation for finalists
#
# Performance features:
#   * Discrete buttons (no monolithic run)
#   * Incremental stats (we add sims in batches; no re-start)
#   * CRN across candidates per batch (variance reduction)
#   * Vectorized Stage 1
#   * Lazy imports for SciPy/NumPy/Altair (faster initial load)
#
# NOTE: This file intentionally omits the "Classic Grid" path.
# =============================================================================

# Lightweight imports at top-level
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# Lazy import helpers (heavy libs only loaded when used)
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
st.set_page_config(page_title="Bayesian Single-Arm Monitor (Staged)", layout="wide")

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULTS = {
    "p0": 0.20,
    "p1": 0.40,
    "a_e": 1.0,
    "b_e": 1.0,
    "gamma_e": 0.95,  # when calibration = Off or Single-γe mode
    "psi_fut": 0.05,  # predictive futility threshold
    "enable_safety": True,
    "qmax": 0.30,
    "q1": 0.15,
    "a_s": 1.0,
    "b_s": 1.0,
    "gamma_s": 0.90,
    # Ranges
    "N_min": 30,
    "N_max": 120,
    "K_min": 0,
    "K_max": 4,
    # Targets
    "alpha_target": 0.10,
    "power_target": 0.80,
    "N_budget": 80,
    # RNG
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
# Simulation batch (path-wise), with CRN via seed discipline
# -----------------------------------------------------------------------------
def simulate_batch(
    design: Design,
    n_batch: int,
    seed: int,
    skip_futility: bool,
    true_p: float,
    true_q: Optional[float]
) -> Tuple[int, int, int, int, int]:
    """
    Simulate n_batch trials (path-wise) and return summary counts:
    returns (n_success, sum_n_used, n_safety_stops, sum_looks, n_total)
    """
    np = lazy_numpy()
    bounds = compute_boundaries(design)
    rng = np.random.default_rng(seed)

    succ = 0
    sum_n_used = 0
    saf = 0
    sum_looks = 0
    for _ in range(n_batch):
        # One trial
        N = design.N
        responses = rng.binomial(1, true_p, N)
        tox = rng.binomial(1, true_q, N) if (true_q is not None) else np.zeros(N, dtype=int)
        looks_used = 0
        decided = False
        for n in design.look_schedule:
            r = int(np.sum(responses[:n]))
            t = int(np.sum(tox[:n]))
            looks_used += 1

            # safety
            t_saf = bounds[n]["t_safety_min"]
            if t_saf is not None and t >= t_saf:
                saf += 1
                sum_n_used += n
                sum_looks += looks_used
                decided = True
                break

            # efficacy
            r_min = bounds[n]["r_success_min"]
            if r_min is not None and r >= r_min:
                succ += 1
                sum_n_used += n
                sum_looks += looks_used
                decided = True
                break

            # futility
            if (not skip_futility) and (n < N):
                r_star_final = bounds[n]["r_star_final"]
                if r_star_final is not None:
                    ppos = beta_binomial_predictive_prob_at_least(
                        r, n, N, design.a_e, design.b_e, r_star_final
                    )
                    if ppos < design.psi_fut:
                        sum_n_used += n
                        sum_looks += looks_used
                        decided = True
                        break

            if n == N:
                r_star_final = bounds[n]["r_star_final"]
                if r_star_final is not None and r >= r_star_final:
                    succ += 1
                sum_n_used += n
                sum_looks += looks_used
                decided = True
                break

        if not decided:
            # Shouldn't happen; but safeguard
            sum_n_used += design.N
            sum_looks += looks_used

    return succ, sum_n_used, saf, sum_looks, n_batch

# -----------------------------------------------------------------------------
# Fast α-only (vectorized, futility skipped) for Stage 1
# -----------------------------------------------------------------------------
def vectorized_alpha_under_p0(
    design: Design, n_sim: int, seed: int
) -> float:
    np = lazy_numpy()
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
# Caching wrappers (boundaries only; large arrays stay local to stages)
# -----------------------------------------------------------------------------
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
# Calibration (fast + small exact refinement)
# -----------------------------------------------------------------------------
def perlook_gamma_vector_from_final(gamma_final: float, L: int, phi: float) -> List[float]:
    np = lazy_numpy()
    idx = np.arange(1, L + 1, dtype=float)
    s = (idx / L) ** max(phi, 0.5)
    return [float(1.0 - (1.0 - gamma_final) * sl) for sl in s]

def fast_alpha_only(design: Design, n_sim: int, seed: int, skip_futility_during_cal: bool=True) -> float:
    if skip_futility_during_cal:
        return vectorized_alpha_under_p0(design, n_sim=n_sim, seed=seed)
    # fall back exact path-wise if futility ON during calibration
    # small n_sim used for refinement only
    succ, _, _, _, n = simulate_batch(design, n_sim, seed, skip_futility=False, true_p=design.p0, true_q=design.q1)
    return succ / max(1, n)

def calibrate_single_gamma_fast(
    N: int, K: int, looks: List[int],
    a_e: float, b_e: float, a_s: float, b_s: float,
    p0: float, p1: float, qmax: Optional[float], q1: Optional[float],
    psi_fut: float, gamma_s: Optional[float],
    alpha_target: float, n_sim_cal: int, seed: int,
    g_low: float, g_high: float, tol_alpha: float,
    fast_mode: bool, skip_futility_during_cal: bool
) -> float:
    def alpha_at_gamma(gamma: float, fast: bool=True) -> float:
        d = Design(N, K, looks, a_e, b_e, a_s, b_s, p0, p1, qmax, q1, psi_fut, gamma_s, gamma_e=gamma)
        if fast:
            if not skip_futility_during_cal:
                return fast_alpha_only(d, n_sim_cal, seed + int(gamma*1e6)%1000000, skip_futility_during_cal=False)
            return fast_alpha_only(d, n_sim_cal, seed + int(gamma*1e6)%1000000, skip_futility_during_cal=True)
        # exact small refinement with futility ON
        succ, _, _, _, n = simulate_batch(d, n_sim_cal, seed + int(gamma*1e6)%1000000, False, true_p=p0, true_q=q1)
        return succ / max(1, n)

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

def calibrate_perlook_gamma_fast(
    N: int, K: int, looks: List[int],
    a_e: float, b_e: float, a_s: float, b_s: float,
    p0: float, p1: float, qmax: Optional[float], q1: Optional[float],
    psi_fut: float, gamma_s: Optional[float],
    alpha_target: float, n_sim_cal: int, seed: int,
    phi: float, g_low: float, g_high: float, tol_alpha: float,
    fast_mode: bool, skip_futility_during_cal: bool
) -> Tuple[float, Tuple[float, ...]]:
    L = len(looks)
    def alpha_at_gamma_final(gamma_final: float, fast: bool=True) -> float:
        gvec = perlook_gamma_vector_from_final(gamma_final, L, phi)
        d = Design(N, K, looks, a_e, b_e, a_s, b_s, p0, p1, qmax, q1, psi_fut, gamma_s,
                   gamma_e=None, gamma_e_vector=gvec)
        if fast:
            if not skip_futility_during_cal:
                return fast_alpha_only(d, n_sim_cal, seed + int(gamma_final*1e6)%1000000, skip_futility_during_cal=False)
            return fast_alpha_only(d, n_sim_cal, seed + int(gamma_final*1e6)%1000000, skip_futility_during_cal=True)
        succ, _, _, _, n = simulate_batch(d, n_sim_cal, seed + int(gamma_final*1e6)%1000000, False, true_p=p0, true_q=q1)
        return succ / max(1, n)

    lo, hi = g_low, g_high
    a_lo = alpha_at_gamma_final(lo, fast=fast_mode)
    a_hi = alpha_at_gamma_final(hi, fast=fast_mode)

    if a_lo <= alpha_target + tol_alpha:
        gamma_fast = float(lo)
    elif a_hi > alpha_target + tol_alpha:
        gamma_fast = float(hi)
    else:
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            a_mid = alpha_at_gamma_final(mid, fast=fast_mode)
            if a_mid <= alpha_target + tol_alpha: hi = mid
            else: lo = mid
            if abs(hi - lo) < 1e-3: break
        gamma_fast = float(hi)

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
# Deterministic pruning (Stage 0)
# -----------------------------------------------------------------------------
def _se(p: float, n: int) -> float:
    np = lazy_numpy()
    return float(np.sqrt(max(p * (1 - p), 0.0) / max(n, 1)))

def _dominates(r_by_look_A: List[Tuple[int,int]], r_by_look_B: List[Tuple[int,int]]) -> bool:
    ge = [ra >= rb for (_, ra), (_, rb) in zip(r_by_look_A, r_by_look_B)]
    gt = [ra >  rb for (_, ra), (_, rb) in zip(r_by_look_A, r_by_look_B)]
    return all(ge) and any(gt)

def _implausible_final_rstar(N: int, p1: float, r_star_final: Optional[int]) -> bool:
    if r_star_final is None:
        return False
    np = lazy_numpy()
    mu = N * p1
    var = N * p1 * (1 - p1)
    thr = mu + 3.0 * np.sqrt(max(var, 1e-12))
    return r_star_final > thr

def deterministic_prune(rows: List[dict], p1: float) -> List[dict]:
    # Implausible final r*
    keep = [r for r in rows if not _implausible_final_rstar(r["N"], p1, r["r_star_final"])]
    # Dominance within (N, looks)
    out = []
    by_key: Dict[Tuple[int, Tuple[int,...]], List[dict]] = {}
    for row in keep:
        key = (row["N"], tuple(row["looks"]))
        by_key.setdefault(key, []).append(row)
    for key, group in by_key.items():
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

# -----------------------------------------------------------------------------
# Successive halving (Stage 4)
# -----------------------------------------------------------------------------
@dataclass
class IncStats:
    # cumulative counts across batches
    n_sim_alpha: int = 0
    succ_alpha: int = 0
    sum_n_used_alpha: int = 0
    sum_looks_alpha: int = 0
    n_safety_alpha: int = 0

    n_sim_power: int = 0
    succ_power: int = 0
    sum_n_used_power: int = 0
    sum_looks_power: int = 0
    n_safety_power: int = 0

    def alpha_mean(self): return self.succ_alpha / max(1, self.n_sim_alpha)
    def power_mean(self): return self.succ_power / max(1, self.n_sim_power)
    def ess_p0_mean(self): return self.sum_n_used_alpha / max(1, self.n_sim_alpha)
    def ess_p1_mean(self): return self.sum_n_used_power / max(1, self.n_sim_power)

def design_key_from_row(row: dict) -> Tuple:
    return (
        int(row["N"]),
        tuple(int(x) for x in row["looks"]),
        tuple(row["gamma_e_vector"]) if row.get("gamma_e_vector") is not None else ("SINGLE", float(row["gamma_e_used"]))
    )

def successive_halving(
    rows_in: List[dict],
    stage_batches: List[Tuple[int,float]],  # [(n_batch, keep_frac), ...]
    seed_base: int,
    alpha_target: float,
    a_e: float, b_e: float, a_s: float, b_s: float,
    p0: float, p1: float, qmax: Optional[float], q1: Optional[float],
    psi_fut: float, gamma_s: Optional[float]
) -> List[dict]:
    # Persistent stats in session state keyed by design
    stats: Dict[Tuple, IncStats] = st.session_state.setdefault("stats_store", {})
    pool = list(rows_in)

    for round_idx, (n_batch, keep_frac) in enumerate(stage_batches):
        if len(pool) <= 1: break
        # Common Random Numbers: use same seed offset for all designs in this batch
        batch_seed = seed_base + 5000*(round_idx+1)

        for row in pool:
            key = design_key_from_row(row)
            stt = stats.setdefault(key, IncStats())
            N, K, looks = int(row["N"]), int(row["K_interims"]), list(row["looks"])
            design = Design(
                N, K, looks, a_e, b_e, a_s, b_s, p0, p1, qmax, q1, psi_fut, gamma_s,
                gamma_e=None if row.get("gamma_e_vector") is not None else float(row["gamma_e_used"]),
                gamma_e_vector=(list(row["gamma_e_vector"]) if row.get("gamma_e_vector") is not None else None)
            )
            # Alpha batch
            succ, sumN, saf, sumL, n = simulate_batch(design, n_batch, batch_seed, False, true_p=p0, true_q=q1)
            stt.n_sim_alpha += n; stt.succ_alpha += succ; stt.sum_n_used_alpha += sumN; stt.n_safety_alpha += saf; stt.sum_looks_alpha += sumL
            # Power batch
            succ, sumN, saf, sumL, n = simulate_batch(design, n_batch, batch_seed, False, true_p=p1, true_q=q1)
            stt.n_sim_power += n; stt.succ_power += succ; stt.sum_n_used_power += sumN; stt.n_safety_power += saf; stt.sum_looks_power += sumL

        # Rank: power high, ESS_p1 low, alpha below target (penalize overflow)
        scored = []
        for row in pool:
            key = design_key_from_row(row)
            stt = stats[key]
            alpha_over = max(0.0, stt.alpha_mean() - alpha_target)
            scored.append((row, stt.power_mean(), -stt.ess_p1_mean(), -alpha_over))
        scored.sort(key=lambda t: (t[1], t[2], t[3]), reverse=True)

        k = max(1, int(len(scored) * keep_frac))
        pool = [r for (r, *_ ) in scored[:k]]

    return pool

# -----------------------------------------------------------------------------
# UI — SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Inputs")
    if st.button("Reset UI & clear cache"):
        st.cache_data.clear()
        st.session_state.clear()
        st.success("Cache & UI state cleared.")
        st.stop()

    st.markdown("⚡ *Staged Workflow — fast & incremental*")
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
    st.subheader("Design space")
    st.number_input("Minimum N", 5, 5000, DEFAULTS["N_min"], 1, key="N_min")
    st.number_input("Maximum N", int(st.session_state["N_min"]), 5000, DEFAULTS["N_max"], 1, key="N_max")
    st.number_input("Min interim looks (K)", 0, 40, DEFAULTS["K_min"], 1, key="K_min")
    st.number_input("Max interim looks (K)", int(st.session_state["K_min"]), 40, DEFAULTS["K_max"], 1, key="K_max")

    st.divider()
    st.subheader("Targets")
    st.number_input("Max Type I error α target", 0.0, 0.5, DEFAULTS["alpha_target"], 0.01, key="alpha_target")
    st.number_input("Min Power target", 0.0, 1.0, DEFAULTS["power_target"], 0.01, key="power_target")
    st.number_input("N budget (optional)", 0, 5000, DEFAULTS["N_budget"], 1, key="N_budget")

    st.divider()
    st.subheader("Calibration mode (success thresholds)")
    cal_mode = st.selectbox(
        "Choose", ["Off (use γₑ below)", "Single γₑ (calibrated)", "Per-look γₑ vector"], index=2
    )
    st.number_input("γₑ (used only when mode = Off)", 0.5, 0.999, DEFAULTS["gamma_e"], 0.01, key="gamma_e")
    phi = st.number_input("Per-look stringency φ (>=1 stricter early)", 0.5, 10.0, 3.0, 0.5)
    cal_n_sim = st.number_input("Fast calibration n_sim per bisection step", 100, 50000, 500, 100)
    cal_skip_fut = st.checkbox("Skip futility during calibration (faster; conservative)", value=True)

    st.divider()
    st.subheader("Stage 1 (fast α screen)")
    s1_nsim = st.number_input("Stage 1 n_sim per design (vectorized α)", 50, 5000, 300, 50)
    s1_z = st.number_input("Early z for α gate (promote/prune/gray)", 0.0, 5.0, 1.64, 0.01)
    max_keep_s1 = st.number_input("Max designs to carry after S1", 1, 5000, 60, 1)

    st.subheader("Stage 2 (exact α, futility ON)")
    s2_nsim_batch = st.number_input("S2 batch size (incremental sims)", 100, 20000, 1000, 100)
    s2_z = st.number_input("Strict z for α gate", 0.0, 5.0, 1.96, 0.01)
    max_keep_s2 = st.number_input("Max designs to carry after S2", 1, 5000, 40, 1)

    st.subheader("Stage 3 (exact power)")
    s3_nsim_batch = st.number_input("S3 batch size (incremental sims)", 100, 20000, 1000, 100)
    s3_z = st.number_input("Strict z for power gate", 0.0, 5.0, 1.96, 0.01)
    max_keep_s3 = st.number_input("Max designs to carry before racing", 1, 5000, 20, 1)

    st.subheader("Stage 4 (racing: successive halving)")
    sched_str = st.text_input("Racing schedule '1000@0.5,2000@0.5,5000@1.0'", "1000@0.5,2000@0.5,5000@1.0")
    seed = st.number_input("Base RNG seed", 0, 9999999, DEFAULTS["seed"], 1)

    st.subheader("Stage 5 (final precise re-evaluation)")
    s5_nsim = st.number_input("Final n_sim for finalists", 1000, 200000, 5000, 500)

    st.divider()
    st.subheader("Controls")
    n_workers = st.number_input("Parallel workers (Stage 1 only)", 1, 64, 8, 1)
    show_only_feasible = st.checkbox("Show only designs meeting α & power (when available)", value=False)

    st.markdown("---")
    st.subheader("Run stages")
    btn_s0_s1 = st.button("▶ Stage 0 + Stage 1 (fast α screen)")
    btn_s2 = st.button("▶ Stage 2 (exact α, add sims)")
    btn_s3 = st.button("▶ Stage 3 (exact power, add sims)")
    btn_s4 = st.button("▶ Stage 4 (racing with schedule)")
    btn_s5 = st.button("▶ Stage 5 (final precise re-evaluation)")

# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
st.title("Bayesian Single‑Arm Monitoring — Staged Workflow")

with st.expander("📋 Workflow (how to use)", expanded=False):
    st.markdown("""
**Step-by-step**
1) **Stage 0+1**: Fast α screen (vectorized, futility skipped). We also run deterministic pruning first.  
2) **Stage 2**: For survivors, add **exact α** sims in a batch; gate again.  
3) **Stage 3**: Add **exact power** sims in a batch; gate again.  
4) **Stage 4**: **Racing** (successive halving) — concentrate sims on the best designs.  
5) **Stage 5**: Re-evaluate finalists with a **large n_sim** and export.

**Tips**
- Increase batch sizes for more stable gates; decrease to iterate faster.
- Racing uses **common random numbers** per batch across designs — fairer comparisons & faster convergence.
- You can re-click Stages 2–3 to accumulate additional sims before racing.
""")

# -----------------------------------------------------------------------------
# Helpers for plain-language display
# -----------------------------------------------------------------------------
def _fmt_looks(looks):
    if looks is None or (isinstance(looks, float) and pd.isna(looks)):
        return "—"
    try:
        return ", ".join(str(int(n)) for n in looks)
    except Exception:
        return str(looks)

def _fmt_gamma_vec(gvec):
    if gvec is None or (isinstance(gvec, float) and pd.isna(gvec)):
        return "—"
    try:
        return ", ".join(f"{float(g):.3f}" for g in gvec)
    except Exception:
        return str(gvec)

def _fmt_r_by_look(r_by_look):
    if r_by_look is None or (isinstance(r_by_look, float) and pd.isna(r_by_look)):
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
    if df_src is None or df_src.empty:
        return pd.DataFrame()
    df = df_src.copy()
    df["Max patients (N)"] = df["N"]
    df["Interim checks (K)"] = df["K_interims"]
    df["When we check (patients enrolled)"] = df["looks"].apply(_fmt_looks)
    def _gamma_text_row(row):
        gv = row.get("gamma_e_vector", None)
        if gv is not None:
            return _fmt_gamma_vec(gv)
        gu = row.get("gamma_e_used", None)
        if gu is None or (isinstance(gu, float) and pd.isna(gu)):
            return "—"
        return f"{float(gu):.3f}"
    df["Success threshold(s) γₑ"] = df.apply(_gamma_text_row, axis=1)
    df["Min responses needed at each check"] = df["r_success_by_look"].apply(_fmt_r_by_look)
    df["Final responses needed"] = df["r_star_final"].apply(lambda x: "—" if x is None or (isinstance(x, float) and pd.isna(x)) else str(int(x)))
    if "alpha" in df:
        df["Type I error (α)"] = df["alpha"].map(lambda x: f"{float(x):.3f}")
    if "power" in df:
        df["Power"] = df["power"].map(lambda x: f"{float(x):.3f}")
    if "ESS_p1" in df:
        df["Avg patients if p = p₁"] = df["ESS_p1"].map(lambda x: f"{float(x):.1f}")
    if "ESS_p0" in df:
        df["Avg patients if p = p₀"] = df["ESS_p0"].map(lambda x: f"{float(x):.1f}")
    return df[[
        "Max patients (N)","Interim checks (K)","When we check (patients enrolled)",
        "Success threshold(s) γₑ","Min responses needed at each check","Final responses needed",
        "Type I error (α)","Power","Avg patients if p = p₁","Avg patients if p = p₀"
    ]]

# -----------------------------------------------------------------------------
# Stage 0 + Stage 1
# -----------------------------------------------------------------------------
if btn_s0_s1:
    start = time.time()
    # Build initial grid
    rows = []
    for N in range(int(st.session_state["N_min"]), int(st.session_state["N_max"]) + 1):
        for K in range(int(st.session_state["K_min"]), int(st.session_state["K_max"]) + 1):
            looks = build_equal_looks(N, K)

            # Calibration choice
            if cal_mode.startswith("Off"):
                gamma_used = float(st.session_state["gamma_e"])
                gamma_vec = None
            elif cal_mode.startswith("Single"):
                gamma_used = calibrate_single_gamma_fast(
                    N, K, looks,
                    float(st.session_state["a_e"]), float(st.session_state["b_e"]),
                    float(st.session_state.get("a_s") or 0.0), float(st.session_state.get("b_s") or 0.0),
                    float(st.session_state["p0"]), float(st.session_state["p1"]),
                    st.session_state.get("qmax"), st.session_state.get("q1"),
                    float(st.session_state["psi_fut"]), st.session_state.get("gamma_s"),
                    float(st.session_state["alpha_target"]), int(cal_n_sim), int(st.session_state["seed"] + 17*N + 11*K),
                    g_low=0.50, g_high=0.999, tol_alpha=0.005,
                    fast_mode=True, skip_futility_during_cal=bool(cal_skip_fut)
                )
                gamma_vec = None
            else:
                gamma_final, gvec = calibrate_perlook_gamma_fast(
                    N, K, looks,
                    float(st.session_state["a_e"]), float(st.session_state["b_e"]),
                    float(st.session_state.get("a_s") or 0.0), float(st.session_state.get("b_s") or 0.0),
                    float(st.session_state["p0"]), float(st.session_state["p1"]),
                    st.session_state.get("qmax"), st.session_state.get("q1"),
                    float(st.session_state["psi_fut"]), st.session_state.get("gamma_s"),
                    float(st.session_state["alpha_target"]), int(cal_n_sim), int(st.session_state["seed"] + 23*N + 13*K),
                    phi=float(phi), g_low=0.50, g_high=0.999, tol_alpha=0.005,
                    fast_mode=True, skip_futility_during_cal=bool(cal_skip_fut)
                )
                gamma_used = gamma_final
                gamma_vec = gvec

            # Boundaries (for Stage 0 pruning info)
            if gamma_vec is not None:
                bounds = cached_boundaries_for_design(N, tuple(looks),
                    float(st.session_state["a_e"]), float(st.session_state["b_e"]), float(st.session_state["p0"]),
                    float(st.session_state.get("a_s") or 0.0), float(st.session_state.get("b_s") or 0.0),
                    st.session_state.get("qmax"), st.session_state.get("gamma_s"),
                    gamma_e=None, gamma_vec=tuple(gamma_vec))
            else:
                bounds = cached_boundaries_for_design(N, tuple(looks),
                    float(st.session_state["a_e"]), float(st.session_state["b_e"]), float(st.session_state["p0"]),
                    float(st.session_state.get("a_s") or 0.0), float(st.session_state.get("b_s") or 0.0),
                    st.session_state.get("qmax"), st.session_state.get("gamma_s"),
                    gamma_e=float(gamma_used), gamma_vec=None)

            r_by_look = [(n, bounds[n]["r_success_min"]) for n in looks]
            r_final = bounds[looks[-1]]["r_star_final"]

            rows.append(dict(
                N=N, K_interims=K, looks=looks,
                gamma_e_used=float(gamma_used),
                gamma_e_vector=(list(gamma_vec) if gamma_vec is not None else None),
                r_success_by_look=r_by_look, r_star_final=r_final
            ))

    # Stage 0 prune
    rows0 = deterministic_prune(rows, float(st.session_state["p1"]))
    # Budget filter (soft)
    if int(st.session_state["N_budget"]) > 0:
        rows0 = [r for r in rows0 if r["N"] <= max(int(st.session_state["N_budget"]), int(st.session_state["N_min"]))]

    # Stage 1 vectorized α under p0 (futility skipped)
    survivors = []
    from concurrent.futures import ThreadPoolExecutor
    seed_base = int(st.session_state["seed"]) + 10000
    def s1_eval(row):
        N, K, looks = int(row["N"]), int(row["K_interims"]), list(row["looks"])
        design = Design(
            N, K, looks,
            float(st.session_state["a_e"]), float(st.session_state["b_e"]),
            float(st.session_state.get("a_s") or 0.0), float(st.session_state.get("b_s") or 0.0),
            float(st.session_state["p0"]), float(st.session_state["p1"]),
            st.session_state.get("qmax"), st.session_state.get("q1"),
            float(st.session_state["psi_fut"]), st.session_state.get("gamma_s"),
            gamma_e=None if row["gamma_e_vector"] is not None else float(row["gamma_e_used"]),
            gamma_e_vector=(list(row["gamma_e_vector"]) if row["gamma_e_vector"] is not None else None)
        )
        a_hat = vectorized_alpha_under_p0(design, int(s1_nsim), seed=seed_base + N + 7*K)
        # Gate using Wald CI
        lo = a_hat - float(s1_z) * _se(a_hat, int(s1_nsim))
        hi = a_hat + float(s1_z) * _se(a_hat, int(s1_nsim))
        gate = "promote" if hi <= float(st.session_state["alpha_target"]) else ("prune" if lo > float(st.session_state["alpha_target"]) else "gray")
        return dict(**row, alpha_s1=a_hat, n_s1=int(s1_nsim), gate_s1=gate)

    with ThreadPoolExecutor(max_workers=int(n_workers)) as ex:
        results = list(ex.map(s1_eval, rows0))

    keep = [r for r in results if r["gate_s1"] in ("promote", "gray")]
    # Prefer low α_s1 then smaller N
    keep.sort(key=lambda r: (r["alpha_s1"], r["N"]))
    keep = keep[:int(max_keep_s1)]

    st.session_state["stage1_df"] = pd.DataFrame(keep)
    st.session_state["stage0_all"] = pd.DataFrame(rows)
    # fresh stats store
    st.session_state["stats_store"] = {}
    st.success(f"Stage 0+1 complete in {time.time()-start:.2f}s — {len(keep)} / {len(rows)} designs carried forward.")

# Show Stage 1 results
if "stage1_df" in st.session_state and not st.session_state["stage1_df"].empty:
    st.subheader("Stage 1 — survivors")
    df1 = st.session_state["stage1_df"].copy()
    view = df1.copy()
    view["When we check (patients enrolled)"] = view["looks"].apply(_fmt_looks)
    view["Success threshold(s) γₑ"] = view.apply(
        lambda r: _fmt_gamma_vec(r["gamma_e_vector"]) if r["gamma_e_vector"] is not None else f"{float(r['gamma_e_used']):.3f}",
        axis=1
    )
    view = view[["N","K_interims","When we check (patients enrolled)","Success threshold(s) γₑ","alpha_s1","n_s1"]]
    st.dataframe(view, use_container_width=True)

# -----------------------------------------------------------------------------
# Stage 2 — Exact α (incremental)
# -----------------------------------------------------------------------------
if btn_s2:
    if "stage1_df" not in st.session_state or st.session_state["stage1_df"].empty:
        st.warning("Run Stage 1 first.")
    else:
        start = time.time()
        stats = st.session_state.setdefault("stats_store", {})
        seed_base = int(st.session_state["seed"]) + 20000
        survivors = []
        for _, row in st.session_state["stage1_df"].iterrows():
            row = row.to_dict()
            key = design_key_from_row(row)
            stt = stats.setdefault(key, IncStats())
            N, K, looks = int(row["N"]), int(row["K_interims"]), list(row["looks"])
            design = Design(
                N, K, looks,
                float(st.session_state["a_e"]), float(st.session_state["b_e"]),
                float(st.session_state.get("a_s") or 0.0), float(st.session_state.get("b_s") or 0.0),
                float(st.session_state["p0"]), float(st.session_state["p1"]),
                st.session_state.get("qmax"), st.session_state.get("q1"),
                float(st.session_state["psi_fut"]), st.session_state.get("gamma_s"),
                gamma_e=None if row["gamma_e_vector"] is not None else float(row["gamma_e_used"]),
                gamma_e_vector=(list(row["gamma_e_vector"]) if row["gamma_e_vector"] is not None else None)
            )
            succ, sumN, saf, sumL, n = simulate_batch(design, int(s2_nsim_batch), seed_base + N + 7*K, False, true_p=float(st.session_state["p0"]), true_q=st.session_state.get("q1"))
            stt.n_sim_alpha += n; stt.succ_alpha += succ; stt.sum_n_used_alpha += sumN; stt.n_safety_alpha += saf; stt.sum_looks_alpha += sumL

            a_hat = stt.alpha_mean()
            lo = a_hat - float(s2_z) * _se(a_hat, stt.n_sim_alpha)
            hi = a_hat + float(s2_z) * _se(a_hat, stt.n_sim_alpha)
            gate = "promote" if hi <= float(st.session_state["alpha_target"]) else ("prune" if lo > float(st.session_state["alpha_target"]) else "gray")
            survivors.append(dict(**row, alpha_s2=a_hat, n_s2=stt.n_sim_alpha, gate_s2=gate))

        keep = [r for r in survivors if r["gate_s2"] in ("promote","gray")]
        keep.sort(key=lambda r: (r["alpha_s2"], r["N"]))
        keep = keep[:int(max_keep_s2)]
        st.session_state["stage2_df"] = pd.DataFrame(keep)
        st.success(f"Stage 2 complete in {time.time()-start:.2f}s — {len(keep)} carried forward.")

if "stage2_df" in st.session_state and not st.session_state["stage2_df"].empty:
    st.subheader("Stage 2 — survivors (α exact)")
    df2 = st.session_state["stage2_df"].copy()
    view = df2.copy()
    view["When we check (patients enrolled)"] = view["looks"].apply(_fmt_looks)
    view["Success threshold(s) γₑ"] = view.apply(
        lambda r: _fmt_gamma_vec(r["gamma_e_vector"]) if r["gamma_e_vector"] is not None else f"{float(r['gamma_e_used']):.3f}", axis=1
    )
    view = view[["N","K_interims","When we check (patients enrolled)","Success threshold(s) γₑ","alpha_s2","n_s2"]]
    st.dataframe(view, use_container_width=True)

# -----------------------------------------------------------------------------
# Stage 3 — Exact power (incremental)
# -----------------------------------------------------------------------------
if btn_s3:
    have = "stage2_df" if "stage2_df" in st.session_state and not st.session_state["stage2_df"].empty else ("stage1_df" if "stage1_df" in st.session_state else None)
    if not have:
        st.warning("Run Stage 1 (and Stage 2 if desired) first.")
    else:
        start = time.time()
        stats = st.session_state.setdefault("stats_store", {})
        seed_base = int(st.session_state["seed"]) + 30000
        source_df = st.session_state["stage2_df"] if "stage2_df" in st.session_state and not st.session_state["stage2_df"].empty else st.session_state["stage1_df"]
        survivors = []
        for _, row in source_df.iterrows():
            row = row.to_dict()
            key = design_key_from_row(row)
            stt = stats.setdefault(key, IncStats())
            N, K, looks = int(row["N"]), int(row["K_interims"]), list(row["looks"])
            design = Design(
                N, K, looks,
                float(st.session_state["a_e"]), float(st.session_state["b_e"]),
                float(st.session_state.get("a_s") or 0.0), float(st.session_state.get("b_s") or 0.0),
                float(st.session_state["p0"]), float(st.session_state["p1"]),
                st.session_state.get("qmax"), st.session_state.get("q1"),
                float(st.session_state["psi_fut"]), st.session_state.get("gamma_s"),
                gamma_e=None if row["gamma_e_vector"] is not None else float(row["gamma_e_used"]),
                gamma_e_vector=(list(row["gamma_e_vector"]) if row["gamma_e_vector"] is not None else None)
            )
            succ, sumN, saf, sumL, n = simulate_batch(design, int(s3_nsim_batch), seed_base + N + 7*K, False, true_p=float(st.session_state["p1"]), true_q=st.session_state.get("q1"))
            stt.n_sim_power += n; stt.succ_power += succ; stt.sum_n_used_power += sumN; stt.n_safety_power += saf; stt.sum_looks_power += sumL

            p_hat = stt.power_mean()
            lo = p_hat - float(s3_z) * _se(p_hat, stt.n_sim_power)
            hi = p_hat + float(s3_z) * _se(p_hat, stt.n_sim_power)
            gate = "promote" if lo >= float(st.session_state["power_target"]) else ("prune" if hi < float(st.session_state["power_target"]) else "gray")
            survivors.append(dict(**row, power_s3=p_hat, n_s3=stt.n_sim_power, gate_s3=gate))

        keep = [r for r in survivors if r["gate_s3"] in ("promote","gray")]
        keep.sort(key=lambda r: (-r["power_s3"], r["N"]))
        keep = keep[:int(max_keep_s3)]
        st.session_state["stage3_df"] = pd.DataFrame(keep)
        st.success(f"Stage 3 complete in {time.time()-start:.2f}s — {len(keep)} carried into racing.")

if "stage3_df" in st.session_state and not st.session_state["stage3_df"].empty:
    st.subheader("Stage 3 — survivors (power exact)")
    df3 = st.session_state["stage3_df"].copy()
    view = df3.copy()
    view["When we check (patients enrolled)"] = view["looks"].apply(_fmt_looks)
    view["Success threshold(s) γₑ"] = view.apply(
        lambda r: _fmt_gamma_vec(r["gamma_e_vector"]) if r["gamma_e_vector"] is not None else f"{float(r['gamma_e_used']):.3f}", axis=1
    )
    view = view[["N","K_interims","When we check (patients enrolled)","Success threshold(s) γₑ","power_s3","n_s3"]]
    st.dataframe(view, use_container_width=True)

# -----------------------------------------------------------------------------
# Stage 4 — Racing (successive halving)
# -----------------------------------------------------------------------------
def _parse_schedule(schedule_str: str) -> List[Tuple[int, float]]:
    out = []
    if not schedule_str:
        return out
    parts = [s.strip() for s in schedule_str.split(",") if s.strip()]
    for p in parts:
        if "@ " in p: p = p.replace("@ ", "@")
        n, k = p.split("@")
        out.append((int(float(n)), float(k)))
    return out

if btn_s4:
    have = "stage3_df" if "stage3_df" in st.session_state and not st.session_state["stage3_df"].empty else \
           ("stage2_df" if "stage2_df" in st.session_state and not st.session_state["stage2_df"].empty else \
            ("stage1_df" if "stage1_df" in st.session_state and not st.session_state["stage1_df"].empty else None))
    if not have:
        st.warning("Run Stage 1 (and optionally 2 & 3) first.")
    else:
        start = time.time()
        schedule = _parse_schedule(sched_str)
        df_src = st.session_state["stage3_df"] if "stage3_df" in st.session_state and not st.session_state["stage3_df"].empty \
                 else (st.session_state["stage2_df"] if "stage2_df" in st.session_state and not st.session_state["stage2_df"].empty \
                 else st.session_state["stage1_df"])
        rows_in = [r._asdict() if hasattr(r, "_asdict") else r for _, r in df_src.iterrows()]
        finalists = successive_halving(
            rows_in, schedule, int(st.session_state["seed"]) + 40000,
            float(st.session_state["alpha_target"]),
            float(st.session_state["a_e"]), float(st.session_state["b_e"]),
            float(st.session_state.get("a_s") or 0.0), float(st.session_state.get("b_s") or 0.0),
            float(st.session_state["p0"]), float(st.session_state["p1"]),
            st.session_state.get("qmax"), st.session_state.get("q1"),
            float(st.session_state["psi_fut"]), st.session_state.get("gamma_s")
        )
        st.session_state["stage4_df"] = pd.DataFrame(finalists)
        st.success(f"Stage 4 racing completed in {time.time()-start:.2f}s — finalists ready.")

if "stage4_df" in st.session_state and not st.session_state["stage4_df"].empty:
    st.subheader("Stage 4 — finalists (post-racing)")
    df4 = st.session_state["stage4_df"].copy()
    # attach current incremental stats
    stats = st.session_state.get("stats_store", {})
    def attach_stats(row):
        key = design_key_from_row(row)
        stt: IncStats = stats.get(key, IncStats())
        return pd.Series({
            "alpha_mean": stt.alpha_mean(),
            "power_mean": stt.power_mean(),
            "ESS_p0": stt.ess_p0_mean(),
            "ESS_p1": stt.ess_p1_mean(),
            "n_alpha": stt.n_sim_alpha,
            "n_power": stt.n_sim_power,
        })
    df4_stats = st.session_state["stage4_df"].apply(attach_stats, axis=1)
    df4 = pd.concat([df4, df4_stats], axis=1)

    view = df4.copy()
    view["When we check (patients enrolled)"] = view["looks"].apply(_fmt_looks)
    view["Success threshold(s) γₑ"] = view.apply(
        lambda r: _fmt_gamma_vec(r["gamma_e_vector"]) if r["gamma_e_vector"] is not None else f"{float(r['gamma_e_used']):.3f}", axis=1
    )
    view = view[["N","K_interims","When we check (patients enrolled)","Success threshold(s) γₑ",
                 "alpha_mean","power_mean","ESS_p1","ESS_p0","n_alpha","n_power"]]
    st.dataframe(view, use_container_width=True)

# -----------------------------------------------------------------------------
# Stage 5 — Final precise re-evaluation
# -----------------------------------------------------------------------------
if btn_s5:
    have = "stage4_df" if "stage4_df" in st.session_state and not st.session_state["stage4_df"].empty else \
           ("stage3_df" if "stage3_df" in st.session_state and not st.session_state["stage3_df"].empty else None)
    if not have:
        st.warning("Run racing (Stage 4) first, or at least Stage 3.")
    else:
        start = time.time()
        seed_base = int(st.session_state["seed"]) + 50000
        rows = []
        df_src = st.session_state["stage4_df"] if "stage4_df" in st.session_state and not st.session_state["stage4_df"].empty else st.session_state["stage3_df"]
        for i, row in df_src.iterrows():
            row = row.to_dict()
            N, K, looks = int(row["N"]), int(row["K_interims"]), list(row["looks"])
            design = Design(
                N, K, looks,
                float(st.session_state["a_e"]), float(st.session_state["b_e"]),
                float(st.session_state.get("a_s") or 0.0), float(st.session_state.get("b_s") or 0.0),
                float(st.session_state["p0"]), float(st.session_state["p1"]),
                st.session_state.get("qmax"), st.session_state.get("q1"),
                float(st.session_state["psi_fut"]), st.session_state.get("gamma_s"),
                gamma_e=None if row.get("gamma_e_vector") is not None else float(row["gamma_e_used"]),
                gamma_e_vector=(list(row["gamma_e_vector"]) if row.get("gamma_e_vector") is not None else None)
            )
            # precise α
            succ_a, sumN_a, saf_a, sumL_a, n_a = simulate_batch(design, int(s5_nsim), seed_base + N + 7*K, False, true_p=float(st.session_state["p0"]), true_q=st.session_state.get("q1"))
            # precise power
            succ_p, sumN_p, saf_p, sumL_p, n_p = simulate_batch(design, int(s5_nsim), seed_base + 2*(N + 7*K), False, true_p=float(st.session_state["p1"]), true_q=st.session_state.get("q1"))

            rows.append(dict(
                N=N, K_interims=K, looks=looks,
                gamma_e_used=design.gamma_e, gamma_e_vector=design.gamma_e_vector,
                r_success_by_look=[(n, compute_boundaries(design)[n]["r_success_min"]) for n in looks],
                r_star_final=compute_boundaries(design)[looks[-1]]["r_star_final"],
                alpha=succ_a / max(1, n_a), power=succ_p / max(1, n_p),
                ESS_p0=sumN_a / max(1, n_a), ESS_p1=sumN_p / max(1, n_p)
            ))

        df5 = pd.DataFrame(rows)
        df5["is_feasible"] = (df5["alpha"] <= float(st.session_state["alpha_target"])) & (df5["power"] >= float(st.session_state["power_target"]))
        st.session_state["stage5_df"] = df5
        st.success(f"Stage 5 (final) completed in {time.time()-start:.2f}s.")

if "stage5_df" in st.session_state and not st.session_state["stage5_df"].empty:
    st.subheader("Final re-evaluation (precise)")
    df5 = st.session_state["stage5_df"].copy()
    view = _make_plain_table(df5)
    st.dataframe(view, use_container_width=True)
    st.download_button("Download CSV", df5.to_csv(index=False).encode(), "finalists_precise.csv")

# -----------------------------------------------------------------------------
# Optional plot (Altair lazy import)
# -----------------------------------------------------------------------------
if "stage5_df" in st.session_state and not st.session_state["stage5_df"].empty:
    alt = lazy_altair()
    chart = (
        alt.Chart(st.session_state["stage5_df"])
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
