# bayes_single_arm_app.py
import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import comb, beta as beta_fn
from scipy.stats import beta

# =============================================================================
# Streamlit page config & defaults
# =============================================================================
st.set_page_config(page_title="Bayesian Single-Arm Monitor Designer", layout="wide")

DEFAULTS = {
    "p0": 0.20,
    "p1": 0.40,
    "a_e": 1.0,
    "b_e": 1.0,
    "gamma_e": 0.95,
    "psi_fut": 0.05,
    "enable_safety": True,
    "qmax": 0.30,
    "q1": 0.15,
    "a_s": 1.0,
    "b_s": 1.0,
    "gamma_s": 0.90,
    "N_min": 30,
    "N_max": 120,
    "K_min": 1,
    "K_max": 4,
    "alpha_target": 0.10,
    "power_target": 0.80,
    "N_budget": 80,
    "n_sim": 5000,          # smaller default for faster first run
    "seed": 12345,
}

# =============================================================================
# Core Bayesian utilities
# =============================================================================
def posterior_prob_p_greater_than(p0: float, a: float, b: float) -> float:
    """Posterior Pr(p > p0) when posterior ~ Beta(a, b)."""
    return 1.0 - beta.cdf(p0, a, b)

def posterior_prob_q_exceeds(qmax: float, a: float, b: float) -> float:
    """Posterior Pr(q > qmax) when posterior ~ Beta(a, b)."""
    return 1.0 - beta.cdf(qmax, a, b)

def minimal_successes_for_posterior_success(n: int, p0: float, a0: float, b0: float, gamma_e: float) -> Optional[int]:
    """Smallest r such that Pr(p > p0 | r, n) ≥ gamma_e. None if impossible."""
    for r in range(0, n + 1):
        a_post = a0 + r
        b_post = b0 + n - r
        if posterior_prob_p_greater_than(p0, a_post, b_post) >= gamma_e:
            return r
    return None

def safety_stop_threshold(n: int, qmax: float, a_s: float, b_s: float, gamma_s: float) -> Optional[int]:
    """Smallest t such that Pr(q > qmax | t, n) ≥ gamma_s. None if impossible."""
    for t in range(0, n + 1):
        a_post = a_s + t
        b_post = b_s + n - t
        if posterior_prob_q_exceeds(qmax, a_post, b_post) >= gamma_s:
            return t
    return None

def beta_binomial_predictive_prob_at_least(current_r: int, current_n: int, final_N: int,
                                           a0: float, b0: float, r_star_final: int) -> float:
    """
    Predictive probability (over future responses) that final total successes ≥ r_star_final.
    Using Beta-Binomial predictive distribution.
    """
    m = final_N - current_n
    a_post = a0 + current_r
    b_post = b0 + current_n - current_r

    if m < 0:
        raise ValueError("final_N must be >= current_n")
    if m == 0:
        return 1.0 if current_r >= r_star_final else 0.0

    j_min = max(0, r_star_final - current_r)
    if j_min <= 0:
        return 1.0

    total = 0.0
    for j in range(j_min, m + 1):
        total += comb(m, j) * beta_fn(j + a_post, m - j + b_post) / beta_fn(a_post, b_post)
    return float(total)

# =============================================================================
# Design structures
# =============================================================================
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
    gamma_e: float
    psi_fut: float
    gamma_s: Optional[float]

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

# =============================================================================
# Boundary computation
# =============================================================================
def compute_boundaries(design: Design) -> Dict[int, Dict[str, Optional[int]]]:
    """Compute efficacy/safety boundaries for each look."""
    bounds = {}
    r_star_final = minimal_successes_for_posterior_success(
        design.N, design.p0, design.a_e, design.b_e, design.gamma_e
    )
    for n in design.look_schedule:
        r_success_min = minimal_successes_for_posterior_success(
            n, design.p0, design.a_e, design.b_e, design.gamma_e
        )
        t_safety_min = None
        if design.qmax is not None and design.gamma_s is not None:
            t_safety_min = safety_stop_threshold(
                n, design.qmax, design.a_s, design.b_s, design.gamma_s
            )
        bounds[n] = {"r_success_min": r_success_min, "t_safety_min": t_safety_min, "r_star_final": r_star_final}
    return bounds

# =============================================================================
# Simulation engine
# =============================================================================
def simulate_one_trial(design: Design, true_p: float, true_q: Optional[float],
                       bounds: Dict[int, Dict[str, Optional[int]]],
                       rng: np.random.Generator) -> Tuple[bool, int, bool, int]:
    """
    Simulate one trial path; returns (efficacy_success, n_used, safety_stopped, looks_used).
    """
    N = design.N
    responses = rng.binomial(1, true_p, size=N)
    tox = rng.binomial(1, true_q, size=N) if (true_q is not None) else np.zeros(N, dtype=int)

    looks_used = 0
    for n in design.look_schedule:
        r = int(np.sum(responses[:n]))
        t = int(np.sum(tox[:n]))
        looks_used += 1

        # Safety first (if enabled)
        if design.qmax is not None and design.gamma_s is not None:
            t_safety_min = bounds[n]["t_safety_min"]
            if t_safety_min is not None and t >= t_safety_min:
                return (False, n, True, looks_used)

        # Efficacy success
        r_success_min = bounds[n]["r_success_min"]
        if r_success_min is not None and r >= r_success_min:
            return (True, n, False, looks_used)

        # Predictive futility (not at final)
        r_star_final = bounds[n]["r_star_final"]
        if n < N and r_star_final is not None:
            ppos = beta_binomial_predictive_prob_at_least(
                current_r=r, current_n=n, final_N=N,
                a0=design.a_e, b0=design.b_e, r_star_final=r_star_final
            )
            if ppos < design.psi_fut:
                return (False, n, False, looks_used)

        # Final look: declare based on final boundary
        if n == N:
            if r_star_final is not None and r >= r_star_final:
                return (True, n, False, looks_used)
            return (False, n, False, looks_used)

    return (False, N, False, looks_used)

def evaluate_design(design: Design, n_sim: int = 5000, seed: int = 12345) -> OperatingCharacteristics:
    """Evaluate operating characteristics via Monte Carlo simulation."""
    rng = np.random.default_rng(seed)
    bounds = compute_boundaries(design)

    # Under p0
    results_p0 = [simulate_one_trial(design, design.p0, design.q1, bounds, rng) for _ in range(n_sim)]
    alpha = float(np.mean([s for (s, _, _, _) in results_p0]))
    ess_p0 = float(np.mean([n_used for (_, n_used, _, _) in results_p0]))

    # Under p1
    results_p1 = [simulate_one_trial(design, design.p1, design.q1, bounds, rng) for _ in range(n_sim)]
    power = float(np.mean([s for (s, _, _, _) in results_p1]))
    ess_p1 = float(np.mean([n_used for (_, n_used, _, _) in results_p1]))
    avg_looks = float(np.mean([looks for (_, _, _, looks) in results_p1]))

    # Safety metrics (if enabled)
    safety_stop_prob_q1 = None
    safety_stop_prob_qmax = None
    if design.qmax is not None and design.gamma_s is not None and design.q1 is not None:
        results_q1 = [simulate_one_trial(design, design.p1, design.q1, bounds, rng) for _ in range(n_sim)]
        safety_stop_prob_q1 = float(np.mean([sf for (_, _, sf, _) in results_q1]))
        results_qmax = [simulate_one_trial(design, design.p1, design.qmax, bounds, rng) for _ in range(n_sim)]
        safety_stop_prob_qmax = float(np.mean([sf for (_, _, sf, _) in results_qmax]))

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

# =============================================================================
# Helpers
# =============================================================================
def build_equal_looks(N: int, K_interims: int) -> List[int]:
    """Build K_interims interim looks + final at N, spaced roughly equally."""
    if K_interims < 0:
        K_interims = 0
    looks = []
    for k in range(1, K_interims + 1):
        looks.append(int(round((k / (K_interims + 1)) * N)))
    if looks and looks[-1] == N:
        looks[-1] = max(1, N - 1)
    if len(looks) == 0 or looks[-1] < N:
        looks.append(N)
    looks = sorted(set([max(1, x) for x in looks]))
    if looks[-1] != N:
        looks.append(N)
    return looks

def final_success_boundary_info(N: int, p0: float, a_e: float, b_e: float, gamma_e: float) -> Dict[str, Optional[int]]:
    """Report minimal successes required at final N to meet posterior success (if it exists)."""
    r_star = minimal_successes_for_posterior_success(N, p0, a_e, b_e, gamma_e)
    return {"N": N, "r_star_final": r_star, "exists": r_star is not None}

def safety_boundary_info(N: int, qmax: float, a_s: float, b_s: float, gamma_s: float) -> Dict[str, Optional[int]]:
    """Report minimal toxicities required at final N to trigger safety stop (if enabled)."""
    t_star = safety_stop_threshold(N, qmax, a_s, b_s, gamma_s)
    return {"N": N, "t_safety_min_final": t_star, "exists": t_star is not None}

# =============================================================================
# Caching wrappers (major performance gains)
# =============================================================================
@st.cache_data(show_spinner=False)
def cached_evaluate_design(
    N: int, K: int, look_schedule_tuple: Tuple[int, ...],
    a_e: float, b_e: float, a_s: float, b_s: float,
    p0: float, p1: float, qmax: Optional[float], q1: Optional[float],
    gamma_e: float, psi_fut: float, gamma_s: Optional[float],
    n_sim: int, seed: int
) -> Dict:
    """
    Cached wrapper around evaluate_design. Uses only hashable primitives as key.
    Returns a dict (JSON/pickle friendly).
    """
    design = Design(
        N=N, K_interims=K, look_schedule=list(look_schedule_tuple),
        a_e=a_e, b_e=b_e, a_s=a_s, b_s=b_s, p0=p0, p1=p1,
        qmax=qmax, q1=q1, gamma_e=gamma_e, psi_fut=psi_fut, gamma_s=gamma_s
    )
    oc = evaluate_design(design, n_sim=n_sim, seed=seed)
    return {
        "alpha": oc.alpha, "power": oc.power,
        "ess_p0": oc.ess_p0, "ess_p1": oc.ess_p1,
        "safety_stop_prob_q1": oc.safety_stop_prob_q1,
        "safety_stop_prob_qmax": oc.safety_stop_prob_qmax,
        "avg_looks": oc.avg_looks,
        "success_prob_by_look": oc.success_prob_by_look
    }

@st.cache_data(show_spinner=True)
def cached_grid_search(
    N_min: int, N_max: int, K_min: int, K_max: int,
    p0: float, p1: float, a_e: float, b_e: float, a_s: float, b_s: float,
    qmax: Optional[float], q1: Optional[float],
    gamma_e: float, psi_fut: float, gamma_s: Optional[float],
    n_sim: int, seed: int, alpha_target: float, power_target: float,
    N_budget: Optional[int]
) -> pd.DataFrame:
    """Cached full grid search over (N, K) with per-design cached evaluation."""
    rows = []
    for N in range(N_min, N_max + 1):
        for K in range(K_min, K_max + 1):
            looks = build_equal_looks(N, K)
            oc = cached_evaluate_design(
                N=N, K=K, look_schedule_tuple=tuple(looks),
                a_e=a_e, b_e=b_e, a_s=a_s, b_s=b_s,
                p0=p0, p1=p1, qmax=qmax, q1=q1,
                gamma_e=gamma_e, psi_fut=psi_fut, gamma_s=gamma_s,
                n_sim=n_sim, seed=seed + N + K
            )
            rows.append({
                "N": N,
                "K_interims": K,
                "looks": looks,
                "alpha": oc["alpha"],
                "power": oc["power"],
                "ESS_p0": oc["ess_p0"],
                "ESS_p1": oc["ess_p1"],
                "avg_looks_p1": oc["avg_looks"],
                "safety_stop_q1": oc["safety_stop_prob_q1"],
                "safety_stop_qmax": oc["safety_stop_prob_qmax"],
                "meets_alpha": oc["alpha"] <= alpha_target,
                "meets_power": oc["power"] >= power_target
            })

    df = pd.DataFrame(rows)
    feasible = df[(df["meets_alpha"]) & (df["meets_power"])]
    df["is_feasible"] = False
    if not feasible.empty:
        df.loc[feasible.index, "is_feasible"] = True

    # Tag selections
    df["selection"] = ""
    if not feasible.empty:
        smallest = feasible.sort_values(["N", "K_interims", "ESS_p1"]).head(3)
        sweet_spot = feasible.sort_values(["ESS_p1", "N", "alpha"]).head(3)
        df.loc[smallest.index, "selection"] += "|smallest_N"
        df.loc[sweet_spot.index, "selection"] += "|sweet_spot"

    if N_budget is not None and N_budget > 0:
        under_budget = df[df["N"] <= N_budget]
        if not under_budget.empty:
            high_power = under_budget.sort_values(["power", "ESS_p1"], ascending=[False, True]).head(3)
            df.loc[high_power.index, "selection"] += "|high_power"

    return df

# =============================================================================
# UI
# =============================================================================
st.title("Bayesian Single‑Arm Monitoring Study Designer")
st.caption("Posterior success, predictive futility, and optional safety monitoring with Beta‑Binomial models.")

with st.sidebar:
    st.header("Inputs")

    # Efficacy inputs
    st.number_input("Null efficacy rate p₀", 0.0, 1.0, DEFAULTS["p0"], 0.01, format="%.2f", key="p0")
    st.number_input("Expected efficacy rate p₁", 0.0, 1.0, DEFAULTS["p1"], 0.01, format="%.2f", key="p1")
    st.number_input("Efficacy prior alpha (aₑ)", 0.0, 100.0, DEFAULTS["a_e"], 0.1, format="%.2f", key="a_e")
    st.number_input("Efficacy prior beta (bₑ)", 0.0, 100.0, DEFAULTS["b_e"], 0.1, format="%.2f", key="b_e")
    st.number_input("Posterior success threshold γₑ", 0.5, 0.999, DEFAULTS["gamma_e"], 0.01, key="gamma_e")
    st.number_input("Predictive futility threshold ψ", 0.0, 0.5, DEFAULTS["psi_fut"], 0.01, key="psi_fut")

    st.divider()
    st.checkbox("Enable safety/toxicity monitoring", value=DEFAULTS["enable_safety"], key="enable_safety")
    if st.session_state["enable_safety"]:
        st.number_input("Unacceptable toxicity rate q_max", 0.0, 1.0, DEFAULTS["qmax"], 0.01, format="%.2f", key="qmax")
        st.number_input("Expected toxicity rate q₁", 0.0, 1.0, DEFAULTS["q1"], 0.01, format="%.2f", key="q1")
        st.number_input("Safety prior alpha (aₛ)", 0.0, 100.0, DEFAULTS["a_s"], 0.1, format="%.2f", key="a_s")
        st.number_input("Safety prior beta (bₛ)", 0.0, 100.0, DEFAULTS["b_s"], 0.1, format="%.2f", key="b_s")
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
    st.subheader("Operating targets & simulation")
    st.number_input("Max Type I error (α target)", 0.0, 0.5, DEFAULTS["alpha_target"], 0.01, key="alpha_target")
    st.number_input("Min power (target)", 0.0, 1.0, DEFAULTS["power_target"], 0.01, key="power_target")
    st.number_input("N budget for 'High power' (optional)", 0, 5000, DEFAULTS["N_budget"], 1, key="N_budget")
    st.number_input("Monte Carlo replicates (n_sim)", 1000, 200000, DEFAULTS["n_sim"], 1000, key="n_sim")
    st.number_input("Random seed", 0, 9999999, DEFAULTS["seed"], 1, key="seed")

    run_btn = st.button("Run grid search", type="primary")

st.markdown("### How it works")
st.write("""
- **Efficacy success**: stop if posterior probability that \(p > p_0\) exceeds γₑ.
- **Futility**: stop if the **predictive probability** of meeting the posterior success at the **final N** drops below ψ.
- **Safety (optional)**: stop if posterior probability that \(q > q_{max}\) exceeds γₛ.
- **Interims**: equally spaced looks up to K (rounded). Boundaries use Beta‑Binomial conjugacy.
- We simulate designs to estimate **Type I error (α)**, **power**, and **expected sample size (ESS)**.
""")

# =============================================================================
# Compute only when requested
# =============================================================================
if run_btn:
    # Gather parameters cleanly
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
        qmax=st.session_state.get("qmax") if st.session_state["enable_safety"] else None,
        q1=st.session_state.get("q1") if st.session_state["enable_safety"] else None,
        gamma_e=float(st.session_state["gamma_e"]),
        psi_fut=float(st.session_state["psi_fut"]),
        gamma_s=st.session_state.get("gamma_s") if st.session_state["enable_safety"] else None,
        n_sim=int(st.session_state["n_sim"]),
        seed=int(st.session_state["seed"]),
        alpha_target=float(st.session_state["alpha_target"]),
        power_target=float(st.session_state["power_target"]),
        N_budget=int(st.session_state["N_budget"]) if int(st.session_state["N_budget"]) > 0 else None
    )

    # Basic workload notice (N*K combos)
    total_designs = (params["N_max"] - params["N_min"] + 1) * (params["K_max"] - params["K_min"] + 1)
    st.info(f"Evaluating ~{total_designs} design(s) × {params['n_sim']:,} simulations each (cached).")

    start = time.perf_counter()
    with st.spinner("Running cached grid search..."):
        df = cached_grid_search(**params)
    elapsed = time.perf_counter() - start
    st.success(f"Grid search finished in {elapsed:0.1f}s (cached on inputs).")

    # If no feasible designs, show friendly guidance (no auto-rerun)
    feasible = df[df["is_feasible"]]
    if feasible.empty:
        st.warning(
            "No feasible designs found that meet both α and power targets in this grid. "
            "Try increasing N_max, relaxing γₑ, or increasing ψ, then press **Run grid search**."
        )
        # Diagnostics at current N_max
        st.markdown("#### Infeasibility diagnostics")
        info = final_success_boundary_info(
            N=int(st.session_state["N_max"]),
            p0=float(st.session_state["p0"]),
            a_e=float(st.session_state["a_e"]),
            b_e=float(st.session_state["b_e"]),
            gamma_e=float(st.session_state["gamma_e"])
        )
        if info["exists"]:
            st.write(f"At N={info['N']}, minimal successes for posterior success: **r* = {info['r_star_final']}**.")
        else:
            st.write(
                f"At N={info['N']}, **no number of successes** can satisfy posterior success with γₑ={st.session_state['gamma_e']:.2f}. "
                "Relax γₑ or increase N."
            )

        if st.session_state["enable_safety"] and st.session_state.get("gamma_s") is not None and st.session_state.get("qmax") is not None:
            s_info = safety_boundary_info(
                N=int(st.session_state["N_max"]),
                qmax=float(st.session_state["qmax"]),
                a_s=float(st.session_state["a_s"]),
                b_s=float(st.session_state["b_s"]),
                gamma_s=float(st.session_state["gamma_s"])
            )
            if s_info["exists"]:
                st.write(f"Safety stop at N={s_info['N']} triggers at **t ≥ {s_info['t_safety_min_final']}** toxicities.")
            else:
                st.write("Safety boundary does not exist at N_max with current γₛ; check inputs.")

        # Convenience tweaks (do NOT auto-run to avoid surprise slowness)
        st.markdown("#### Quick input tweaks (adjust, then click **Run grid search**)")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Expand N_max by +50"):
                st.session_state["N_max"] = int(st.session_state["N_max"]) + 50
                st.toast("N_max increased. Click **Run grid search** to recompute.", icon="🟢")
        with c2:
            if st.button("Relax γₑ by −0.05"):
                st.session_state["gamma_e"] = max(0.50, float(st.session_state["gamma_e"]) - 0.05)
                st.toast("γₑ relaxed. Click **Run grid search** to recompute.", icon="🟢")
        with c3:
            if st.button("Increase ψ by +0.05"):
                st.session_state["psi_fut"] = min(0.50, float(st.session_state["psi_fut"]) + 0.05)
                st.toast("ψ increased. Click **Run grid search** to recompute.", icon="🟢")

    # Summary table
    st.markdown("### Summary of Candidate Designs")
    st.dataframe(
        df[[
            "selection","is_feasible","N","K_interims","looks",
            "alpha","power","ESS_p0","ESS_p1","avg_looks_p1",
            "safety_stop_q1","safety_stop_qmax"
        ]].style.format({
            "alpha": "{:.3f}", "power": "{:.3f}", "ESS_p0": "{:.1f}",
            "ESS_p1": "{:.1f}", "avg_looks_p1": "{:.2f}",
            "safety_stop_q1": "{:.3f}", "safety_stop_qmax": "{:.3f}"
        }),
        use_container_width=True
    )

    # Recommendations
    st.markdown("### Recommended Designs")
    rec_small = df[df["selection"].str.contains("smallest_N", na=False)]
    rec_high = df[df["selection"].str.contains("high_power", na=False)]
    rec_sweet = df[df["selection"].str.contains("sweet_spot", na=False)]

    cols = st.columns(3)
    with cols[0]:
        st.subheader("Smallest N (meets α & power)")
        if rec_small.empty:
            st.info("None in current grid.")
        else:
            st.dataframe(rec_small[["N","K_interims","looks","alpha","power","ESS_p1"]].style.format({
                "alpha":"{:.3f}","power":"{:.3f}","ESS_p1":"{:.1f}"
            }))
    with cols[1]:
        st.subheader("High power (≤ N budget)")
        if rec_high.empty:
            st.info("None under N budget.")
        else:
            st.dataframe(rec_high[["N","K_interims","looks","alpha","power","ESS_p1"]].style.format({
                "alpha":"{:.3f}","power":"{:.3f}","ESS_p1":"{:.1f}"
            }))
    with cols[2]:
        st.subheader("Sweet spot (min ESS₍p₁₎)")
        if rec_sweet.empty:
            st.info("No feasible sweet spots.")
        else:
            st.dataframe(rec_sweet[["N","K_interims","looks","alpha","power","ESS_p1"]].style.format({
                "alpha":"{:.3f}","power":"{:.3f}","ESS_p1":"{:.1f}"
            }))

    # Download
    st.download_button(
        "Download full results (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="bayes_single_arm_designs.csv",
        mime="text/csv"
    )

    # Visualization
    st.markdown("### Power vs N")
    import altair as alt
    chart_df = df.copy()
    chart_df["feasible_flag"] = chart_df["is_feasible"].map({True:"Feasible", False:"Not feasible"})
    power_chart = alt.Chart(chart_df).mark_circle(size=80).encode(
        x=alt.X("N:Q", title="Max N"),
        y=alt.Y("power:Q", title="Power"),
        color=alt.Color("feasible_flag:N", title="Feasibility"),
        tooltip=["N","K_interims","alpha","power","ESS_p1","looks"]
    ).properties(width=900, height=360)
    st.altair_chart(power_chart, use_container_width=True)

    # Optional, on-demand sensitivity sweep (FAST, small sims)
    with st.expander("Quick sensitivity sweep (γₑ & ψ) — on demand"):
        if st.button("Run sensitivity sweep (small n_sim)"):
            gamma_center = float(st.session_state["gamma_e"])
            psi_center = float(st.session_state["psi_fut"])
            gamma_grid = np.clip(np.linspace(max(0.5, gamma_center - 0.05), min(0.999, gamma_center + 0.05), 5), 0.5, 0.999)
            psi_grid = np.clip(np.linspace(max(0.0, psi_center - 0.05), min(0.5, psi_center + 0.05), 5), 0.0, 0.5)

            sens_rows = []
            Nmax = int(st.session_state["N_max"])
            Kmax = int(st.session_state["K_max"])
            looks = build_equal_looks(Nmax, Kmax)
            for ge in gamma_grid:
                for ps in psi_grid:
                    oc = cached_evaluate_design(
                        N=Nmax, K=Kmax, look_schedule_tuple=tuple(looks),
                        a_e=float(st.session_state["a_e"]), b_e=float(st.session_state["b_e"]),
                        a_s=float(st.session_state.get("a_s") or 0.0), b_s=float(st.session_state.get("b_s") or 0.0),
                        p0=float(st.session_state["p0"]), p1=float(st.session_state["p1"]),
                        qmax=st.session_state.get("qmax") if st.session_state["enable_safety"] else None,
                        q1=st.session_state.get("q1") if st.session_state["enable_safety"] else None,
                        gamma_e=float(ge), psi_fut=float(ps),
                        gamma_s=st.session_state.get("gamma_s") if st.session_state["enable_safety"] else None,
                        n_sim=max(3000, int(int(st.session_state["n_sim"]) / 5)),
                        seed=int(st.session_state["seed"]) + int(ge * 1000) + int(ps * 1000)
                    )
                    sens_rows.append({"γₑ": ge, "ψ": ps, "alpha": oc["alpha"], "power": oc["power"], "ESS_p1": oc["ess_p1"]})
            sens_df = pd.DataFrame(sens_rows).sort_values(["power","alpha"], ascending=[False, True])
            st.dataframe(sens_df.style.format({"γₑ":"{:.3f}","ψ":"{:.3f}","alpha":"{:.3f}","power":"{:.3f}","ESS_p1":"{:.1f}"}))

# Footer guidance
st.divider()
st.markdown("""
**Tips & Notes**
- Consider **Jeffreys priors** \\(\\text{Beta}(0.5,0.5)\\) or **Beta(1,1)** as non-informative starts.
- Typical choices: γₑ ∈ [0.90, 0.99], ψ ∈ [0.05, 0.20], γₛ ∈ [0.80, 0.95].
- Safety is checked **before** efficacy at each look.
- Reported: **Type I error** under \\(p=p_0\\) and **Power** under \\(p=p_1\\) (toxicity at \\(q=q_1\\)).
- Increase **n_sim** after identifying promising regions to firm up estimates.
""")

st.caption("Exploratory design/simulation. Final trial designs should be reviewed by a qualified statistician and align with regulatory guidance.")
