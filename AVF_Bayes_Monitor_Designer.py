# =============================================================================
# AVF_Bayes_Monitor_Designer.py  (Streamlit-optimized, with n_sim fixes)
# =============================================================================

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
# Streamlit configuration
# =============================================================================
st.set_page_config(
    page_title="Bayesian Single-Arm Monitor Designer",
    layout="wide",
)


# =============================================================================
# Default settings (SUPER FAST MODE)
# =============================================================================
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

    # Grid search
    "N_min": 30,
    "N_max": 120,
    "K_min": 1,
    "K_max": 4,

    # Targets
    "alpha_target": 0.10,
    "power_target": 0.80,
    "N_budget": 80,

    # Simulation defaults (key speed-up)
    "n_sim": 1,
    "seed": 12345,
}


# =============================================================================
# Bayesian helper functions
# =============================================================================
def posterior_prob_p_greater_than(p0: float, a: float, b: float) -> float:
    return 1 - beta.cdf(p0, a, b)


def posterior_prob_q_exceeds(qmax: float, a: float, b: float) -> float:
    return 1 - beta.cdf(qmax, a, b)


def minimal_successes_for_posterior_success(
    n: int, p0: float, a0: float, b0: float, gamma_e: float
) -> Optional[int]:
    for r in range(n + 1):
        if posterior_prob_p_greater_than(p0, a0 + r, b0 + n - r) >= gamma_e:
            return r
    return None


def safety_stop_threshold(
    n: int, qmax: float, a_s: float, b_s: float, gamma_s: float
) -> Optional[int]:
    for t in range(n + 1):
        if posterior_prob_q_exceeds(qmax, a_s + t, b_s + n - t) >= gamma_s:
            return t
    return None


def beta_binomial_predictive_prob_at_least(
    current_r: int, current_n: int, final_N: int,
    a0: float, b0: float, r_star_final: int
) -> float:
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


# =============================================================================
# Data structures
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
    bounds = {}
    r_star_final = minimal_successes_for_posterior_success(
        design.N, design.p0, design.a_e, design.b_e, design.gamma_e
    )

    for n in design.look_schedule:
        r_success_min = minimal_successes_for_posterior_success(
            n, design.p0, design.a_e, design.b_e, design.gamma_e
        )
        t_safety_min = (
            safety_stop_threshold(n, design.qmax, design.a_s, design.b_s, design.gamma_s)
            if (design.qmax is not None and design.gamma_s is not None)
            else None
        )
        bounds[n] = {
            "r_success_min": r_success_min,
            "t_safety_min": t_safety_min,
            "r_star_final": r_star_final,
        }
    return bounds


# =============================================================================
# Simulation engine
# =============================================================================
def simulate_one_trial(
    design: Design, true_p: float, true_q: Optional[float],
    bounds: Dict[int, Dict[str, Optional[int]]], rng: np.random.Generator
) -> Tuple[bool, int, bool, int]:
    N = design.N
    responses = rng.binomial(1, true_p, N)
    tox = rng.binomial(1, true_q, N) if true_q is not None else np.zeros(N)

    looks_used = 0
    for n in design.look_schedule:
        r = int(responses[:n].sum())
        t = int(tox[:n].sum())
        looks_used += 1

        # Safety
        if design.qmax is not None and design.gamma_s is not None:
            t_s = bounds[n]["t_safety_min"]
            if t_s is not None and t >= t_s:
                return (False, n, True, looks_used)

        # Efficacy success
        r_s = bounds[n]["r_success_min"]
        if r_s is not None and r >= r_s:
            return (True, n, False, looks_used)

        # Futility
        r_star_final = bounds[n]["r_star_final"]
        if n < N and r_star_final is not None:
            ppos = beta_binomial_predictive_prob_at_least(
                r, n, N, design.a_e, design.b_e, r_star_final
            )
            if ppos < design.psi_fut:
                return (False, n, False, looks_used)

        # Final look
        if n == N:
            if r_star_final is not None and r >= r_star_final:
                return (True, n, False, looks_used)
            return (False, n, False, looks_used)

    return (False, N, False, looks_used)


def evaluate_design(design: Design, n_sim: int = 1, seed: int = 123) -> OperatingCharacteristics:
    """Default n_sim=1 (it will be overridden by the widget)."""
    rng = np.random.default_rng(seed)
    bounds = compute_boundaries(design)

    # p0
    runs_p0 = [
        simulate_one_trial(design, design.p0, design.q1, bounds, rng)
        for _ in range(n_sim)
    ]
    alpha = np.mean([s for (s, _, _, _) in runs_p0])
    ess_p0 = np.mean([n_used for (_, n_used, _, _) in runs_p0])

    # p1
    runs_p1 = [
        simulate_one_trial(design, design.p1, design.q1, bounds, rng)
        for _ in range(n_sim)
    ]
    power = np.mean([s for (s, _, _, _) in runs_p1])
    ess_p1 = np.mean([n_used for (_, n_used, _, _) in runs_p1])
    avg_looks = np.mean([looks for (_, _, _, looks) in runs_p1])

    # Safety (q1 and qmax)
    safety_stop_prob_q1 = None
    safety_stop_prob_qmax = None
    if design.qmax is not None and design.q1 is not None:
        saf_runs_q1 = [
            simulate_one_trial(design, design.p1, design.q1, bounds, rng)
            for _ in range(n_sim)
        ]
        safety_stop_prob_q1 = np.mean([sf for (_, _, sf, _) in saf_runs_q1])

        saf_runs_qmax = [
            simulate_one_trial(design, design.p1, design.qmax, bounds, rng)
            for _ in range(n_sim)
        ]
        safety_stop_prob_qmax = np.mean([sf for (_, _, sf, _) in saf_runs_qmax])

    # Success by look
    success_by_look = {}
    for n in design.look_schedule:
        success_by_look[n] = np.mean(
            [1 if (s and n_used == n) else 0 for (s, n_used, _, _) in runs_p1]
        )

    return OperatingCharacteristics(
        alpha=float(alpha),
        power=float(power),
        ess_p0=float(ess_p0),
        ess_p1=float(ess_p1),
        safety_stop_prob_q1=(None if safety_stop_prob_q1 is None else float(safety_stop_prob_q1)),
        safety_stop_prob_qmax=(None if safety_stop_prob_qmax is None else float(safety_stop_prob_qmax)),
        avg_looks=float(avg_looks),
        success_prob_by_look=success_by_look,
    )


# =============================================================================
# Look schedule helper
# =============================================================================
def build_equal_looks(N: int, K: int) -> List[int]:
    looks = []
    for k in range(1, K + 1):
        looks.append(int(round((k / (K + 1)) * N)))
    if not looks or looks[-1] != N:
        looks.append(N)
    looks = sorted(set(looks))
    return looks


# =============================================================================
# Cached wrappers (massive speedups)
# =============================================================================
@st.cache_data(show_spinner=True)
def cached_evaluate(
    N: int, K: int, look_tuple: Tuple[int, ...],
    a_e: float, b_e: float, a_s: float, b_s: float,
    p0: float, p1: float, qmax: Optional[float], q1: Optional[float],
    gamma_e: float, psi_fut: float, gamma_s: Optional[float],
    n_sim: int, seed: int
) -> Dict:
    design = Design(
        N, K, list(look_tuple),
        a_e, b_e, a_s, b_s, p0, p1,
        qmax, q1, gamma_e, psi_fut, gamma_s
    )
    oc = evaluate_design(design, n_sim=n_sim, seed=seed)
    return oc.__dict__


@st.cache_data(show_spinner=True)
def cached_grid_search(
    N_min, N_max, K_min, K_max,
    p0, p1, a_e, b_e, a_s, b_s, qmax, q1,
    gamma_e, psi_fut, gamma_s, n_sim, seed,
    alpha_target, power_target, N_budget
) -> pd.DataFrame:

    rows = []
    for N in range(N_min, N_max + 1):
        for K in range(K_min, K_max + 1):
            looks = build_equal_looks(N, K)

            oc = cached_evaluate(
                N, K, tuple(looks),
                a_e, b_e, a_s, b_s,
                p0, p1, qmax, q1,
                gamma_e, psi_fut, gamma_s,
                n_sim, seed + N + K
            )

            rows.append(
                dict(
                    N=N,
                    K_interims=K,
                    looks=looks,
                    alpha=oc["alpha"],
                    power=oc["power"],
                    ESS_p0=oc["ess_p0"],
                    ESS_p1=oc["ess_p1"],
                    avg_looks_p1=oc["avg_looks"],
                    safety_stop_q1=oc["safety_stop_prob_q1"],
                    safety_stop_qmax=oc["safety_stop_prob_qmax"],
                    meets_alpha=oc["alpha"] <= alpha_target,
                    meets_power=oc["power"] >= power_target,
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
        if not under.empty:
            best = under.sort_values(["power", "ESS_p1"], ascending=[False, True]).head(3)
            df.loc[best.index, "selection"] += "|high_power"

    return df


# =============================================================================
# UI – Sidebar
# =============================================================================
with st.sidebar:
    st.header("Inputs")

    # One-click way to clear OLD widget state and cached results
    if st.button("Reset UI & clear cache"):
        st.cache_data.clear()
        st.session_state.clear()  # clears all widget keys and values
        st.success("Cache & UI state cleared. Please rerun your grid search.")
        st.stop()  # end this run so the app reloads fresh

    st.markdown("⚡ *Quick-Scan Mode Enabled (n_sim defaults to 1)*")

    st.number_input("Null efficacy rate p₀", 0.0, 1.0, DEFAULTS["p0"], 0.01, key="p0")
    st.number_input("Expected efficacy rate p₁", 0.0, 1.0, DEFAULTS["p1"], 0.01, key="p1")

    st.number_input("Efficacy prior alpha aₑ", 0.0, 100.0, DEFAULTS["a_e"], 0.1, key="a_e")
    st.number_input("Efficacy prior beta bₑ", 0.0, 100.0, DEFAULTS["b_e"], 0.1, key="b_e")

    st.number_input("Posterior success threshold γₑ", 0.5, 0.999, DEFAULTS["gamma_e"], 0.01, key="gamma_e")
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
    st.subheader("Grid Search Settings")
    st.number_input("Minimum N", 5, 5000, DEFAULTS["N_min"], 1, key="N_min")
    st.number_input("Maximum N", st.session_state["N_min"], 5000, DEFAULTS["N_max"], 1, key="N_max")
    st.number_input("Min interims K", 0, 40, DEFAULTS["K_min"], 1, key="K_min")
    st.number_input("Max interims K", st.session_state["K_min"], 40, DEFAULTS["K_max"], 1, key="K_max")

    st.divider()
    st.subheader("Targets & Simulation")

    st.number_input("Max Type I error α target", 0.0, 0.5, DEFAULTS["alpha_target"], 0.01, key="alpha_target")
    st.number_input("Min Power target", 0.0, 1.0, DEFAULTS["power_target"], 0.01, key="power_target")
    st.number_input("N budget (optional)", 0, 5000, DEFAULTS["N_budget"], 1, key="N_budget")

    # **Allow sub-1k simulations (down to 1)** — new key to break any stale constraints
    st.number_input(
        "Monte Carlo replicates (n_sim)",
        min_value=1,
        max_value=200000,  # generous headroom
        value=DEFAULTS["n_sim"],
        step=1,
        key="n_sim_v2",
        help="1–100: instant quick scan; 100–1000: fast; 2000+: accurate"
    )

    st.number_input("Random seed", 0, 9999999, DEFAULTS["seed"], 1, key="seed")

    run_btn = st.button("Run grid search", type="primary")


# =============================================================================
# Main UI
# =============================================================================
st.title("Bayesian Single‑Arm Monitoring Study Designer")

st.info(
    "⚡ **Quick‑Scan Mode:** Default `n_sim = 1`.\n\n"
    "Set `n_sim` ≥ 2000 for more stable, accurate operating characteristics."
)

st.markdown("---")

# =============================================================================
# Run grid search on demand
# =============================================================================
if run_btn:

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
    )

    num_designs = (params["N_max"] - params["N_min"] + 1) * (params["K_max"] - params["K_min"] + 1)
    st.write(f"Evaluating ~{num_designs} designs × {params['n_sim']} sims each (cached).")

    # Sanity check: show what n_sim is actually used
    st.caption(f"Running with n_sim = {params['n_sim']}")

    start = time.time()
    with st.spinner("Running grid search..."):
        df = cached_grid_search(**params)
    st.success(f"Completed in {time.time() - start:.2f} sec.")

    # Display results
    feasible = df[df["is_feasible"]]
    if feasible.empty:
        st.warning("No feasible designs met α & power targets.")

    st.subheader("Summary of Designs")
    st.dataframe(df, use_container_width=True)

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
            st.dataframe(subset, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        "bayes_monitor_designs.csv"
    )

    st.markdown("---")
    # Quick chart (import altair only when needed)
    import altair as alt
    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x="N:Q",
            y="power:Q",
            color="is_feasible:N",
            tooltip=["N","K_interims","alpha","power","ESS_p1"]
        )
        .properties(width=800, height=300)
    )
    st.altair_chart(chart, use_container_width=True)
