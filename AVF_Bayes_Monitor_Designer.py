# bayes_single_arm_app.py
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import streamlit as st
from scipy.stats import beta
from scipy.special import comb, beta as beta_fn

# ------------------------------
# Utility functions (Bayesian)
# ------------------------------

def posterior_prob_p_greater_than(p0: float, a: float, b: float) -> float:
    """
    Compute posterior Pr(p > p0) when posterior is Beta(a, b).
    """
    return 1.0 - beta.cdf(p0, a, b)

def posterior_prob_q_exceeds(qmax: float, a: float, b: float) -> float:
    """
    Compute posterior Pr(q > qmax) when posterior is Beta(a, b).
    """
    return 1.0 - beta.cdf(qmax, a, b)

def minimal_successes_for_posterior_success(n: int, p0: float, a0: float, b0: float, gamma_e: float) -> Optional[int]:
    """
    For final look size n, compute the smallest r such that Pr(p > p0 | r, n) >= gamma_e.
    Returns None if no such r exists (i.e., criterion cannot be met).
    """
    for r in range(0, n + 1):
        a_post = a0 + r
        b_post = b0 + n - r
        if posterior_prob_p_greater_than(p0, a_post, b_post) >= gamma_e:
            return r
    return None

def safety_stop_threshold(n: int, qmax: float, a_s: float, b_s: float, gamma_s: float) -> Optional[int]:
    """
    For look size n, compute the smallest t (toxicities) such that Pr(q > qmax | t, n) >= gamma_s.
    Returns None if no such t exists.
    """
    for t in range(0, n + 1):
        a_post = a_s + t
        b_post = b_s + n - t
        if posterior_prob_q_exceeds(qmax, a_post, b_post) >= gamma_s:
            return t
    return None

def beta_binomial_predictive_prob_at_least(current_r: int, current_n: int, final_N: int,
                                           a0: float, b0: float, r_star_final: int) -> float:
    """
    Predictive probability (over future responses) that final total successes >= r_star_final.
    Posterior after current data is Beta(a_post, b_post). Future m = final_N - current_n.
    Predictive distribution for future successes J is Beta-Binomial:
        Pr(J=j) = C(m, j) * Beta(j + a_post, m - j + b_post) / Beta(a_post, b_post)
    """
    m = final_N - current_n
    a_post = a0 + current_r
    b_post = b0 + current_n - current_r
    if m < 0:
        raise ValueError("final_N must be >= current_n")
    if m == 0:
        # No patients left; we're at the final analysis.
        return 1.0 if current_r >= r_star_final else 0.0

    # Sum Pr(J >= r_star_final - current_r)
    j_min = max(0, r_star_final - current_r)
    if j_min <= 0:
        return 1.0
    total = 0.0
    for j in range(j_min, m + 1):
        total += comb(m, j) * beta_fn(j + a_post, m - j + b_post) / beta_fn(a_post, b_post)
    return float(total)

# ------------------------------
# Design structures
# ------------------------------

@dataclass
class Design:
    N: int                      # Maximum sample size
    K_interims: int             # Number of interim looks (excluding final)
    look_schedule: List[int]    # Cumulative sample sizes at each look (includes final N)
    a_e: float                  # Efficacy prior alpha
    b_e: float                  # Efficacy prior beta
    a_s: float                  # Safety prior alpha
    b_s: float                  # Safety prior beta
    p0: float                   # Null efficacy rate
    p1: float                   # Expected efficacy rate
    qmax: Optional[float]       # Unacceptable toxicity rate (None to disable safety)
    q1: Optional[float]         # Expected toxicity rate
    gamma_e: float              # Posterior success threshold for efficacy
    psi_fut: float              # Predictive futility threshold (stop if PP(success)<psi_fut)
    gamma_s: Optional[float]    # Posterior threshold for safety stop (None to disable safety)

@dataclass
class OperatingCharacteristics:
    alpha: float                   # Type I error (efficacy success when p=p0)
    power: float                   # Power (efficacy success when p=p1)
    ess_p0: float                  # Expected sample size under p0
    ess_p1: float                  # Expected sample size under p1
    safety_stop_prob_q1: Optional[float]    # P(stop for safety when q=q1)
    safety_stop_prob_qmax: Optional[float]  # P(stop for safety when q=qmax)
    avg_looks: float               # Average number of looks used under p1
    success_prob_by_look: Dict[int, float]  # Success probability at each look under p1

# ------------------------------
# Boundary computation
# ------------------------------

def compute_boundaries(design: Design) -> Dict[int, Dict[str, Optional[int]]]:
    """
    For each look size n in the schedule, compute:
      - r_success_min: minimal successes to trigger posterior success
      - t_safety_min: minimal toxicities to trigger safety stop
      - r_star_final: minimal successes at final N to meet posterior success
    Also defines that futility is triggered when predictive probability of meeting r_star_final at final
    is < psi_fut (computed dynamically during simulation, not a static boundary).
    """
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
        bounds[n] = {
            "r_success_min": r_success_min,
            "t_safety_min": t_safety_min,
            "r_star_final": r_star_final
        }
    return bounds

# ------------------------------
# Simulation engine
# ------------------------------

def simulate_one_trial(design: Design, true_p: float, true_q: Optional[float],
                       bounds: Dict[int, Dict[str, Optional[int]]],
                       rng: np.random.Generator) -> Tuple[bool, int, bool, int]:
    """
    Simulate a single trial path with sequential Bernoulli outcomes for efficacy and safety.
    Returns:
      (success_flag, n_used, safety_stopped_flag, looks_used)
    """
    N = design.N
    # Generate patient-level outcomes
    responses = rng.binomial(1, true_p, size=N)
    tox = rng.binomial(1, true_q, size=N) if (true_q is not None) else np.zeros(N, dtype=int)

    r = 0
    t = 0
    success = False
    safety_stopped = False
    looks_used = 0

    for n in design.look_schedule:
        # Accumulate to this look
        r = int(np.sum(responses[:n]))
        t = int(np.sum(tox[:n]))
        looks_used += 1

        # Safety check first (if enabled)
        if design.qmax is not None and design.gamma_s is not None:
            t_safety_min = bounds[n]["t_safety_min"]
            if t_safety_min is not None and t >= t_safety_min:
                safety_stopped = True
                return (False, n, True, looks_used)  # Stop for safety; no efficacy success

        # Efficacy success check
        r_success_min = bounds[n]["r_success_min"]
        if r_success_min is not None and r >= r_success_min:
            success = True
            return (True, n, False, looks_used)

        # Futility via predictive probability (except at final look)
        r_star_final = bounds[n]["r_star_final"]
        if n < N and r_star_final is not None:
            ppos = beta_binomial_predictive_prob_at_least(
                current_r=r, current_n=n, final_N=N,
                a0=design.a_e, b0=design.b_e, r_star_final=r_star_final
            )
            if ppos < design.psi_fut:
                return (False, n, False, looks_used)

        # Final look: conclude success/failure if no early stop
        if n == N:
            if r_star_final is not None and r >= r_star_final:
                success = True
            return (success, n, False, looks_used)

    # Should not reach here
    return (False, N, False, looks_used)

def evaluate_design(design: Design, n_sim: int = 20000, seed: int = 12345) -> OperatingCharacteristics:
    """
    Evaluate operating characteristics using Monte Carlo simulation.
    """
    rng = np.random.default_rng(seed)

    bounds = compute_boundaries(design)

    # Under p0
    results_p0 = [simulate_one_trial(design, design.p0, design.q1, bounds, rng) for _ in range(n_sim)]
    alpha = np.mean([s for (s, _, _, _) in results_p0])
    ess_p0 = np.mean([n_used for (_, n_used, _, _) in results_p0])

    # Under p1
    results_p1 = [simulate_one_trial(design, design.p1, design.q1, bounds, rng) for _ in range(n_sim)]
    power = np.mean([s for (s, _, _, _) in results_p1])
    ess_p1 = np.mean([n_used for (_, n_used, _, _) in results_p1])
    avg_looks = np.mean([looks for (_, _, _, looks) in results_p1])

    # Safety metrics (if enabled)
    safety_stop_prob_q1 = None
    safety_stop_prob_qmax = None
    if design.qmax is not None and design.gamma_s is not None and design.q1 is not None:
        results_q1 = [simulate_one_trial(design, design.p1, design.q1, bounds, rng) for _ in range(n_sim)]
        safety_stop_prob_q1 = np.mean([sf for (_, _, sf, _) in results_q1])

        results_qmax = [simulate_one_trial(design, design.p1, design.qmax, bounds, rng) for _ in range(n_sim)]
        safety_stop_prob_qmax = np.mean([sf for (_, _, sf, _) in results_qmax])

    # Success probability by look under p1
    success_by_look_counts = {n: 0 for n in design.look_schedule}
    for (s, n_used, _, _) in results_p1:
        if s:
            success_by_look_counts[n_used] += 1
    success_prob_by_look = {n: success_by_look_counts[n] / n_sim for n in design.look_schedule}

    return OperatingCharacteristics(
        alpha=alpha, power=power,
        ess_p0=ess_p0, ess_p1=ess_p1,
        safety_stop_prob_q1=safety_stop_prob_q1,
        safety_stop_prob_qmax=safety_stop_prob_qmax,
        avg_looks=avg_looks,
        success_prob_by_look=success_prob_by_look
    )

# ------------------------------
# Helper: build look schedule
# ------------------------------

def build_equal_looks(N: int, K_interims: int) -> List[int]:
    """
    Build K_interims interim looks + final at N, spaced roughly equally.
    """
    if K_interims < 0:
        K_interims = 0
    looks = []
    for k in range(1, K_interims + 1):
        looks.append(int(round((k / (K_interims + 1)) * N)))
    if looks and looks[-1] == N:
        looks[-1] = max(1, N - 1)
    if len(looks) == 0 or looks[-1] < N:
        looks.append(N)
    # Ensure strictly increasing and >=1
    looks = sorted(set([max(1, x) for x in looks]))
    if looks[-1] != N:
        looks.append(N)
    return looks

# ------------------------------
# Grid search over designs
# ------------------------------

def grid_search_designs(
    N_min: int, N_max: int, K_min: int, K_max: int,
    p0: float, p1: float,
    a_e: float, b_e: float,
    a_s: float, b_s: float,
    qmax: Optional[float], q1: Optional[float],
    gamma_e: float, psi_fut: float, gamma_s: Optional[float],
    n_sim: int, seed: int,
    alpha_target: float, power_target: float,
    N_budget: Optional[int] = None
) -> pd.DataFrame:
    """
    Scan a grid of (N, K_interims), evaluate characteristics, and return a DataFrame.
    """
    rows = []
    for N in range(N_min, N_max + 1):
        for K in range(K_min, K_max + 1):
            look_schedule = build_equal_looks(N, K)
            design = Design(
                N=N, K_interims=K, look_schedule=look_schedule,
                a_e=a_e, b_e=b_e, a_s=a_s, b_s=b_s,
                p0=p0, p1=p1, qmax=qmax, q1=q1,
                gamma_e=gamma_e, psi_fut=psi_fut, gamma_s=gamma_s
            )
            oc = evaluate_design(design, n_sim=n_sim, seed=seed + N + K)
            rows.append({
                "N": N,
                "K_interims": K,
                "looks": look_schedule,
                "alpha": oc.alpha,
                "power": oc.power,
                "ESS_p0": oc.ess_p0,
                "ESS_p1": oc.ess_p1,
                "avg_looks_p1": oc.avg_looks,
                "safety_stop_q1": oc.safety_stop_prob_q1,
                "safety_stop_qmax": oc.safety_stop_prob_qmax,
                "meets_alpha": oc.alpha <= alpha_target,
                "meets_power": oc.power >= power_target
            })
    df = pd.DataFrame(rows)
    # Objective tags
    feasible = df[(df["meets_alpha"]) & (df["meets_power"])]
    df["is_feasible"] = False
    df.loc[feasible.index, "is_feasible"] = True

    # Smallest N among feasible
    smallest = None
    if not feasible.empty:
        smallest = feasible.sort_values(["N", "K_interims", "ESS_p1"]).head(3)

    # High power under N budget
    high_power = None
    if N_budget is not None:
        under_budget = df[df["N"] <= N_budget]
        if not under_budget.empty:
            high_power = under_budget.sort_values(["power", "ESS_p1"], ascending=[False, True]).head(3)

    # Sweet spot: feasible designs minimizing ESS_p1
    sweet_spot = None
    if not feasible.empty:
        sweet_spot = feasible.sort_values(["ESS_p1", "N", "alpha"]).head(3)

    # Tag selections
    df["selection"] = ""
    if smallest is not None:
        df.loc[smallest.index, "selection"] = df.loc[smallest.index, "selection"] + "|smallest_N"
    if high_power is not None:
        df.loc[high_power.index, "selection"] = df.loc[high_power.index, "selection"] + "|high_power"
    if sweet_spot is not None:
        df.loc[sweet_spot.index, "selection"] = df.loc[sweet_spot.index, "selection"] + "|sweet_spot"

    return df

# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="Bayesian Single-Arm Monitor Designer", layout="wide")

st.title("Bayesian Single‑Arm Monitoring Study Designer")
st.caption("Posterior success, predictive futility, and optional safety monitoring with Beta‑Binomial models.")

with st.sidebar:
    st.header("Inputs")

    # Efficacy inputs
    p0 = st.number_input("Null efficacy rate p₀", min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")
    p1 = st.number_input("Expected efficacy rate p₁", min_value=0.0, max_value=1.0, value=0.40, step=0.01, format="%.2f")
    a_e = st.number_input("Efficacy prior alpha (aₑ)", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    b_e = st.number_input("Efficacy prior beta (bₑ)", min_value=0.0, value=1.0, step=0.1, format="%.2f")
    gamma_e = st.number_input("Posterior success threshold γₑ (e.g., 0.95)", min_value=0.5, max_value=0.999, value=0.95, step=0.01)

    psi_fut = st.number_input("Predictive futility threshold ψ (e.g., 0.05)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)

    st.divider()
    enable_safety = st.checkbox("Enable safety/toxicity monitoring", value=True)
    qmax = None
    q1_val = None
    a_s = 1.0
    b_s = 1.0
    gamma_s = None
    if enable_safety:
        qmax = st.number_input("Unacceptable toxicity rate q_max", min_value=0.0, max_value=1.0, value=0.30, step=0.01, format="%.2f")
        q1_val = st.number_input("Expected toxicity rate q₁", min_value=0.0, max_value=1.0, value=0.15, step=0.01, format="%.2f")
        a_s = st.number_input("Safety prior alpha (aₛ)", min_value=0.0, value=1.0, step=0.1, format="%.2f")
        b_s = st.number_input("Safety prior beta (bₛ)", min_value=0.0, value=1.0, step=0.1, format="%.2f")
        gamma_s = st.number_input("Posterior safety threshold γₛ (e.g., 0.90)", min_value=0.5, max_value=0.999, value=0.90, step=0.01)

    st.divider()
    st.subheader("Grid search")
    N_min = st.number_input("Minimum N", min_value=5, max_value=5000, value=30, step=1)
    N_max = st.number_input("Maximum N", min_value=N_min, max_value=5000, value=120, step=1)
    K_min = st.number_input("Min interim looks (K)", min_value=0, max_value=40, value=1, step=1)
    K_max = st.number_input("Max interim looks (K)", min_value=K_min, max_value=40, value=4, step=1)

    st.divider()
    st.subheader("Operating targets & simulation")
    alpha_target = st.number_input("Max Type I error (α target)", min_value=0.0, max_value=0.5, value=0.10, step=0.01)
    power_target = st.number_input("Min power (target)", min_value=0.0, max_value=1.0, value=0.80, step=0.01)
    N_budget = st.number_input("N budget for 'High power' (optional)", min_value=0, max_value=5000, value=80, step=1)
    n_sim = st.number_input("Monte Carlo replicates", min_value=1000, max_value=200000, value=20000, step=1000)
    seed = st.number_input("Random seed", min_value=0, max_value=9999999, value=12345, step=1)

run_btn = st.sidebar.button("Run grid search")

st.markdown("### How it works")
st.write("""
- **Efficacy success**: stop if posterior probability that \(p > p_0\) exceeds γₑ.
- **Futility**: stop if the **predictive probability** of meeting the posterior success at the **final N** drops below ψ.
- **Safety (optional)**: stop if posterior probability that \(q > q_{max}\) exceeds γₛ.
- **Interims**: equally spaced looks up to K (rounded). All boundaries are computed from Beta‑Binomial conjugacy.
- Designs are simulated to estimate **Type I error (α)**, **power**, and **expected sample size (ESS)**.
""")

if run_btn:
    with st.spinner("Running simulations and evaluating designs..."):
        df = grid_search_designs(
            N_min=N_min, N_max=N_max, K_min=K_min, K_max=K_max,
            p0=p0, p1=p1,
            a_e=a_e, b_e=b_e,
            a_s=a_s, b_s=b_s,
            qmax=qmax if enable_safety else None,
            q1=q1_val if enable_safety else None,
            gamma_e=gamma_e, psi_fut=psi_fut, gamma_s=gamma_s if enable_safety else None,
            n_sim=int(n_sim), seed=int(seed),
            alpha_target=alpha_target, power_target=power_target,
            N_budget=N_budget if N_budget > 0 else None
        )
    st.success("Grid search complete.")

    st.markdown("### Summary of Candidate Designs")
    st.dataframe(
        df[[
            "selection","is_feasible","N","K_interims","looks",
            "alpha","power","ESS_p0","ESS_p1","avg_looks_p1",
            "safety_stop_q1","safety_stop_qmax"
        ]].style.format({
            "alpha": "{:.3f}", "power": "{:.3f}", "ESS_p0": "{:.1f}", "ESS_p1": "{:.1f}",
            "avg_looks_p1": "{:.2f}",
            "safety_stop_q1": "{:.3f}", "safety_stop_qmax": "{:.3f}"
        }),
        use_container_width=True
    )

    # Recommendations sections
    st.markdown("### Recommended Designs")
    rec_small = df[df["selection"].str.contains("smallest_N", na=False)]
    rec_high = df[df["selection"].str.contains("high_power", na=False)]
    rec_sweet = df[df["selection"].str.contains("sweet_spot", na=False)]

    cols = st.columns(3)
    with cols[0]:
        st.subheader("Smallest N (meeting α & power)")
        if rec_small.empty:
            st.info("No feasible designs meeting both α and power targets in the grid.")
        else:
            st.dataframe(rec_small[["N","K_interims","looks","alpha","power","ESS_p1"]].style.format({"alpha":"{:.3f}","power":"{:.3f}","ESS_p1":"{:.1f}"}))

    with cols[1]:
        st.subheader("High power (≤ N budget)")
        if rec_high.empty:
            st.info("No designs under the N budget in the grid.")
        else:
            st.dataframe(rec_high[["N","K_interims","looks","alpha","power","ESS_p1"]].style.format({"alpha":"{:.3f}","power":"{:.3f}","ESS_p1":"{:.1f}"}))

    with cols[2]:
        st.subheader("Sweet spot (min ESS₍p₁₎)")
        if rec_sweet.empty:
            st.info("No feasible designs found to minimize ESS under p₁.")
        else:
            st.dataframe(rec_sweet[["N","K_interims","looks","alpha","power","ESS_p1"]].style.format({"alpha":"{:.3f}","power":"{:.3f}","ESS_p1":"{:.1f}"}))

    # Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download full results (CSV)", data=csv, file_name="bayes_single_arm_designs.csv", mime="text/csv")

    # Quick visualization: Power vs N (feasible highlight)
    st.markdown("### Power vs N")
    import altair as alt
    chart_df = df.copy()
    chart_df["feasible_flag"] = chart_df["is_feasible"].map({True:"Feasible", False:"Not feasible"})
    power_chart = alt.Chart(chart_df).mark_circle(size=80).encode(
        x=alt.X("N:Q", title="Max N"),
        y=alt.Y("power:Q", title="Power"),
        color=alt.Color("feasible_flag:N", title="Feasibility"),
        tooltip=["N","K_interims","alpha","power","ESS_p1","looks"]
    ).properties(width=800, height=350)
    st.altair_chart(power_chart, use_container_width=True)

# Footer guidance
st.divider()
st.markdown("""
**Tips & Notes**
- Consider **Jeffreys priors** \\(\\text{Beta}(0.5, 0.5)\\) for robustness, or **Beta(1,1)** if you prefer non-informative.
- Typical choices: γₑ ∈ [0.90, 0.99], ψ ∈ [0.05, 0.20], γₛ ∈ [0.80, 0.95].
- If safety is enabled, the safety rule is checked **before** efficacy rules at each look.
- The app reports **Type I error** under \\(p=p_0\\) and **Power** under \\(p=p_1\\), both with toxicity at \\(q=q_1\\).
- For more nuanced futility, you can switch to **PPoS at interims** or use a **posterior futility** criterion \\(\\Pr(p > p_1)\\) if desired.
""")


