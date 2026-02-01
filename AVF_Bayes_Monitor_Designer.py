# streamlit_app.py
# Bayesian Single-Arm Trial Design & Simulation Workbench (Streamlit)
# Author: M365 Copilot for Phil Price
# Date: 2026-02-01
#
# This app helps design and evaluate a single-arm Bayesian monitored trial with
# separate safety, efficacy, and futility schedules. It implements:
#   1) Quick scan across a range of sample sizes (N) for Type I error & power
#   2) Targeted comparison of specific N values
#   3) Deep dive on a specific N (large sims)
#   4) OC matrix across grids of true efficacy and toxicity rates
#
# Key modeling conventions (defaults can be modified in the UI):
#   - Efficacy and safety modeled as independent Bernoulli endpoints
#   - Beta priors: efficacy ~ Beta(alpha_e, beta_e), toxicity ~ Beta(alpha_s, beta_s)
#   - Efficacy rule (interim & final): posterior Pr(p_eff > p0) > threshold
#   - Safety rule: posterior Pr(p_tox > p_tox_star) > threshold -> stop for safety
#   - Futility rule: predictive probability of meeting the FINAL efficacy rule at N
#                    is below futility threshold -> stop for futility
#   - Safety checks take precedence, then efficacy success, then futility
#
# Notes on computation:
#   - Posterior tail probability Pr(p > x | Beta(a,b)) uses SciPy's betainc if available; 
#     otherwise uses Monte Carlo approximation via numpy.random.beta. This can be slower; 
#     consider installing SciPy for best performance and reproducibility.
#   - Predictive probability uses the exact Beta-Binomial predictive mass function.
#
# How to run locally:
#   1) pip install streamlit numpy pandas matplotlib
#      (Optional but recommended): pip install scipy
#   2) streamlit run streamlit_app.py

import io
import json
import math
from math import lgamma
import zipfile
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import streamlit as st
except Exception:
    raise SystemExit("This script is a Streamlit app. Please run with: streamlit run streamlit_app.py")

# Optional SciPy for fast and exact Beta CDF
tRY_SCIPY = True
try:
    if tRY_SCIPY:
        from scipy.special import betainc  # regularized incomplete beta I_x(a,b)
        SCIPY_AVAILABLE = True
    else:
        SCIPY_AVAILABLE = False
except Exception:
    SCIPY_AVAILABLE = False


# ----------------------------- Utility Functions ----------------------------- #

def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Return I_x(a,b), the regularized incomplete beta function.
    Requires SciPy. If not available, raises RuntimeError.
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is not available for exact Beta CDF.")
    # betainc(a, b, x) already returns the regularized incomplete beta
    return betainc(a, b, x)


def posterior_tail_prob_beta(a: float, b: float, threshold: float, rng: np.random.Generator | None = None,
                             mc_samples: int = 5000) -> float:
    """Compute Pr(p > threshold | p ~ Beta(a,b)).
    Prefer exact computation via SciPy if available; otherwise Monte Carlo.
    """
    threshold = float(np.clip(threshold, 0.0, 1.0))
    if a <= 0 or b <= 0:
        return float('nan')
    if SCIPY_AVAILABLE:
        try:
            cdf_val = _regularized_incomplete_beta(a, b, threshold)
            return float(max(0.0, min(1.0, 1.0 - cdf_val)))
        except Exception:
            pass
    # Fallback MC approximation
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.beta(a, b, size=mc_samples)
    return float(np.mean(samples > threshold))


def log_beta(a: float, b: float) -> float:
    return lgamma(a) + lgamma(b) - lgamma(a + b)


def beta_binomial_pmf(k: int, n: int, alpha: float, beta: float) -> float:
    """Predictive pmf: P(K=k | future n, prior Beta(alpha,beta)).
    Using: C(n,k) * B(alpha+k, beta+n-k) / B(alpha, beta)
    """
    if k < 0 or k > n:
        return 0.0
    log_coef = lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    val = math.exp(log_coef + log_beta(alpha + k, beta + n - k) - log_beta(alpha, beta))
    return float(val)


def predictive_prob_success_at_final(r_eff: int, n_curr: int, N_final: int,
                                     alpha_e: float, beta_e: float,
                                     p0: float, eff_final_prob: float,
                                     rng: np.random.Generator | None = None,
                                     mc_samples_beta: int = 5000) -> float:
    """Compute the predictive probability, given current r_eff and n_curr, that the
    FINAL efficacy criterion (posterior Pr(p_eff > p0) > eff_final_prob at N_final)
    will be met after enrolling the remaining patients.
    Uses exact Beta-Binomial predictive distribution. Each state check uses
    posterior tail prob via SciPy if available, else MC Beta sampling.
    """
    n_remain = N_final - n_curr
    if n_remain < 0:
        return 0.0
    total = 0.0
    for k in range(n_remain + 1):
        # future successes
        r_tot = r_eff + k
        a_post = alpha_e + r_tot
        b_post = beta_e + N_final - r_tot
        tail = posterior_tail_prob_beta(a_post, b_post, p0, rng=rng, mc_samples=mc_samples_beta)
        if tail > eff_final_prob:
            w = beta_binomial_pmf(k, n_remain, alpha_e + r_eff, beta_e + (n_curr - r_eff))
            total += w
    return float(max(0.0, min(1.0, total)))


def build_look_schedule(N: int, run_in: int, interval: int) -> list[int]:
    """Return a sorted, unique list of look sizes, including N as final look.
    If run_in <= 0, start at interval; if interval <= 0, only the final look.
    """
    looks = set()
    if interval <= 0 and run_in <= 0:
        looks.add(N)
        return sorted(looks)
    start = max(1, run_in) if run_in and run_in > 0 else max(1, interval)
    x = start
    while x < N:
        looks.add(int(x))
        if interval <= 0:
            break
        x += interval
    looks.add(N)
    return sorted(looks)


# ----------------------------- Simulation Engine ----------------------------- #

def simulate_single_trial(N: int,
                          eff_looks: list[int], saf_looks: list[int], fut_looks: list[int],
                          alpha_e: float, beta_e: float,
                          alpha_s: float, beta_s: float,
                          p0: float, p1: float,
                          p_tox_star: float,
                          eff_prob_interim: float, eff_prob_final: float,
                          saf_prob_cut: float, fut_prob_cut: float,
                          true_p_eff: float, true_p_tox: float,
                          rng: np.random.Generator,
                          mc_samples_beta: int = 2000) -> dict:
    """Simulate one trial path under given true rates.
    Returns dict with outcome details.
    Stopping order priority: safety -> efficacy success -> futility.
    """
    eff_looks_set = set(eff_looks)
    saf_looks_set = set(saf_looks)
    fut_looks_set = set(fut_looks)

    # Pre-generate outcomes for speed
    eff_outcomes = rng.binomial(1, true_p_eff, size=N)
    tox_outcomes = rng.binomial(1, true_p_tox, size=N)

    r_eff = 0
    r_tox = 0
    stop_reason = None
    declared_success = False
    n_enrolled = 0

    for t in range(1, N + 1):
        r_eff += int(eff_outcomes[t - 1])
        r_tox += int(tox_outcomes[t - 1])
        n_enrolled = t

        # 1) Safety
        if t in saf_looks_set:
            a_s = alpha_s + r_tox
            b_s = beta_s + t - r_tox
            prob_tox_high = posterior_tail_prob_beta(a_s, b_s, p_tox_star, rng=rng, mc_samples=mc_samples_beta)
            if prob_tox_high > saf_prob_cut:
                stop_reason = "safety"
                declared_success = False
                break

        # 2) Efficacy (interim and final)
        if t in eff_looks_set:
            a_e = alpha_e + r_eff
            b_e = beta_e + t - r_eff
            prob_eff_above_p0 = posterior_tail_prob_beta(a_e, b_e, p0, rng=rng, mc_samples=mc_samples_beta)
            if t < N:
                if prob_eff_above_p0 > eff_prob_interim:
                    stop_reason = "efficacy_early"
                    declared_success = True
                    break
            else:
                if prob_eff_above_p0 > eff_prob_final:
                    stop_reason = "efficacy_final"
                    declared_success = True
                else:
                    stop_reason = "no_effect"
                break

        # 3) Futility (only prior to final)
        if t in fut_looks_set and t < N:
            pp = predictive_prob_success_at_final(
                r_eff=r_eff, n_curr=t, N_final=N,
                alpha_e=alpha_e, beta_e=beta_e,
                p0=p0, eff_final_prob=eff_prob_final,
                rng=rng, mc_samples_beta=mc_samples_beta,
            )
            if pp < fut_prob_cut:
                stop_reason = "futility"
                declared_success = False
                break

    if n_enrolled == N and stop_reason is None:
        # If we exit loop without firing any rule at N (e.g., no efficacy looks configured),
        # apply final decision.
        a_e = alpha_e + r_eff
        b_e = beta_e + N - r_eff
        prob_eff_above_p0 = posterior_tail_prob_beta(a_e, b_e, p0, rng=rng, mc_samples=mc_samples_beta)
        if prob_eff_above_p0 > eff_prob_final:
            stop_reason = "efficacy_final"
            declared_success = True
        else:
            stop_reason = "no_effect"

    return {
        "n": N,
        "n_enrolled": n_enrolled,
        "r_eff": r_eff,
        "r_tox": r_tox,
        "stop_reason": stop_reason,
        "success": declared_success,
    }


def simulate_many_trials(num_sims: int, seed: int | None,
                         **kwargs) -> pd.DataFrame:
    """Run many trial simulations; return a DataFrame of results.
    kwargs are passed to simulate_single_trial.
    """
    rng = np.random.default_rng(seed)
    rows = []
    # Lighter MC for posterior tail when running many sims quickly
    mc_samples_beta = kwargs.pop("mc_samples_beta", 1500)
    for _ in range(num_sims):
        res = simulate_single_trial(rng=rng, mc_samples_beta=mc_samples_beta, **kwargs)
        rows.append(res)
    return pd.DataFrame(rows)


def summarize_simulation(df: pd.DataFrame) -> dict:
    """Compute key metrics from simulation results DataFrame."""
    N = int(df["n"].iloc[0]) if not df.empty else None
    type1 = None
    power = None
    # The caller should compute type I or power by running under the corresponding truths
    prob_success = float(df["success"].mean()) if not df.empty else 0.0
    ess = float(df["n_enrolled"].mean()) if not df.empty else 0.0
    stop_counts = df["stop_reason"].value_counts(dropna=False)
    stop_probs = {k: float(v) / len(df) for k, v in stop_counts.items()}
    return {
        "N": N,
        "prob_success": prob_success,
        "ESS": ess,
        "stop_probs": stop_probs,
    }


# ----------------------------- Streamlit UI ----------------------------- #

st.set_page_config(page_title="Bayesian Single-Arm Trial Designer", layout="wide")

st.title("Bayesian Single-Arm Trial Designer & Simulator")

st.caption(
    "A lightweight Streamlit tool to design and evaluate single-arm Bayesian monitored trials with separate schedules for efficacy, safety, and futility. "
    "Defaults follow common conventions in the Phase II literature (Beta-Binomial posterior probability rules and predictive futility)."
)

with st.expander("⚙️ Performance Tip", expanded=False):
    st.markdown(
        "- This app is optimized for fast simulations, but exact Beta CDFs require **SciPy**.\n"
        "- If SciPy is not installed, the app uses Monte Carlo approximation for Beta tail probabilities, which is a bit slower.\n"
        "- You can install SciPy with: `pip install scipy`."
    )

# Sidebar: Global design settings
st.sidebar.header("Design Inputs")

# Basic rates
p0 = st.sidebar.number_input(
    "Null response rate (p0)", min_value=0.0, max_value=1.0, value=0.20, step=0.01,
    help="The minimal efficacy rate under the null hypothesis."
)
p1 = st.sidebar.number_input(
    "Target response rate (p1)", min_value=0.0, max_value=1.0, value=0.40, step=0.01,
    help="A clinically meaningful response rate under the alternative hypothesis."
)

# Priors
st.sidebar.subheader("Priors (Beta)")
colA, colB = st.sidebar.columns(2)
alpha_e = colA.number_input("Efficacy α", min_value=0.01, value=1.0, step=0.1,
                            help="Alpha parameter for Beta prior on efficacy.")
beta_e = colB.number_input("Efficacy β", min_value=0.01, value=1.0, step=0.1,
                           help="Beta parameter for Beta prior on efficacy.")
colC, colD = st.sidebar.columns(2)
alpha_s = colC.number_input("Safety α", min_value=0.01, value=1.0, step=0.1,
                            help="Alpha parameter for Beta prior on toxicity.")
beta_s = colD.number_input("Safety β", min_value=0.01, value=1.0, step=0.1,
                           help="Beta parameter for Beta prior on toxicity.")

# Schedules
st.sidebar.subheader("Look Schedules")

st.sidebar.markdown("**Efficacy**")
colE1, colE2 = st.sidebar.columns(2)
run_in_eff = colE1.number_input("Run-in (first look)", min_value=0, value=10, step=1,
                                help="Number of patients at the first efficacy look. If 0, the first look is at the interval.")
interval_eff = colE2.number_input("Interval", min_value=0, value=10, step=1,
                                  help="Patients between efficacy looks. If 0, only final look.")

st.sidebar.markdown("**Safety**")
colS1, colS2 = st.sidebar.columns(2)
run_in_saf = colS1.number_input("Run-in (first look)", min_value=0, value=10, step=1,
                                help="Number of patients at the first safety look. If 0, the first look is at the interval.")
interval_saf = colS2.number_input("Interval", min_value=0, value=5, step=1,
                                  help="Patients between safety looks. If 0, only final look.")

st.sidebar.markdown("**Futility**")
colF1, colF2 = st.sidebar.columns(2)
run_in_fut = colF1.number_input("Run-in (first look)", min_value=0, value=10, step=1,
                                help="Number of patients at the first futility look. If 0, the first look is at the interval.")
interval_fut = colF2.number_input("Interval", min_value=0, value=10, step=1,
                                  help="Patients between futility looks. If 0, only final look.")

# Thresholds
st.sidebar.subheader("Decision Thresholds")
colT1, colT2 = st.sidebar.columns(2)
eff_prob_interim = colT1.number_input(
    "Efficacy interim prob. (Pr[p>p0])", min_value=0.5, max_value=1.0, value=0.99, step=0.01,
    help="Stop early for efficacy if posterior Pr(p_eff > p0) exceeds this level at an interim look."
)
eff_prob_final = colT2.number_input(
    "Efficacy final prob. (Pr[p>p0])", min_value=0.5, max_value=1.0, value=0.975, step=0.005,
    help="Declare success at the final look if posterior Pr(p_eff > p0) exceeds this level."
)

colT3, colT4 = st.sidebar.columns(2)
p_tox_star = colT3.number_input(
    "Max acceptable tox rate (p_tox*)", min_value=0.0, max_value=1.0, value=0.30, step=0.01,
    help="If posterior Pr(p_tox > p_tox*) exceeds the safety cutoff at a safety look, stop for safety."
)
saf_prob_cut = colT4.number_input(
    "Safety cutoff prob.", min_value=0.5, max_value=1.0, value=0.90, step=0.01,
    help="Stop for safety if posterior Pr(p_tox > p_tox*) exceeds this level at a safety look."
)

fut_prob_cut = st.sidebar.number_input(
    "Futility cutoff (predictive Pr[final criterion])", min_value=0.0, max_value=0.5, value=0.05, step=0.01,
    help="Stop for futility if predictive probability of meeting the FINAL efficacy criterion at N is below this threshold."
)

# Error constraints
st.sidebar.subheader("Operating Characteristics Targets")
colO1, colO2, colO3 = st.sidebar.columns(3)
max_type1 = colO1.number_input("Max Type I error", min_value=0.01, max_value=0.5, value=0.10, step=0.01,
                               help="Target upper bound for Type I error under p_eff = p0 (tox at acceptable level).")
min_power = colO2.number_input("Min Power", min_value=0.5, max_value=1.0, value=0.80, step=0.01,
                               help="Target lower bound for power under p_eff = p1 (tox at acceptable level).")
flex_tol = colO3.number_input("Flex tolerance (±)", min_value=0.0, max_value=0.2, value=0.02, step=0.01,
                              help="Accept designs slightly outside targets within this tolerance.")

# Simulation settings
st.sidebar.subheader("Simulation Settings")
colSim1, colSim2 = st.sidebar.columns(2)
rand_seed = int(colSim1.number_input("Random seed", min_value=0, value=12345, step=1,
                                     help="Seed for reproducibility. Each run reuses this seed."))
def_mc_beta = int(colSim2.number_input("MC samples (Beta tail, fallback only)", min_value=500, value=3000, step=500,
                                       help="Used only if SciPy is unavailable. Higher is more accurate but slower."))

# Tabs for four main functions

main_tabs = st.tabs([
    "1) Quick Scan (N range)",
    "2) Compare Specific Ns",
    "3) Deep Dive (single N)",
    "4) OC Matrix (grid)",
    "Selected Design & Export",
])


def current_schedules(N: int):
    eff_looks = build_look_schedule(N, run_in_eff, interval_eff)
    saf_looks = build_look_schedule(N, run_in_saf, interval_saf)
    fut_looks = build_look_schedule(N, run_in_fut, interval_fut)
    return eff_looks, saf_looks, fut_looks


def eval_design_for_truth(N: int, true_p_eff: float, true_p_tox: float, sims: int, seed: int):
    eff_looks, saf_looks, fut_looks = current_schedules(N)
    df = simulate_many_trials(
        num_sims=sims, seed=seed,
        N=N,
        eff_looks=eff_looks, saf_looks=saf_looks, fut_looks=fut_looks,
        alpha_e=alpha_e, beta_e=beta_e,
        alpha_s=alpha_s, beta_s=beta_s,
        p0=p0, p1=p1,
        p_tox_star=p_tox_star,
        eff_prob_interim=eff_prob_interim, eff_prob_final=eff_prob_final,
        saf_prob_cut=saf_prob_cut, fut_prob_cut=fut_prob_cut,
        true_p_eff=true_p_eff, true_p_tox=true_p_tox,
        mc_samples_beta=def_mc_beta,
    )
    return df


with main_tabs[0]:
    st.subheader("1) Quick Scan Across N")
    st.markdown(
        "Evaluate Type I error and power quickly across a range of maximum sample sizes (N).\n"
        "Uses moderate simulation counts for speed."
    )
    colQS1, colQS2, colQS3 = st.columns(3)
    N_min = int(colQS1.number_input("N min", min_value=5, value=40, step=1,
                                    help="Lower bound of N to scan."))
    N_max = int(colQS2.number_input("N max", min_value=N_min, value=80, step=1,
                                    help="Upper bound of N to scan."))
    N_step = int(colQS3.number_input("Step", min_value=1, value=5, step=1,
                                     help="Step size for N between min and max."))

    colQS4, colQS5 = st.columns(2)
    sims_fast = int(colQS4.number_input("Sims per N (fast)", min_value=500, value=3000, step=500,
                                        help="Number of simulations per N for quick scan."))
    ptox_for_eval = float(colQS5.number_input("True toxicity for OC eval", min_value=0.0, max_value=1.0, value=max(0.01, min(p_tox_star - 0.05, 0.99)), step=0.01,
                                             help="Toxicity rate used when computing Type I error and Power (assumed acceptable)."))

    run_btn = st.button("Run Quick Scan", type="primary")

    if run_btn:
        results = []
        Ns = list(range(N_min, N_max + 1, N_step))
        prog = st.progress(0.0)
        for i, N in enumerate(Ns, start=1):
            # Type I: true efficacy = p0
            df_h0 = eval_design_for_truth(N=N, true_p_eff=p0, true_p_tox=ptox_for_eval, sims=sims_fast, seed=rand_seed + 11*N)
            # Power: true efficacy = p1
            df_h1 = eval_design_for_truth(N=N, true_p_eff=p1, true_p_tox=ptox_for_eval, sims=sims_fast, seed=rand_seed + 13*N)
            type1 = float(df_h0["success"].mean())
            power = float(df_h1["success"].mean())
            ess_h0 = float(df_h0["n_enrolled"].mean())
            ess_h1 = float(df_h1["n_enrolled"].mean())
            results.append({
                "N": N,
                "Type I": type1,
                "Power": power,
                "ESS@H0": ess_h0,
                "ESS@H1": ess_h1,
            })
            prog.progress(i/len(Ns))
        scan_df = pd.DataFrame(results)
        st.session_state["latest_scan_df"] = scan_df
        st.dataframe(scan_df.style.format({"Type I": "{:.3f}", "Power": "{:.3f}", "ESS@H0": "{:.1f}", "ESS@H1": "{:.1f}"}), use_container_width=True)
        # Plot
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(scan_df["N"], scan_df["Type I"], marker='o', label='Type I')
        ax.plot(scan_df["N"], scan_df["Power"], marker='o', label='Power')
        ax.axhline(max_type1, color='r', linestyle='--', alpha=0.5, label='Max Type I target')
        ax.axhline(min_power, color='g', linestyle='--', alpha=0.5, label='Min Power target')
        ax.set_xlabel("N (final sample size)")
        ax.set_ylabel("Probability")
        ax.set_title("Type I & Power vs N (Quick Scan)")
        ax.legend()
        st.pyplot(fig)

        # Highlight feasible Ns
        feas = scan_df[(scan_df["Type I"] <= max_type1 + flex_tol) & (scan_df["Power"] >= min_power - flex_tol)]
        st.markdown("**Feasible designs (within tolerance):**")
        st.dataframe(feas, use_container_width=True)


with main_tabs[1]:
    st.subheader("2) Compare Specific Ns")
    st.markdown("Enter a list of N values to evaluate with a bit more precision.")
    Ns_text = st.text_input("N values (comma-separated)", value="40, 50, 60, 70, 80",
                            help="Example: 36, 48, 60")
    sims_cmp = int(st.number_input("Sims per N (balanced)", min_value=1000, value=5000, step=1000,
                                   help="Number of simulations per N for targeted comparison."))
    ptox_for_eval2 = float(st.number_input("True toxicity for OC eval", min_value=0.0, max_value=1.0, value=max(0.01, min(p_tox_star - 0.05, 0.99)), step=0.01))
    cmp_btn = st.button("Run Comparison")

    if cmp_btn:
        Ns_list = [int(x.strip()) for x in Ns_text.split(',') if x.strip().isdigit()]
        rows = []
        prog = st.progress(0.0)
        for i, N in enumerate(Ns_list, start=1):
            df_h0 = eval_design_for_truth(N=N, true_p_eff=p0, true_p_tox=ptox_for_eval2, sims=sims_cmp, seed=rand_seed + 101*N)
            df_h1 = eval_design_for_truth(N=N, true_p_eff=p1, true_p_tox=ptox_for_eval2, sims=sims_cmp, seed=rand_seed + 103*N)
            rows.append({
                "N": N,
                "Type I": float(df_h0["success"].mean()),
                "Power": float(df_h1["success"].mean()),
                "ESS@H0": float(df_h0["n_enrolled"].mean()),
                "ESS@H1": float(df_h1["n_enrolled"].mean()),
            })
            prog.progress(i/len(Ns_list))
        cmp_df = pd.DataFrame(rows).sort_values("N")
        st.session_state["latest_compare_df"] = cmp_df
        st.dataframe(cmp_df.style.format({"Type I": "{:.3f}", "Power": "{:.3f}", "ESS@H0": "{:.1f}", "ESS@H1": "{:.1f}"}), use_container_width=True)
        # Bar plot
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.plot(cmp_df["N"], cmp_df["Type I"], marker='o', label='Type I')
        ax2.plot(cmp_df["N"], cmp_df["Power"], marker='o', label='Power')
        ax2.axhline(max_type1, color='r', linestyle='--', alpha=0.5, label='Max Type I target')
        ax2.axhline(min_power, color='g', linestyle='--', alpha=0.5, label='Min Power target')
        ax2.set_xlabel("N")
        ax2.set_ylabel("Probability")
        ax2.set_title("Type I & Power vs N (Comparison)")
        ax2.legend()
        st.pyplot(fig2)


with main_tabs[2]:
    st.subheader("3) Deep Dive on a Specific N")
    N_deep = int(st.number_input("Select N (final)", min_value=5, value=60, step=1))
    sims_deep = int(st.number_input("Sims (deep dive)", min_value=5000, value=50000, step=5000,
                                    help="Use a high number for stable estimates."))
    ptox_deep = float(st.number_input("True toxicity for deep dive", min_value=0.0, max_value=1.0, value=max(0.01, min(p_tox_star - 0.05, 0.99)), step=0.01))
    deep_btn = st.button("Run Deep Dive")

    if deep_btn:
        df_h0 = eval_design_for_truth(N=N_deep, true_p_eff=p0, true_p_tox=ptox_deep, sims=sims_deep, seed=rand_seed + 1001)
        df_h1 = eval_design_for_truth(N=N_deep, true_p_eff=p1, true_p_tox=ptox_deep, sims=sims_deep, seed=rand_seed + 1003)

        type1 = float(df_h0["success"].mean())
        power = float(df_h1["success"].mean())
        ess_h0 = float(df_h0["n_enrolled"].mean())
        ess_h1 = float(df_h1["n_enrolled"].mean())
        st.markdown(f"**Type I** (p={p0:.2f}) = **{type1:.3f}**  |  **Power** (p={p1:.2f}) = **{power:.3f}**")
        st.markdown(f"**ESS@H0** = {ess_h0:.1f} | **ESS@H1** = {ess_h1:.1f}")

        # Stop reason breakdown plots
        def plot_stop_reasons(df: pd.DataFrame, title: str):
            counts = df["stop_reason"].value_counts()
            fig, ax = plt.subplots(figsize=(6,4))
            counts.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            ax.set_title(title)
            ax.set_ylabel("Count")
            return fig

        fig_h0 = plot_stop_reasons(df_h0, "Stop reasons @ H0")
        fig_h1 = plot_stop_reasons(df_h1, "Stop reasons @ H1")
        st.pyplot(fig_h0)
        st.pyplot(fig_h1)

        # Distribution of sample size
        fig_ss, ax_ss = plt.subplots(figsize=(6,4))
        ax_ss.hist(df_h0["n_enrolled"], bins=20, alpha=0.6, label='H0')
        ax_ss.hist(df_h1["n_enrolled"], bins=20, alpha=0.6, label='H1')
        ax_ss.set_title("Distribution of enrolled sample size")
        ax_ss.set_xlabel("n enrolled")
        ax_ss.set_ylabel("Frequency")
        ax_ss.legend()
        st.pyplot(fig_ss)

        # Save to session for export
        st.session_state["deep_dive_h0_df"] = df_h0
        st.session_state["deep_dive_h1_df"] = df_h1


with main_tabs[3]:
    st.subheader("4) Operating Characteristics (OC) Matrix")
    st.markdown("Compute trial performance over grids of true efficacy and toxicity rates.")
    colOC1, colOC2 = st.columns(2)
    p_eff_min = float(colOC1.number_input("Efficacy min", min_value=0.0, max_value=1.0, value=max(0.0, p0 - 0.1), step=0.01))
    p_eff_max = float(colOC1.number_input("Efficacy max", min_value=p_eff_min, max_value=1.0, value=min(1.0, p1 + 0.2), step=0.01))
    p_eff_step = float(colOC1.number_input("Efficacy step", min_value=0.01, max_value=0.5, value=0.05, step=0.01))

    p_tox_min = float(colOC2.number_input("Toxicity min", min_value=0.0, max_value=1.0, value=max(0.0, p_tox_star - 0.2), step=0.01))
    p_tox_max = float(colOC2.number_input("Toxicity max", min_value=p_tox_min, max_value=1.0, value=min(1.0, p_tox_star + 0.2), step=0.01))
    p_tox_step = float(colOC2.number_input("Toxicity step", min_value=0.01, max_value=0.5, value=0.05, step=0.01))

    N_oc = int(st.number_input("Select N for OC matrix", min_value=5, value=60, step=1))
    sims_oc = int(st.number_input("Sims per cell", min_value=500, value=2000, step=500))
    run_oc_btn = st.button("Run OC Matrix")

    if run_oc_btn:
        eff_vals = np.round(np.arange(p_eff_min, p_eff_max + 1e-9, p_eff_step), 3)
        tox_vals = np.round(np.arange(p_tox_min, p_tox_max + 1e-9, p_tox_step), 3)
        records = []
        prog = st.progress(0.0)
        total_cells = len(eff_vals) * len(tox_vals)
        cell_idx = 0
        for pe in eff_vals:
            for pt in tox_vals:
                df_cell = eval_design_for_truth(N=N_oc, true_p_eff=float(pe), true_p_tox=float(pt), sims=sims_oc, seed=rand_seed + 2001 + cell_idx)
                records.append({
                    "p_eff": float(pe),
                    "p_tox": float(pt),
                    "Pr(Success)": float(df_cell["success"].mean()),
                    "ESS": float(df_cell["n_enrolled"].mean()),
                    "Pr(SafetyStop)": float((df_cell["stop_reason"] == "safety").mean()),
                    "Pr(Futility)": float((df_cell["stop_reason"] == "futility").mean()),
                })
                cell_idx += 1
                prog.progress(cell_idx / total_cells)
        oc_df = pd.DataFrame(records)
        st.session_state["oc_matrix_df"] = oc_df

        st.markdown("**OC Matrix Table**")
        st.dataframe(oc_df.head(1000), use_container_width=True)

        # Heatmaps
        def plot_heatmap(pivot_df: pd.DataFrame, title: str, cmap: str = 'viridis'):
            fig, ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(pivot_df.values, origin='lower', aspect='auto', cmap=cmap,
                           extent=[pivot_df.columns.min(), pivot_df.columns.max(), pivot_df.index.min(), pivot_df.index.max()])
            ax.set_xlabel("p_tox")
            ax.set_ylabel("p_eff")
            ax.set_title(title)
            cbar = fig.colorbar(im, ax=ax)
            return fig

        piv_success = oc_df.pivot(index='p_eff', columns='p_tox', values='Pr(Success)')
        piv_ess = oc_df.pivot(index='p_eff', columns='p_tox', values='ESS')
        piv_safety = oc_df.pivot(index='p_eff', columns='p_tox', values='Pr(SafetyStop)')
        piv_futility = oc_df.pivot(index='p_eff', columns='p_tox', values='Pr(Futility)')

        st.markdown("**Heatmaps**")
        st.pyplot(plot_heatmap(piv_success, "Pr(Success)"))
        st.pyplot(plot_heatmap(piv_ess, "Expected Sample Size", cmap='magma'))
        st.pyplot(plot_heatmap(piv_safety, "Pr(Safety Stop)", cmap='Reds'))
        st.pyplot(plot_heatmap(piv_futility, "Pr(Futility)", cmap='Blues'))


with main_tabs[4]:
    st.subheader("Selected Design & Export")

    st.markdown("Use this panel to finalize your chosen design and export a zip of tables, figures, and a JSON summary.")
    N_sel = int(st.number_input("Selected final N", min_value=5, value=60, step=1))
    save_btn = st.button("Set as Selected Design")

    if save_btn:
        eff_looks, saf_looks, fut_looks = current_schedules(N_sel)
        selection = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "N": N_sel,
            "p0": p0, "p1": p1,
            "priors": {"eff": [alpha_e, beta_e], "saf": [alpha_s, beta_s]},
            "eff_looks": eff_looks, "saf_looks": saf_looks, "fut_looks": fut_looks,
            "thresholds": {
                "eff_interim": eff_prob_interim, "eff_final": eff_prob_final,
                "p_tox_star": p_tox_star, "safety_cut": saf_prob_cut, "futility_cut": fut_prob_cut,
            },
            "oc_targets": {"max_type1": max_type1, "min_power": min_power, "tolerance": flex_tol},
            "sim_settings": {"seed": rand_seed, "mc_beta_samples": def_mc_beta, "scipy": SCIPY_AVAILABLE},
        }
        st.session_state["final_design_selection"] = selection
        st.success("Selected design saved for export.")

    if "final_design_selection" in st.session_state:
        st.markdown("### Final Design Summary")
        st.json(st.session_state["final_design_selection"])

    st.markdown("---")
    st.subheader("Export")

    def _fig_to_png_bytes(fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=160)
        buf.seek(0)
        return buf.read()

    def build_export_zip() -> bytes:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            # README
            readme = (
                "Bayesian Single-Arm Trial Designer Export\n"
                f"Created: {datetime.utcnow().isoformat()}Z\n\n"
                "Contents:\n"
                "- final_design.json: JSON summary of selected design\n"
                "- scan.csv: Quick scan table (if available)\n"
                "- compare.csv: Comparison table (if available)\n"
                "- deep_dive_h0.csv / deep_dive_h1.csv: Deep dive simulation raw outputs (if available)\n"
                "- oc_matrix.csv: OC matrix table (if available)\n"
                "- figures/*.png: Plots\n"
            )
            zf.writestr("README.txt", readme)
            # Final design JSON
            if "final_design_selection" in st.session_state:
                zf.writestr("final_design.json", json.dumps(st.session_state["final_design_selection"], indent=2))
            # Tables
            if "latest_scan_df" in st.session_state:
                zf.writestr("scan.csv", st.session_state["latest_scan_df"].to_csv(index=False))
                # Also add a figure if we can reconstruct
                try:
                    df = st.session_state["latest_scan_df"]
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.plot(df["N"], df["Type I"], marker='o', label='Type I')
                    ax.plot(df["N"], df["Power"], marker='o', label='Power')
                    ax.set_xlabel("N")
                    ax.set_ylabel("Probability")
                    ax.set_title("Quick Scan: Type I & Power vs N")
                    ax.legend()
                    zf.writestr("figures/quick_scan.png", _fig_to_png_bytes(fig))
                    plt.close(fig)
                except Exception:
                    pass
            if "latest_compare_df" in st.session_state:
                zf.writestr("compare.csv", st.session_state["latest_compare_df"].to_csv(index=False))
            if "deep_dive_h0_df" in st.session_state:
                zf.writestr("deep_dive_h0.csv", st.session_state["deep_dive_h0_df"].to_csv(index=False))
            if "deep_dive_h1_df" in st.session_state:
                zf.writestr("deep_dive_h1.csv", st.session_state["deep_dive_h1_df"].to_csv(index=False))
            if "oc_matrix_df" in st.session_state:
                zf.writestr("oc_matrix.csv", st.session_state["oc_matrix_df"].to_csv(index=False))
                # Add heatmap figures
                try:
                    oc_df = st.session_state["oc_matrix_df"]
                    piv_success = oc_df.pivot(index='p_eff', columns='p_tox', values='Pr(Success)')
                    piv_ess = oc_df.pivot(index='p_eff', columns='p_tox', values='ESS')
                    piv_safety = oc_df.pivot(index='p_eff', columns='p_tox', values='Pr(SafetyStop)')
                    piv_futility = oc_df.pivot(index='p_eff', columns='p_tox', values='Pr(Futility)')
                    def heat(pivot_df, title, cmap='viridis'):
                        fig, ax = plt.subplots(figsize=(6,5))
                        im = ax.imshow(pivot_df.values, origin='lower', aspect='auto', cmap=cmap,
                                       extent=[pivot_df.columns.min(), pivot_df.columns.max(), pivot_df.index.min(), pivot_df.index.max()])
                        ax.set_xlabel("p_tox")
                        ax.set_ylabel("p_eff")
                        ax.set_title(title)
                        fig.colorbar(im, ax=ax)
                        return fig
                    zf.writestr("figures/oc_success.png", _fig_to_png_bytes(heat(piv_success, "Pr(Success)")))
                    zf.writestr("figures/oc_ess.png", _fig_to_png_bytes(heat(piv_ess, "Expected Sample Size", cmap='magma')))
                    zf.writestr("figures/oc_safety.png", _fig_to_png_bytes(heat(piv_safety, "Pr(Safety Stop)", cmap='Reds')))
                    zf.writestr("figures/oc_futility.png", _fig_to_png_bytes(heat(piv_futility, "Pr(Futility)", cmap='Blues')))
                    plt.close('all')
                except Exception:
                    pass
        zbuf.seek(0)
        return zbuf.read()

    if st.button("Build export zip"):
        try:
            zip_bytes = build_export_zip()
            st.download_button(
                label="Download Export (.zip)",
                data=zip_bytes,
                file_name="bayesian_single_arm_design_export.zip",
                mime="application/zip",
            )
        except Exception as e:
            st.error(f"Export failed: {e}")


# Footer notes
st.markdown("---")
with st.expander("Methodology & Defaults", expanded=False):
    st.markdown(
        """
        **Model & Rules**

        - **Efficacy** and **Safety** are modeled as Bernoulli endpoints with independent Beta priors.\
          Posteriors are Beta(α + r, β + n − r).
        - **Efficacy decision**: at each efficacy look, compute posterior Pr(p_eff > p0).\
          If above the interim threshold (for non-final looks), stop early for success. At the final look (N), declare success if above the final threshold.
        - **Safety decision**: at safety looks, compute posterior Pr(p_tox > p_tox*). If above the safety cutoff, stop for safety.
        - **Futility decision**: at futility looks, compute the **predictive probability** of meeting the **final** efficacy rule at N.\
          If below futility cutoff, stop for futility.
        - **Stopping priority**: Safety → Efficacy success → Futility.

        **Defaults**

        - Priors: Beta(1,1) for both efficacy and toxicity (uninformative).\
        - Thresholds (typical values used in Phase II practice): interim efficacy 0.99, final efficacy 0.975, safety cutoff 0.90 with p_tox* = 0.30, futility cutoff 0.05.\
        - OC targets: one-sided Type I ≤ 0.10; Power ≥ 0.80 (with ±0.02 tolerance by default).

        **Assumptions & Notes**

        - Efficacy and toxicity are simulated independently per subject (no correlation model).\
          If correlation is important, consider extending the generator to a bivariate Bernoulli model.
        - Exact Beta tail probabilities use SciPy's incomplete beta when available; otherwise Monte Carlo approximation is used.\
          Install SciPy for exact and faster evaluations.
        - Predictive probabilities use the exact Beta–Binomial predictive mass function.
        - Schedules are specified via a run-in and interval; the final look at N is always included.
        - When you change N, look times are recomputed from the run-in and interval settings.

        **Export**

        - The export zip includes a JSON summary of the selected design, tables (CSV), and key figures (PNG) for any analyses you've run in this session.
        """
    )
