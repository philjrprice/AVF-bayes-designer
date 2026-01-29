# app.py
# Streamlit app for single-arm Bayesian monitored design (binary endpoint)
# Rapid screener + deep-dive simulation WITH optional safety monitoring
#
# ✦ This version focuses on:
#   • Plain-language UI labels and tooltips for every control
#   • Rich code comments/docstrings explaining what each block does and why
#   • ZERO changes to the underlying mathematics/logic (only UX and comments)
#
# Author: M365 Copilot for Phil

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple
from scipy.stats import beta
from scipy.special import betaln, comb

# ──────────────────────────────────────────────────────────────────────────────
# Optional plotting: Plotly if available, else fall back to Streamlit charts.
# ──────────────────────────────────────────────────────────────────────────────
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CORE BAYESIAN UTILITIES                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# These functions implement the math for Beta–Binomial Bayesian monitoring,
# which the UI and simulation engine call repeatedly.

def beta_posterior_params(a0: float, b0: float, x: int, n: int) -> Tuple[float, float]:
    """
    Given a Beta(a0, b0) prior for a success probability p, and x successes in n trials,
    return the posterior parameters for p | data ~ Beta(a0 + x, b0 + n - x).
    """
    return a0 + x, b0 + (n - x)


def posterior_prob_p_greater_than(p_cut: float, a_post: float, b_post: float) -> float:
    """
    Compute posterior probability P(p > p_cut | data) under Beta(a_post, b_post).
    """
    return 1.0 - beta.cdf(p_cut, a_post, b_post)


def min_successes_for_posterior_threshold(
    a0: float, b0: float, N: int, p0: float, theta_final: float
) -> Optional[int]:
    """
    For a fixed maximum sample size N and a final rule "declare success if
    P(p > p0 | data) ≥ theta_final", find the SMALLEST total number of successes
    s_min in {0, 1, ..., N} that meets this threshold. (Monotone in s, so use binary search.)
    Returns None if no s can meet the rule (rare unless thresholds are inconsistent).
    """
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
    """
    Log PMF for Y ~ Beta–Binomial(m; a, b):
      P(Y=y) = C(m,y) * B(y+a, m-y+b) / B(a, b)
    Using logs keeps numerical stability when probabilities are tiny.
    """
    if y < 0 or y > m:
        return -np.inf
    return np.log(comb(m, y)) + betaln(y + a, m - y + b) - betaln(a, b)


def beta_binomial_cdf_upper_tail(y_min: int, m: int, a: float, b: float) -> float:
    """
    Compute sum_{y >= y_min} P(Y=y) for Y ~ Beta–Binomial(m; a, b).
    This is the predictive probability that "at least y_min future successes" occur.
    """
    if y_min <= 0:
        return 1.0
    if y_min > m:
        return 0.0
    ys = np.arange(y_min, m + 1)
    logs = np.array([log_beta_binomial_pmf(int(y), m, a, b) for y in ys])
    # log-sum-exp trick for stability
    mlog = np.max(logs)
    return float(np.exp(mlog) * np.sum(np.exp(logs - mlog)))


def predictive_prob_of_final_success(
    a0: float, b0: float, N_total: int, x_curr: int, n_curr: int, p0: float, theta_final: float
) -> float:
    """
    Predictive Probability of Success (PPoS) at an interim:
    Given current data (x_curr successes in n_curr), compute the probability—over FUTURE
    data only—that the study will meet the FINAL success rule at N_total.
    """
    # Posterior after the current data
    a_post, b_post = beta_posterior_params(a0, b0, x_curr, n_curr)
    # Future observations still to collect
    m_remain = N_total - n_curr
    # Final success boundary (in total successes)
    s_min = min_successes_for_posterior_threshold(a0, b0, N_total, p0, theta_final)
    if s_min is None:
        return 0.0
    # How many more successes we need from here to reach s_min overall?
    y_needed = s_min - x_curr
    # Predictive distribution for future successes is Beta–Binomial with posterior parameters
    return beta_binomial_cdf_upper_tail(y_needed, m_remain, a_post, b_post)


def compute_interim_futility_cutoffs(
    a0: float, b0: float, N_total: int, looks: List[int], p0: float, theta_final: float, c_futility: float
) -> Dict[int, Optional[int]]:
    """
    For each interim sample size n in `looks`, compute the minimum CURRENT number
    of successes x_min to CONTINUE (i.e., not stop for futility), such that
    PPoS(final success) ≥ c_futility. If no x can meet this, threshold is None (always stop).
    """
    cutoffs: Dict[int, Optional[int]] = {}
    for n in looks:
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
# ║                           SIMULATION ENGINES                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# We keep two engines:
#   1) simulate_design            – efficacy ONLY (used in "Rapid Screener" for speed)
#   2) simulate_design_with_safety – efficacy + safety (used in "Deep Dive" for accuracy)

def simulate_design(design: Dict, p: float, U: np.ndarray) -> Dict:
    """
    Efficacy-only simulation (fast screener).
    - p: true efficacy rate.
    - U: matrix of uniforms of shape (n_sims, N_total) to generate Bernoulli(p) outcomes.
    Returns reject_rate, ESS, and a sample-size distribution (stop_dist).
    """
    N = design["N_total"]
    looks = design["looks"]
    a0 = design["a0"]; b0 = design["b0"]
    p0 = design["p0"]; theta_final = design["theta_final"]
    theta_interim = design.get("theta_interim", theta_final)
    c_fut = design["c_futility"]
    allow_early = design["allow_early_success"]
    s_min_final = design["s_min_final"]
    x_min_to_continue = design["x_min_to_continue_by_look"]

    n_sims = U.shape[0]
    # Generate Bernoulli(p) from uniforms
    X = (U[:, :N] < p).astype(np.int16)

    # Track cumulative successes, current n, active trials, etc.
    cum_x = np.zeros(n_sims, dtype=np.int32)
    n_curr = 0
    active = np.ones(n_sims, dtype=bool)
    success = np.zeros(n_sims, dtype=bool)
    final_n = np.zeros(n_sims, dtype=np.int32)
    stop_by_look_counts = np.zeros(len(looks), dtype=np.int64)

    for li, look_n in enumerate(looks):
        # Enroll from n_curr up to the interim look
        add = look_n - n_curr
        if add > 0:
            cum_x[active] += np.sum(X[active, n_curr:look_n], axis=1)
            n_curr = look_n

        # (Optional) Early success at interim using θ_interim
        if allow_early and active.any():
            a_post, b_post = beta_posterior_params(a0, b0, int(0), int(0))  # placeholders for typing
            a_post, b_post = beta_posterior_params(a0, b0, cum_x[active], n_curr)
            post_probs = 1.0 - beta.cdf(p0, a_post, b_post)
            early_succ = (post_probs >= theta_interim)
            if np.any(early_succ):
                idx_active = np.where(active)[0]
                idx = idx_active[early_succ]
                success[idx] = True
                final_n[idx] = n_curr
                active[idx] = False
                stop_by_look_counts[li] += early_succ.sum()

        if not active.any():
            break

        # Futility at interim using PPoS threshold -> continue only if x >= x_min
        x_min = x_min_to_continue.get(look_n, None)
        if x_min is None:
            # "never continue" at this look: everyone still active stops here
            idx = np.where(active)[0]
            if idx.size > 0:
                final_n[idx] = n_curr
                active[idx] = False
                stop_by_look_counts[li] += idx.size
        else:
            need_continue = cum_x[active] >= x_min
            idx_all_active = np.where(active)[0]
            idx_stop = idx_all_active[~need_continue]
            if idx_stop.size > 0:
                final_n[idx_stop] = n_curr
                active[idx_stop] = False
                stop_by_look_counts[li] += idx_stop.size

        if not active.any():
            break

    # Final analysis for those still active
    if active.any():
        add = N - n_curr
        if add > 0:
            cum_x[active] += np.sum(X[active, n_curr:N], axis=1)
            n_curr = N
        # Final success uses s_min_final shortcut
        succ_final = (cum_x[active] >= s_min_final)
        idx_active = np.where(active)[0]
        success[idx_active[succ_final]] = True
        final_n[idx_active] = N

    reject_rate = success.mean()
    ess = final_n.mean()

    # Empirical distribution of total sample size at stop
    unique_ns, counts = np.unique(final_n, return_counts=True)
    stop_dist = pd.DataFrame({"N_stop": unique_ns, "Probability": counts / n_sims}).sort_values("N_stop")

    return {
        "reject_rate": float(reject_rate),
        "ess": float(ess),
        "stop_probs_by_look": (stop_by_look_counts / n_sims).tolist(),
        "stop_dist": stop_dist,
    }


def simulate_design_with_safety(
    design: Dict, p_eff: float, p_tox: float, U_eff: np.ndarray, U_tox: np.ndarray
) -> Dict:
    """
    Joint efficacy + safety simulation (deep dive).
    Safety rule: stop at any interim/final if P(q > q_max | data) ≥ θ_tox.
    Returns: reject_rate (efficacy success), ess, safety_stop_prob, stop_dist.
    """
    N = design["N_total"]
    looks = design["looks"]
    a0 = design["a0"]; b0 = design["b0"]
    p0 = design["p0"]; theta_final = design["theta_final"]
    theta_interim = design.get("theta_interim", theta_final)
    c_fut = design["c_futility"]
    allow_early = design["allow_early_success"]
    s_min_final = design["s_min_final"]
    x_min_to_continue = design["x_min_to_continue_by_look"]
    tox = design.get("safety", None)

    n_sims = U_eff.shape[0]
    X_eff = (U_eff[:, :N] < p_eff).astype(np.int16)           # efficacy outcomes
    X_tox = (U_tox[:, :N] < p_tox).astype(np.int16) if tox is not None else None

    cum_x = np.zeros(n_sims, dtype=np.int32)                  # cum responders
    cum_t = np.zeros(n_sims, dtype=np.int32) if tox is not None else None
    n_curr = 0
    active = np.ones(n_sims, dtype=bool)                      # trials still in play
    success = np.zeros(n_sims, dtype=bool)                    # efficacy success flag
    final_n = np.zeros(n_sims, dtype=np.int32)
    safety_stop_counts = np.zeros(len(looks) + 1, dtype=np.int64)  # stops due to safety (per look + final)

    for li, look_n in enumerate(looks):
        # Enroll up to this interim look
        add = look_n - n_curr
        if add > 0:
            cum_x[active] += np.sum(X_eff[active, n_curr:look_n], axis=1)
            if tox is not None:
                cum_t[active] += np.sum(X_tox[active, n_curr:look_n], axis=1)
            n_curr = look_n

        # 1) Safety gate FIRST
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
                safety_stop_counts[li] += unsafe.sum()

        if not active.any():
            break

        # 2) (Optional) Early efficacy success using θ_interim
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

        if not active.any():
            break

        # 3) Interim futility via PPoS continue threshold
        x_min = x_min_to_continue.get(look_n, None)
        if x_min is None:
            idx = np.where(active)[0]
            if idx.size > 0:
                final_n[idx] = n_curr
                active[idx] = False
        else:
            need_continue = cum_x[active] >= x_min
            idx_all_active = np.where(active)[0]
            idx_stop = idx_all_active[~need_continue]
            if idx_stop.size > 0:
                final_n[idx_stop] = n_curr
                active[idx_stop] = False

        if not active.any():
            break

    # Final stage for those still active
    if active.any():
        add = N - n_curr
        if add > 0:
            cum_x[active] += np.sum(X_eff[active, n_curr:N], axis=1)
            if tox is not None:
                cum_t[active] += np.sum(X_tox[active, n_curr:N], axis=1)
            n_curr = N

        # Safety check at final
        if tox is not None and active.any():
            a_t = tox["a_t0"] + cum_t[active]
            b_t = tox["b_t0"] + (n_curr - cum_t[active])
            prob_tox_high = 1.0 - beta.cdf(tox["q_max"], a_t, b_t)
            unsafe = (prob_tox_high >= tox["theta_tox"])
            if np.any(unsafe):
                idx_active = np.where(active)[0]
                idx = idx_active[unsafe]
                final_n[idx] = n_curr
                active[idx] = False
                safety_stop_counts[-1] += unsafe.sum()

        # Final efficacy decision
        if active.any():
            succ_final = (cum_x[active] >= s_min_final)
            idx_active = np.where(active)[0]
            success[idx_active[succ_final]] = True
            final_n[idx_active] = N

    reject_rate = success.mean()                         # efficacy success rate under scenario
    ess = final_n.mean()                                 # expected sample size
    safety_stop_prob = safety_stop_counts.sum() / U_eff.shape[0] if tox is not None else 0.0

    # Distribution of total N at stop
    unique_ns, counts = np.unique(final_n, return_counts=True)
    stop_dist = pd.DataFrame({"N_stop": unique_ns, "Probability": counts / U_eff.shape[0]}).sort_values("N_stop")

    return {
        "reject_rate": float(reject_rate),
        "ess": float(ess),
        "safety_stop_prob": float(safety_stop_prob),
        "stop_dist": stop_dist,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         SCREENING (FAST) PIPELINE                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# The screener evaluates many candidate designs quickly using efficacy-only
# simulations and exact Beta–Binomial thresholds. Safety is added in deep dive.

def shortlist_designs(param_grid: List[Dict], n_sims_small: int, seed: int, U: Optional[np.ndarray] = None):
    """
    Build and screen many candidate designs (varying N, look schedules, thresholds).
    Uses efficacy-only simulation for speed. Returns a DataFrame plus the raw designs.
    """
    rng = np.random.default_rng(seed)
    if U is None:
        Nmax = max([g["N_total"] for g in param_grid])
        U = rng.uniform(size=(n_sims_small, Nmax))

    rows = []
    designs_built = []

    for g in param_grid:
        N = g["N_total"]
        looks = g["looks"]
        a0 = g["a0"]; b0 = g["b0"]
        p0 = g["p0"]; p1 = g["p1"]
        theta_final = g["theta_final"]; c_futility = g["c_futility"]
        theta_interim = g.get("theta_interim", theta_final)
        allow_early = g["allow_early_success"]

        s_min = min_successes_for_posterior_threshold(a0, b0, N, p0, theta_final)
        if s_min is None:
            # design cannot ever meet the final rule → skip
            continue

        x_min_to_continue = compute_interim_futility_cutoffs(a0, b0, N, looks, p0, theta_final, c_futility)

        design = dict(
            N_total=N, looks=looks, a0=a0, b0=b0, p0=p0,
            theta_final=theta_final, theta_interim=theta_interim,
            c_futility=c_futility, allow_early_success=allow_early,
            s_min_final=s_min, x_min_to_continue_by_look=x_min_to_continue,
            p1=p1
        )

        res_p0 = simulate_design(design, p0, U[:, :N])
        res_p1 = simulate_design(design, p1, U[:, :N])

        rows.append({
            "N_total": N,
            "looks": looks,
            "theta_final": theta_final,
            "theta_interim": theta_interim,
            "c_futility": c_futility,
            "allow_early_success": allow_early,
            "Type I error @ p0": res_p0["reject_rate"],
            "Power @ p1": res_p1["reject_rate"],
            "ESS @ p0": res_p0["ess"],
            "ESS @ p1": res_p1["ess"],
            "s_min_final": s_min,
            "x_min_to_continue": x_min_to_continue,
            "_design": design
        })
        designs_built.append(design)

    df = pd.DataFrame(rows)
    return df, designs_built


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          PLOTTING CONVENIENCE                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_lines(df: pd.DataFrame, x: str, y: str, title: str):
    """Uniform interface to line plots regardless of Plotly availability."""
    if _HAS_PLOTLY:
        fig = px.line(df, x=x, y=y, title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader(title)
        st.line_chart(df[[x, y]].set_index(x))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         STREAMLIT USER INTERFACE                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="Bayesian Single‑Arm Designer (Binary) – Screener & Simulator",
    layout="wide"
)

st.title("Bayesian Single‑Arm Monitored Study Designer (Binary Endpoint)")
st.caption("Design single‑arm trials with Bayesian interim monitoring for **efficacy** and optional **safety**. Plain‑language UI, rigorous math.")

with st.expander("What this tool does (in simple terms)"):
    st.markdown(
        """
**Goal.** Help you pick a trial design that:  
- Has the **power** you want when the true response rate is good,  
- Keeps **Type I error** modest when it's not, and  
- Stops **early** either for **futility**, **clear success**, or **safety**.

**How it works.**
- We assume a **binary outcome** (responder / non‑responder) and a **Beta prior**.
- **Final success**: we call the trial positive if the posterior probability that the response rate exceeds your **null** rate is high enough.
- **Interim futility**: we compute the **predictive probability** of meeting that final rule and stop if it's too low.
- **Safety** (optional): we monitor an SAE rate using its own Beta prior and stop if the posterior chance of exceeding your maximum acceptable rate is too high.
        """
    )

# ── Sidebar: inputs ─────────────────────────────────────────────────────────

st.sidebar.header("1) Set your design targets")
col_sb1, col_sb2 = st.sidebar.columns(2)

with col_sb1:
    p0 = st.number_input(
        "Null response rate (p₀)",
        min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f",
        help="The highest response rate you'd still consider clinically unacceptable."
    )
    a0 = st.number_input(
        "Prior a₀ (efficacy)",
        min_value=0.0, value=1.0, step=0.5,
        help="Efficacy prior Beta(a₀, b₀). Use (1,1) for uninformative. Increase a₀+b₀ for stronger prior."
    )
    theta_final = st.number_input(
        "Final success threshold (θ_final)",
        min_value=0.5, max_value=0.999, value=0.95, step=0.01, format="%.3f",
        help="At the final look, declare success if P(p > p₀ | data) ≥ θ_final."
    )

with col_sb2:
    p1 = st.number_input(
        "Target response rate (p₁)",
        min_value=0.0, max_value=1.0, value=0.40, step=0.01, format="%.2f",
        help="A response rate you consider clinically promising; power is evaluated here."
    )
    b0 = st.number_input(
        "Prior b₀ (efficacy)",
        min_value=0.0, value=1.0, step=0.5,
        help="Efficacy prior Beta(a₀, b₀). Mean=a₀/(a₀+b₀); ESS=a₀+b₀."
    )
    c_futility = st.number_input(
        "Futility cutoff (PPoS)",
        min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.3f",
        help="At interims, stop if the predictive probability of final success is below this."
    )

# Display implied prior mean/ESS to guide users
with st.sidebar.expander("About your efficacy prior"):
    if (a0 + b0) > 0:
        st.write(f"• Prior mean = **{a0/(a0+b0):.3f}**, prior ESS ≈ **{a0+b0:.1f}**.")
    else:
        st.write("• Prior mean = undefined (a₀+b₀=0). Consider Beta(1,1) for uninformative.")

allow_early_success = st.sidebar.checkbox(
    "Allow early success at interim looks?",
    value=False,
    help="If checked: at interims, stop early if P(p>p₀|data) ≥ θ_interim."
)
theta_interim = st.sidebar.number_input(
    "Interim success threshold (θ_interim)",
    min_value=0.5, max_value=0.999, value=float(theta_final), step=0.01, format="%.3f",
    help="Recommended ≥ θ_final for stricter early success. Used only if early success is enabled."
)

# ── Safety monitoring ───────────────────────────────────────────────────────

st.sidebar.header("2) Safety monitoring (optional)")
enable_safety = st.sidebar.checkbox(
    "Enable safety (toxicity) monitoring?",
    value=True,
    help="Monitor an SAE rate with a Beta prior and stop if it likely exceeds q_max."
)

if enable_safety:
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        a_t0 = st.number_input(
            "Safety prior a_t0",
            min_value=0.0, value=1.0, step=0.5,
            help="Safety prior Beta(a_t0, b_t0). Mean=a_t0/(a_t0+b_t0); ESS=a_t0+b_t0."
        )
        q_max = st.number_input(
            "Max acceptable SAE (q_max)",
            min_value=0.0, max_value=1.0, value=0.15, step=0.01, format="%.2f",
            help="If the posterior chance that SAE rate exceeds this is large, the study stops for safety."
        )
    with col_s2:
        b_t0 = st.number_input(
            "Safety prior b_t0",
            min_value=0.0, value=9.0, step=0.5,
            help="Safety prior Beta(a_t0, b_t0). Increase a_t0+b_t0 for stronger historical belief."
        )
        theta_tox = st.number_input(
            "Safety stop threshold (θ_tox)",
            min_value=0.5, max_value=0.999, value=0.90, step=0.01, format="%.3f",
            help="Stop for safety when P(q>q_max | data) ≥ θ_tox."
        )

    with st.sidebar.expander("About your safety prior"):
        if (a_t0 + b_t0) > 0:
            st.write(f"• Prior mean = **{a_t0/(a_t0+b_t0):.3f}**, prior ESS ≈ **{a_t0+b_t0:.1f}**.")
        else:
            st.write("• Prior mean = undefined (a_t0+b_t0=0). Consider Beta(1,9) for ~10% prior mean.")

else:
    a_t0 = b_t0 = q_max = theta_tox = None

# ── Look schedule ───────────────────────────────────────────────────────────

st.sidebar.header("3) Interim look schedule")
looks_mode = st.sidebar.selectbox(
    "How should we time the interims?",
    options=[
        "None (final only)",
        "Equal‑spaced (choose number of looks)",
        "Custom percentages of N",
        "Custom absolute Ns"
    ],
    index=1,
    help="Equal‑spaced: evenly spaced looks; Custom: specify either % of N or exact Ns."
)

k_looks = perc_str = ns_str = None
if looks_mode == "Equal‑spaced (choose number of looks)":
    k_looks = st.sidebar.slider(
        "Number of interim looks",
        min_value=1, max_value=8, value=2, step=1,
        help="Interims at roughly i/(k+1) of N; e.g., 2 looks → ~33% and ~67%."
    )
elif looks_mode == "Custom percentages of N":
    perc_str = st.sidebar.text_input(
        "Interim percentages (comma‑separated)",
        value="33,67",
        help="Example: 25,50,75 (values are percent of the planned maximum N)"
    )
elif looks_mode == "Custom absolute Ns":
    ns_str = st.sidebar.text_input(
        "Interim sample sizes N (comma‑separated)",
        value="",
        help="Example: 20,40 (each must be < total N)"
    )

# ── Screening grid ──────────────────────────────────────────────────────────

st.sidebar.header("4) Rapid Screener settings")
N_min, N_max = st.sidebar.slider(
    "Range of maximum sample sizes to test",
    min_value=10, max_value=400, value=(30, 120), step=1,
    help="We'll scan through this range to find feasible, efficient designs."
)
N_step = st.sidebar.number_input(
    "Step size for N grid",
    min_value=1, max_value=50, value=5, step=1,
    help="Larger steps scan fewer values of N (faster)."
)
n_sims_small = st.sidebar.number_input(
    "Screening simulations per design",
    min_value=100, max_value=200000, value=5000, step=500,
    help="Higher = more precise screening, slower. Deep dive later is where precision matters most."
)
alpha_max = st.sidebar.number_input(
    "Max Type I error (α) allowed",
    min_value=0.0, max_value=0.5, value=0.10, step=0.01, format="%.2f",
    help="We keep only designs whose false‑positive rate is at or below this."
)
power_min = st.sidebar.number_input(
    "Min power at p₁",
    min_value=0.0, max_value=1.0, value=0.80, step=0.01, format="%.2f",
    help="We keep only designs whose success rate at p₁ is at or above this."
)
seed = st.sidebar.number_input(
    "Random seed",
    min_value=1, value=2026, step=1,
    help="Controls reproducibility of the simulations."
)

# ── Build candidate grid from UI selections ─────────────────────────────────

def parse_percent_list(s: str) -> List[float]:
    vals = []
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
    vals = []
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

def look_schedule(N: int, mode: str, k_looks=None, perc_str=None, ns_str=None) -> List[int]:
    if mode == "None (final only)":
        looks = []
    elif mode == "Equal‑spaced (choose number of looks)":
        k = int(k_looks or 1)
        looks = [int(np.floor(i * N / (k + 1))) for i in range(1, k + 1)]
    elif mode == "Custom percentages of N":
        fracs = parse_percent_list(perc_str or "")
        looks = [int(np.floor(f * N)) for f in fracs]
    elif mode == "Custom absolute Ns":
        ns = parse_n_list(ns_str or "")
        looks = ns
    else:
        looks = []
    # Clean-up: enforce 1..N-1, uniqueness, sorting
    looks = [int(min(max(1, l), N - 1)) for l in looks if 0 < l < N]
    looks = sorted(list(dict.fromkeys(looks)))
    return looks

Ns = list(range(N_min, N_max + 1, N_step))
param_grid = []
for N in Ns:
    looks = look_schedule(N, looks_mode, k_looks=k_looks, perc_str=perc_str, ns_str=ns_str)
    param_grid.append({
        "N_total": N,
        "looks": looks,
        "a0": a0,
        "b0": b0,
        "p0": p0,
        "p1": p1,
        "theta_final": theta_final,
        "theta_interim": float(theta_interim),
        "c_futility": c_futility,
        "allow_early_success": allow_early_success
    })

# ── Rapid Screener section ──────────────────────────────────────────────────

st.write("### 1) Rapid Screener")
st.caption(
    "We quickly evaluate many candidate designs using efficacy‑only simulations and exact thresholds. "
    "Then we keep those that meet your α/power criteria and sort by low average N under the null."
)

@st.cache_data(show_spinner=False)
def _screen(param_grid, n_sims_small, seed):
    rng = np.random.default_rng(seed)
    Nmax = max([g["N_total"] for g in param_grid])
    U = rng.uniform(size=(n_sims_small, Nmax))
    df, designs = shortlist_designs(param_grid, n_sims_small, seed, U)
    return df, designs

df_screen, designs_built = _screen(param_grid, n_sims_small, seed)

if df_screen.empty:
    st.warning("No viable designs found (final rule may be impossible). Try relaxing θ_final or expanding N.")
else:
    df_ok = df_screen[
        (df_screen["Type I error @ p0"] <= alpha_max) &
        (df_screen["Power @ p1"] >= power_min)
    ].copy()

    if df_ok.empty:
        st.info("No designs met both α and power. Showing the full screening table instead (for inspection).")
        st.dataframe(df_screen.sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True))
        df_to_select_from = df_screen.sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True)
    else:
        df_ranked = df_ok.sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True)
        st.dataframe(df_ranked.head(15))
        st.success(f"Found {len(df_ok)} candidates that meet your criteria. Showing top 15 ranked by low ESS @ p₀.")
        df_to_select_from = df_ranked

    # ── Deep dive selection & controls ───────────────────────────────────────

    st.write("### 2) Deep Dive on Selected Design")
    st.caption("Now we run **joint** simulations (efficacy + optional safety) to get precise operating characteristics.")

    idx = st.number_input(
        "Which row (0-based) from the table above should we deep-dive?",
        min_value=0, value=0, step=1,
        help="Pick any row index you see in the table above."
    )

    if len(df_to_select_from) > 0:
        idx_used = int(np.clip(idx, 0, len(df_to_select_from) - 1))
        chosen = df_to_select_from.iloc[idx_used]
    else:
        chosen = None

    if chosen is not None:
        st.write("**Chosen design (key settings)**")
        show_cols = [
            "N_total", "looks", "theta_final", "theta_interim", "c_futility",
            "allow_early_success", "Type I error @ p0", "Power @ p1",
            "ESS @ p0", "ESS @ p1", "s_min_final"
        ]
        st.json({k: (int(chosen[k]) if isinstance(chosen[k], (np.integer,)) else chosen.get(k, None)) for k in show_cols})

        st.caption("Interim CONTINUE thresholds (you must have at least x responders to continue):")
        st.write(chosen["x_min_to_continue"])

        st.write("#### Deep-dive simulation settings")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            n_sims_deep = st.number_input(
                "Simulations to run",
                min_value=2000, max_value=800000, value=150000, step=5000,
                help="More simulations → tighter precision; also slower."
            )
        with col_d2:
            seed_deep = st.number_input(
                "Random seed (deep dive)",
                min_value=1, value=seed + 1, step=1,
                help="Change this if you want independent runs with similar precision."
            )

        st.write("#### Toxicity scenarios (for joint analysis)")
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            q_good = st.number_input(
                "q_good (typical/benign toxicity)",
                min_value=0.0, max_value=1.0, value=0.10, step=0.01,
                help="We’ll report metrics here to reflect expected safety."
            )
        with col_q2:
            q_bad = st.number_input(
                "q_bad (concerning/high toxicity)",
                min_value=0.0, max_value=1.0, value=0.20, step=0.01,
                help="We’ll report the chance that the safety rule would stop the trial here."
            )
        with col_q3:
            q_for_OC = st.number_input(
                "Fixed q for OC/ESS curves",
                min_value=0.0, max_value=1.0, value=0.10, step=0.01,
                help="When plotting success probability vs p, hold toxicity fixed at this value."
            )

        # Prepare selected design for simulation (include safety if enabled)
        design = dict(chosen["_design"])  # shallow copy
        if enable_safety:
            design["safety"] = dict(a_t0=a_t0, b_t0=b_t0, q_max=q_max, theta_tox=theta_tox)

        p0_used = design.get("p0", p0)
        p1_used = design.get("p1", p1)

        if st.button("Run DEEP‑DIVE simulation (efficacy + safety)"):
            rng = np.random.default_rng(seed_deep)
            Ueff = rng.uniform(size=(n_sims_deep, design["N_total"]))
            Utox = rng.uniform(size=(n_sims_deep, design["N_total"]))

            # Point scenarios (with safety logic if enabled)
            res_p0_qgood = simulate_design_with_safety(design, p_eff=p0_used, p_tox=q_good, U_eff=Ueff, U_tox=Utox)
            res_p1_qgood = simulate_design_with_safety(design, p_eff=p1_used, p_tox=q_good, U_eff=Ueff, U_tox=Utox)
            res_p1_qbad  = simulate_design_with_safety(design, p_eff=p1_used, p_tox=q_bad,  U_eff=Ueff, U_tox=Utox)

            st.write("##### Point estimates (joint, with safety if enabled)")
            cols = st.columns(5)
            cols[0].metric("Type I @ p₀, q_good", f"{res_p0_qgood['reject_rate']:.3f}")
            cols[1].metric("Power @ p₁, q_good", f"{res_p1_qgood['reject_rate']:.3f}")
            cols[2].metric("ESS @ p₀, q_good", f"{res_p0_qgood['ess']:.1f}")
            cols[3].metric("ESS @ p₁, q_good", f"{res_p1_qgood['ess']:.1f}")
            if enable_safety:
                cols[4].metric("P(Safety stop) @ p₁, q_bad", f"{res_p1_qbad['safety_stop_prob']:.3f}")

            st.write("##### Sample-size distribution (where trials tend to stop)")
            st.write("At the null (p₀, q_good):")
            st.dataframe(res_p0_qgood["stop_dist"])
            st.write("At the target (p₁, q_good):")
            st.dataframe(res_p1_qgood["stop_dist"])

            # OC & ESS curves vs p with fixed q
            st.write("##### OC & ESS vs efficacy p (toxicity held fixed)")
            p_grid_min = st.number_input(
                "OC curve p‑min",
                min_value=0.0, max_value=1.0, value=max(0.0, p0_used - 0.15), step=0.01
            )
            p_grid_max = st.number_input(
                "OC curve p‑max",
                min_value=0.0, max_value=1.0, value=min(1.0, p1_used + 0.20), step=0.01
            )
            n_grid = st.slider(
                "Number of points on the grid",
                min_value=5, max_value=40, value=15, step=1
            )

            ps = np.linspace(p_grid_min, p_grid_max, n_grid)
            oc, ess_curve = [], []
            for pp in ps:
                r = simulate_design_with_safety(design, p_eff=pp, p_tox=q_for_OC, U_eff=Ueff, U_tox=Utox)
                oc.append(r["reject_rate"])
                ess_curve.append(r["ess"])

            df_oc = pd.DataFrame({"p": ps, "P(Declare success)": oc, "Expected N": ess_curve})
            plot_lines(df_oc.rename(columns={"P(Declare success)":"Reject_Prob"}), x="p", y="Reject_Prob",
                       title=f"Operating Characteristic vs p (toxicity fixed at q={q_for_OC:.2f})")
            plot_lines(df_oc.rename(columns={"Expected N":"ESS"}), x="p", y="ESS",
                       title=f"Expected sample size vs p (toxicity fixed at q={q_for_OC:.2f})")

            st.write("##### Exportable design summary (copy‑paste into protocol/SAP)")
            export = dict(
                N_total=int(design["N_total"]),
                looks=[int(x) for x in design["looks"]],
                prior_efficacy=dict(a0=float(design["a0"]), b0=float(design["b0"])),
                prior_safety=(dict(a_t0=float(a_t0), b_t0=float(b_t0)) if enable_safety else None),
                null_p0=float(p0_used),
                target_p1=float(p1_used),
                theta_final=float(design["theta_final"]),
                theta_interim=float(design.get("theta_interim", design["theta_final"])),
                c_futility=float(design["c_futility"]),
                allow_early_success=bool(design["allow_early_success"]),
                final_success_min_successes=int(design["s_min_final"]),
                interim_continue_thresholds={int(k): (None if v is None else int(v)) for k, v in design["x_min_to_continue_by_look"].items()},
                safety_rule=(dict(q_max=float(q_max), theta_tox=float(theta_tox)) if enable_safety else None),
                notes=(
                    "Safety: stop if P(q>q_max | data) ≥ θ_tox at interims/final; "
                    "Efficacy: early success if P(p>p0 | data) ≥ θ_interim; "
                    "Futility: stop if PPoS(final success) < c_futility; "
                    "Final success if P(p>p0 | final data) ≥ θ_final."
                )
            )
            st.code(repr(export), language="python")

st.write("---")
st.write("### Methodological glossary (quick reference)")
st.markdown(
    """
- **p₀ (null rate)** – The highest response rate you'd still call unacceptable.
- **p₁ (target rate)** – A response rate worth declaring success if supported by the data.
- **θ_final** – Posterior threshold used at the final look: declare success if P(p>p₀|data) ≥ θ_final.
- **θ_interim** – Optional stricter posterior threshold to allow early success at interims.
- **PPoS (Predictive Probability of Success)** – Given data so far, the chance that the final analysis will meet the success rule after collecting the remaining data.
- **c_futility** – If PPoS falls below this at an interim look, stop early for futility.
- **Safety rule** – Stop if P(q>q_max|data) ≥ θ_tox, where q is the SAE rate.
- **ESS (Expected Sample Size)** – The average number of patients enrolled before stopping.
"""
)
st.caption("Tip: Use stricter θ_interim and/or larger c_futility for more early stopping; adjust safety θ_tox and q_max to match DSMB preferences.")
