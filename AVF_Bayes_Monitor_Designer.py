# AVF_Bayes_Monitor_Designer.py
# Streamlit app for single-arm Bayesian monitored design (binary endpoint)
# Rapid screener + deep-dive simulation WITH optional safety monitoring
#
# ✦ This version adds:
#   • Early-stop / early-success rates in the Screener (efficacy-only)
#   • Early-stop breakdown in Deep-dive (success / futility / safety) with by-look arrays
#   • Stacked by-look charts (Plotly if available; Streamlit fallback)
#   • Compare panel to evaluate multiple Ns side-by-side under joint (efficacy+safety) simulation
#   • Robust cache invalidation + safe column selection to prevent KeyError after updates
#   • Extensive tooltips and comments throughout
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
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# ──────────────────────────────────────────────────────────────────────────────
# Cache schema version: bump this string whenever you change screening columns.
# This forces Streamlit's cache to refresh stale dataframes after updates.
# ──────────────────────────────────────────────────────────────────────────────
SCHEMA_VERSION = "screen_v2_earlystop"  # bump if you edit screening columns again


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CORE BAYESIAN UTILITIES                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def beta_posterior_params(a0: float, b0: float, x: int, n: int) -> Tuple[float, float]:
    """Posterior Beta(a0+x, b0+n-x) after observing x successes in n."""
    return a0 + x, b0 + (n - x)

def posterior_prob_p_greater_than(p_cut: float, a_post: float, b_post: float) -> float:
    """Compute P(p > p_cut | data) under Beta(a_post, b_post)."""
    return 1.0 - beta.cdf(p_cut, a_post, b_post)

def min_successes_for_posterior_threshold(
    a0: float, b0: float, N: int, p0: float, theta_final: float
) -> Optional[int]:
    """
    For fixed max N, find the smallest total successes s_min so that
    P(p>p0 | Beta(a0+s, b0+N-s)) ≥ theta_final. Binary search in s.
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
    """log PMF of Beta–Binomial(m; a, b): C(m,y)*B(y+a, m-y+b)/B(a,b)."""
    if y < 0 or y > m:
        return -np.inf
    return np.log(comb(m, y)) + betaln(y + a, m - y + b) - betaln(a, b)

def beta_binomial_cdf_upper_tail(y_min: int, m: int, a: float, b: float) -> float:
    """Sum_{y>=y_min} PMF for Beta–Binomial(m; a, b) via log-sum-exp."""
    if y_min <= 0:
        return 1.0
    if y_min > m:
        return 0.0
    ys = np.arange(y_min, m + 1)
    logs = np.array([log_beta_binomial_pmf(int(y), m, a, b) for y in ys])
    mlog = np.max(logs)
    return float(np.exp(mlog) * np.sum(np.exp(logs - mlog)))

def predictive_prob_of_final_success(
    a0: float, b0: float, N_total: int, x_curr: int, n_curr: int, p0: float, theta_final: float
) -> float:
    """
    Predictive Probability of Success (PPoS) at an interim:
    chance (over FUTURE data) that the final rule will be met, given current x_curr/n_curr.
    """
    a_post, b_post = beta_posterior_params(a0, b0, x_curr, n_curr)
    m_remain = N_total - n_curr
    s_min = min_successes_for_posterior_threshold(a0, b0, N_total, p0, theta_final)
    if s_min is None:
        return 0.0
    y_needed = s_min - x_curr
    return beta_binomial_cdf_upper_tail(y_needed, m_remain, a_post, b_post)

def compute_interim_futility_cutoffs(
    a0: float, b0: float, N_total: int, looks: List[int], p0: float, theta_final: float, c_futility: float
) -> Dict[int, Optional[int]]:
    """
    For each interim n in looks, compute minimal x_min (current successes) to CONTINUE:
    the smallest x such that PPoS ≥ c_futility. None ⇒ always stop at that look.
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

def simulate_design(design: Dict, p: float, U: np.ndarray) -> Dict:
    """
    Efficacy-only simulation (used in SCREENING for speed).
    Adds early-stop accounting by reason (success vs futility) and by-look arrays.
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
    X = (U[:, :N] < p).astype(np.int16)

    cum_x = np.zeros(n_sims, dtype=np.int32)
    n_curr = 0
    active = np.ones(n_sims, dtype=bool)
    success = np.zeros(n_sims, dtype=bool)
    final_n = np.zeros(n_sims, dtype=np.int32)

    # Per-look counts and reason-specific counts
    stop_by_look_counts = np.zeros(len(looks), dtype=np.int64)
    early_succ_by_look = np.zeros(len(looks), dtype=np.int64)
    early_fut_by_look  = np.zeros(len(looks), dtype=np.int64)

    for li, look_n in enumerate(looks):
        # Enroll up to the interim
        add = look_n - n_curr
        if add > 0:
            cum_x[active] += np.sum(X[active, n_curr:look_n], axis=1)
            n_curr = look_n

        # Early success (efficacy) if enabled
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
                stop_by_look_counts[li] += early_succ.sum()

        if not active.any():
            break

        # Interim futility via PPoS continue threshold
        x_min = x_min_to_continue.get(look_n, None)
        if x_min is None:
            idx = np.where(active)[0]
            if idx.size > 0:
                final_n[idx] = n_curr
                active[idx] = False
                early_fut_by_look[li] += idx.size
                stop_by_look_counts[li] += idx.size
        else:
            need_continue = cum_x[active] >= x_min
            idx_all_active = np.where(active)[0]
            idx_stop = idx_all_active[~need_continue]
            if idx_stop.size > 0:
                final_n[idx_stop] = n_curr
                active[idx_stop] = False
                early_fut_by_look[li] += idx_stop.size
                stop_by_look_counts[li] += idx_stop.size

        if not active.any():
            break

    # Final analysis for those still active
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

    # Early-stop totals as rates
    early_succ_rate = early_succ_by_look.sum() / n_sims
    early_fut_rate  = early_fut_by_look.sum()  / n_sims
    early_stop_rate = early_succ_rate + early_fut_rate

    unique_ns, counts = np.unique(final_n, return_counts=True)
    stop_dist = pd.DataFrame({"N_stop": unique_ns, "Probability": counts / n_sims}).sort_values("N_stop")

    return {
        "reject_rate": float(reject_rate),
        "ess": float(ess),
        "stop_probs_by_look": (stop_by_look_counts / n_sims).tolist(),
        "stop_dist": stop_dist,
        # New: early-stop outputs
        "early_succ_by_look": (early_succ_by_look / n_sims).tolist(),
        "early_fut_by_look":  (early_fut_by_look  / n_sims).tolist(),
        "early_succ_rate": float(early_succ_rate),
        "early_fut_rate":  float(early_fut_rate),
        "early_stop_rate": float(early_stop_rate),
    }


def simulate_design_with_safety(
    design: Dict, p_eff: float, p_tox: float, U_eff: np.ndarray, U_tox: np.ndarray
) -> Dict:
    """
    Joint efficacy + safety simulation (deep dive).
    Adds early-stop breakdown by reason (success / futility / safety) and by-look arrays.
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
    X_eff = (U_eff[:, :N] < p_eff).astype(np.int16)
    X_tox = (U_tox[:, :N] < p_tox).astype(np.int16) if tox is not None else None

    cum_x = np.zeros(n_sims, dtype=np.int32)
    cum_t = np.zeros(n_sims, dtype=np.int32) if tox is not None else None
    n_curr = 0
    active = np.ones(n_sims, dtype=bool)
    success = np.zeros(n_sims, dtype=bool)
    final_n = np.zeros(n_sims, dtype=np.int32)

    # Reason-specific by-look counts
    early_succ_by_look = np.zeros(len(looks), dtype=np.int64)
    early_fut_by_look  = np.zeros(len(looks), dtype=np.int64)
    safety_by_look     = np.zeros(len(looks) + 1, dtype=np.int64)  # include slot [-1] for final safety stops

    for li, look_n in enumerate(looks):
        add = look_n - n_curr
        if add > 0:
            cum_x[active] += np.sum(X_eff[active, n_curr:look_n], axis=1)
            if tox is not None:
                cum_t[active] += np.sum(X_tox[active, n_curr:look_n], axis=1)
            n_curr = look_n

        # 1) Safety check first
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
                safety_by_look[li] += unsafe.sum()

        if not active.any():
            break

        # 2) Early efficacy success if enabled
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

        # 3) Futility by PPoS continue threshold
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

    # Final stage for those still active
    if active.any():
        add = N - n_curr
        if add > 0:
            cum_x[active] += np.sum(X_eff[active, n_curr:N], axis=1)
            if tox is not None:
                cum_t[active] += np.sum(X_tox[active, n_curr:N], axis=1)
            n_curr = N

        # Final safety check
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
                safety_by_look[-1] += unsafe.sum()

        # Final efficacy decision
        if active.any():
            succ_final = (cum_x[active] >= s_min_final)
            idx_active = np.where(active)[0]
            success[idx_active[succ_final]] = True
            final_n[idx_active] = N

    reject_rate = success.mean()
    ess = final_n.mean()

    # Early-stop rates (fractions of trials)
    n_sims = U_eff.shape[0]
    early_succ_rate   = early_succ_by_look.sum() / n_sims
    early_fut_rate    = early_fut_by_look.sum()  / n_sims
    early_safety_rate = safety_by_look[:-1].sum() / n_sims  # interims only
    any_safety_rate   = safety_by_look.sum() / n_sims       # interims + final safety stops
    early_stop_rate   = early_succ_rate + early_fut_rate + early_safety_rate

    unique_ns, counts = np.unique(final_n, return_counts=True)
    stop_dist = pd.DataFrame({"N_stop": unique_ns, "Probability": counts / n_sims}).sort_values("N_stop")

    return {
        "reject_rate": float(reject_rate),
        "ess": float(ess),
        "safety_stop_prob": float(any_safety_rate),
        "stop_dist": stop_dist,
        # Reason/by-look outputs
        "early_succ_by_look": (early_succ_by_look / n_sims).tolist(),
        "early_fut_by_look":  (early_fut_by_look  / n_sims).tolist(),
        "safety_by_look":     (safety_by_look     / n_sims).tolist(),
        "early_succ_rate":    float(early_succ_rate),
        "early_fut_rate":     float(early_fut_rate),
        "early_safety_rate":  float(early_safety_rate),
        "early_stop_rate":    float(early_stop_rate),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         SCREENING (FAST) PIPELINE                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def shortlist_designs(param_grid: List[Dict], n_sims_small: int, seed: int, U: Optional[np.ndarray] = None):
    """
    Screen many candidate designs quickly (efficacy-only).
    Records early-stop metrics at p0 for quick comparisons.
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
            "Early stop @ p0 (any)": res_p0["early_stop_rate"],   # NEW
            "Early success @ p0":    res_p0["early_succ_rate"],   # NEW
            "Early futility @ p0":   res_p0["early_fut_rate"],    # NEW
            "s_min_final": s_min,
            "x_min_to_continue": x_min_to_continue,
            "_design": design,
            "_early_succ_by_look_p0": res_p0["early_succ_by_look"],   # optional detail
            "_early_fut_by_look_p0":  res_p0["early_fut_by_look"],    # optional detail
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

def plot_stacked_bylook(looks: List[int], succ: List[float], fut: List[float], safety: Optional[List[float]], title: str):
    """
    Stacked bar chart per look for early decisions (success/futility/safety).
    safety can be None if not tracked (e.g., screening).
    """
    labels = [str(n) for n in looks]
    if safety is not None and len(safety) == len(looks) + 1 and labels:
        # ignore the last "final" slot for by-look chart
        safety = safety[:-1]

    df = pd.DataFrame({
        "look": labels,
        "Early success": succ,
        "Early futility": fut,
        **({"Early safety": safety} if safety is not None else {})
    })

    if _HAS_PLOTLY:
        fig = go.Figure()
        fig.add_bar(x=df["look"], y=df["Early success"], name="Early success")
        fig.add_bar(x=df["look"], y=df["Early futility"], name="Early futility")
        if safety is not None:
            fig.add_bar(x=df["look"], y=df["Early safety"], name="Early safety")
        fig.update_layout(barmode="stack", title=title, xaxis_title="Interim look (n)", yaxis_title="Fraction of trials")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader(title)
        st.dataframe(df.set_index("look"))  # fallback: show table; streamlit's stacked bar is limited


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
**Goal.** Pick a trial design that has **adequate power**, **controls Type I error**, and **stops early** for **futility**, **clear success**, or **safety**.

**How it works (short):**
- **Binary endpoint** with Beta prior.
- **Final success** if P(p>p₀|data) ≥ θ_final.
- **Futility** at interims if **PPoS** < c_futility.
- **Early success** at interims if P(p>p₀|data) ≥ θ_interim (optional).
- **Safety** (optional): stop if P(q>q_max|data) ≥ θ_tox at interims/final.
        """
    )

# ── Sidebar: inputs ─────────────────────────────────────────────────────────

st.sidebar.header("1) Set your design targets")
col_sb1, col_sb2 = st.sidebar.columns(2)

with col_sb1:
    p0 = st.number_input(
        "Null response rate (p₀)",
        min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f",
        help="Highest response rate still considered clinically unacceptable."
    )
    a0 = st.number_input(
        "Prior a₀ (efficacy)",
        min_value=0.0, value=1.0, step=0.5,
        help="Efficacy prior Beta(a₀, b₀). Use (1,1) for uninformative. Increase a₀+b₀ for a stronger prior."
    )
    theta_final = st.number_input(
        "Final success threshold (θ_final)",
        min_value=0.5, max_value=0.999, value=0.95, step=0.01, format="%.3f",
        help="At the final look, declare success if P(p>p₀ | data) ≥ θ_final."
    )

with col_sb2:
    p1 = st.number_input(
        "Target response rate (p₁)",
        min_value=0.0, max_value=1.0, value=0.40, step=0.01, format="%.2f",
        help="A rate you consider clinically promising; power is evaluated here."
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

with st.sidebar.expander("About your efficacy prior"):
    if (a0 + b0) > 0:
        st.write(f"• Prior mean = **{a0/(a0+b0):.3f}**, prior ESS ≈ **{a0+b0:.1f}**.")
    else:
        st.write("• Prior mean undefined (a₀+b₀=0). Consider Beta(1,1).")

allow_early_success = st.sidebar.checkbox(
    "Allow early success at interim looks?",
    value=False,
    help="If checked, at interims stop early for success when P(p>p₀|data) ≥ θ_interim."
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
    help="Monitor an SAE rate with a Beta prior; stop if it likely exceeds q_max."
)
if enable_safety:
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        a_t0 = st.number_input(
            "Safety prior a_t0", min_value=0.0, value=1.0, step=0.5,
            help="Safety prior Beta(a_t0, b_t0). Mean=a_t0/(a_t0+b_t0); ESS=a_t0+b_t0."
        )
        q_max = st.number_input(
            "Max acceptable SAE (q_max)", min_value=0.0, max_value=1.0, value=0.15, step=0.01, format="%.2f",
            help="If P(q>q_max|data) is high, the study stops for safety."
        )
    with col_s2:
        b_t0 = st.number_input(
            "Safety prior b_t0", min_value=0.0, value=9.0, step=0.5,
            help="Increase a_t0+b_t0 for stronger prior. Beta(1,9) ⇒ mean ~0.10."
        )
        theta_tox = st.number_input(
            "Safety stop threshold (θ_tox)",
            min_value=0.5, max_value=0.999, value=0.90, step=0.01, format="%.3f",
            help="Stop when P(q>q_max | data) ≥ θ_tox."
        )
    with st.sidebar.expander("About your safety prior"):
        if (a_t0 + b_t0) > 0:
            st.write(f"• Prior mean = **{a_t0/(a_t0+b_t0):.3f}**, prior ESS ≈ **{a_t0+b_t0:.1f}**.")
        else:
            st.write("• Prior mean undefined (a_t0+b_t0=0). Consider Beta(1,9) for ~10% prior mean.")
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
    help="Equal‑spaced: evenly spaced looks; Custom: specify % of N or exact Ns."
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
    help="Larger steps scan fewer N values (faster)."
)
n_sims_small = st.sidebar.number_input(
    "Screening simulations per design",
    min_value=100, max_value=200000, value=5000, step=500,
    help="Higher = more precision; deep dive is where precision matters most."
)
alpha_max = st.sidebar.number_input(
    "Max Type I error (α) allowed",
    min_value=0.0, max_value=0.5, value=0.10, step=0.01, format="%.2f",
    help="Keep only designs with false‑positive rate ≤ this."
)
power_min = st.sidebar.number_input(
    "Min power at p₁",
    min_value=0.0, max_value=1.0, value=0.80, step=0.01, format="%.2f",
    help="Keep only designs with success rate at p₁ ≥ this."
)
seed = st.sidebar.number_input(
    "Random seed",
    min_value=1, value=2026, step=1,
    help="Controls reproducibility."
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

# ── Helper: safe column selection to avoid KeyError on cached DF schema changes ──
def _safe_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Return only the columns that actually exist in df (avoid KeyError)."""
    return [c for c in cols if c in df.columns]

# ── Rapid Screener ──────────────────────────────────────────────────────────

st.write("### 1) Rapid Screener")
st.caption(
    "We quickly evaluate many candidate designs (efficacy‑only) and keep those that meet your α/power criteria. "
    "We now also display **early-stop rates** under the null to show how aggressive each design is."
)

@st.cache_data(show_spinner=False)
def _screen(param_grid, n_sims_small, seed, schema_version: str):
    """
    Cached screening function.
    schema_version is a dummy key we pass to force cache invalidation after schema changes.
    """
    rng = np.random.default_rng(seed)
    Nmax = max([g["N_total"] for g in param_grid])
    U = rng.uniform(size=(n_sims_small, Nmax))
    df, designs = shortlist_designs(param_grid, n_sims_small, seed, U)
    _ = schema_version  # make the version a dependency
    return df, designs

df_screen, designs_built = _screen(param_grid, n_sims_small, seed, SCHEMA_VERSION)

if df_screen.empty:
    st.warning("No viable designs found (final rule may be impossible). Try relaxing θ_final or expanding N.")
else:
    df_ok = df_screen[
        (df_screen["Type I error @ p0"] <= alpha_max) &
        (df_screen["Power @ p1"] >= power_min)
    ].copy()

    cols_to_show = [
        "N_total", "looks", "theta_final", "c_futility",
        "Type I error @ p0", "Power @ p1", "ESS @ p0",
        "Early stop @ p0 (any)", "Early success @ p0", "Early futility @ p0"
    ]

    if df_ok.empty:
        st.info("No designs met both α and power. Showing the full screening table instead (for inspection).")
        st.dataframe(
            df_screen[_safe_cols(df_screen, cols_to_show)]
            .sort_values(["ESS @ p0", "N_total"])
            .reset_index(drop=True)
        )
        df_to_select_from = df_screen.sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True)
    else:
        df_ranked = df_ok.sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True)
        st.dataframe(df_ranked[_safe_cols(df_ranked, cols_to_show)].head(15))
        st.success(f"Found {len(df_ok)} candidates that meet your criteria. Showing top 15 ranked by low ESS @ p₀.")
        df_to_select_from = df_ranked

    # ── Deep dive selection & controls ───────────────────────────────────────

    st.write("### 2) Deep Dive on Selected Design")
    st.caption("Now we run **joint** simulations (efficacy + optional safety) for precise operating characteristics and early-stop breakdowns.")

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

        st.caption("Interim CONTINUE thresholds (must have at least x responders to continue):")
        st.write(chosen["x_min_to_continue"])

        st.write("#### Deep-dive simulation settings")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            n_sims_deep = st.number_input(
                "Simulations to run",
                min_value=2000, max_value=800000, value=150000, step=5000,
                help="More simulations → tighter precision; slower."
            )
        with col_d2:
            seed_deep = st.number_input(
                "Random seed (deep dive)",
                min_value=1, value=seed + 1, step=1,
                help="Change to re-simulate independently."
            )

        st.write("#### Toxicity scenarios (for joint analysis)")
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            q_good = st.number_input(
                "q_good (typical/benign toxicity)",
                min_value=0.0, max_value=1.0, value=0.10, step=0.01,
                help="We’ll report main metrics under this toxicity level."
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

        design = dict(chosen["_design"])  # copy
        if enable_safety:
            design["safety"] = dict(a_t0=a_t0, b_t0=b_t0, q_max=q_max, theta_tox=theta_tox)

        p0_used = design.get("p0", p0)
        p1_used = design.get("p1", p1)

        if st.button("Run DEEP‑DIVE simulation (efficacy + safety)"):
            rng = np.random.default_rng(seed_deep)
            Ueff = rng.uniform(size=(n_sims_deep, design["N_total"]))
            Utox = rng.uniform(size=(n_sims_deep, design["N_total"]))

            # Point scenarios with joint logic
            res_p0_qgood = simulate_design_with_safety(design, p_eff=p0_used, p_tox=q_good, U_eff=Ueff, U_tox=Utox)
            res_p1_qgood = simulate_design_with_safety(design, p_eff=p1_used, p_tox=q_good, U_eff=Ueff, U_tox=Utox)
            res_p1_qbad  = simulate_design_with_safety(design, p_eff=p1_used, p_tox=q_bad,  U_eff=Ueff, U_tox=Utox)

            st.write("##### Point estimates (joint, with safety if enabled)")
            cols = st.columns(6 if enable_safety else 5)
            cols[0].metric("Type I @ p₀, q_good", f"{res_p0_qgood['reject_rate']:.3f}")
            cols[1].metric("Power @ p₁, q_good", f"{res_p1_qgood['reject_rate']:.3f}")
            cols[2].metric("ESS @ p₀, q_good", f"{res_p0_qgood['ess']:.1f}")
            cols[3].metric("ESS @ p₁, q_good", f"{res_p1_qgood['ess']:.1f}")
            cols[4].metric("Early stop @ p₁ (any)", f"{res_p1_qgood['early_stop_rate']:.3f}")
            if enable_safety:
                cols[5].metric("P(Safety stop) @ p₁, q_bad", f"{res_p1_qbad['safety_stop_prob']:.3f}")

            # Early-stop breakdown (totals)
            st.write("##### Early stopping breakdown (q_good scenario)")
            cols2 = st.columns(4 if enable_safety else 3)
            cols2[0].metric("… early success @ p₁",  f"{res_p1_qgood['early_succ_rate']:.3f}")
            cols2[1].metric("… early futility @ p₁", f"{res_p1_qgood['early_fut_rate']:.3f}")
            if enable_safety:
                cols2[2].metric("… early safety @ p₁", f"{res_p1_qgood['early_safety_rate']:.3f}")
                cols2[3].metric("Early stop (any) @ p₁", f"{res_p1_qgood['early_stop_rate']:.3f}")
            else:
                cols2[2].metric("Early stop (any) @ p₁", f"{res_p1_qgood['early_stop_rate']:.3f}")

            # By-look table
            st.write("###### By‑look details @ p₁, q_good (fractions of trials)")
            bylook_df = pd.DataFrame({
                "look_n": design["looks"],
                "early_success": res_p1_qgood["early_succ_by_look"],
                "early_futility": res_p1_qgood["early_fut_by_look"],
            })
            if enable_safety:
                # trim off final safety slot for by-look
                safety_bylook = res_p1_qgood["safety_by_look"][:-1]
                bylook_df["early_safety"] = safety_bylook
            st.dataframe(bylook_df)

            # Stacked by-look chart
            plot_stacked_bylook(
                design["looks"],
                succ=res_p1_qgood["early_succ_by_look"],
                fut=res_p1_qgood["early_fut_by_look"],
                safety=(res_p1_qgood["safety_by_look"] if enable_safety else None),
                title="Early decisions by interim look @ p₁ (stacked)"
            )

            # Sample-size distributions
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

            df_oc = pd.DataFrame({"p": ps, "Reject_Prob": oc, "ESS": ess_curve})
            plot_lines(df_oc, x="p", y="Reject_Prob",
                       title=f"Operating Characteristic vs p (toxicity fixed at q={q_for_OC:.2f})")
            plot_lines(df_oc, x="p", y="ESS",
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

    # ── Compare panel ────────────────────────────────────────────────────────

    st.write("### 3) Compare multiple Ns (deep‑dive, joint with safety)")
    st.caption("Evaluate several maximum sample sizes with the SAME rule settings, then compare Power, Type I, ESS, and early/safety stops.")

    with st.expander("Open compare panel"):
        ns_str_compare = st.text_input(
            "Enter N values (comma-separated)",
            value="60,70,80",
            help="We’ll construct designs with the SAME thresholds you set above and the SAME look rule type."
        )
        n_sims_compare = st.number_input(
            "Simulations per design (compare run)",
            min_value=5000, max_value=400000, value=80000, step=5000,
            help="Higher = more precise, slower."
        )
        seed_compare = st.number_input(
            "Random seed (compare run)",
            min_value=1, value=seed + 2, step=1
        )
        q_good_cmp = st.number_input(
            "q_good for compare run",
            min_value=0.0, max_value=1.0, value=0.10, step=0.01
        )
        q_bad_cmp = st.number_input(
            "q_bad for compare run",
            min_value=0.0, max_value=1.0, value=0.20, step=0.01
        )

        # Helper to build looks per N reusing current look mode settings
        def build_looks_for_N(N):
            return look_schedule(N, looks_mode, k_looks=k_looks, perc_str=perc_str, ns_str=ns_str)

        if st.button("Run compare"):
            Ns_cmp = []
            for tok in ns_str_compare.split(","):
                tok = tok.strip()
                if tok:
                    try:
                        Ns_cmp.append(int(round(float(tok))))
                    except Exception:
                        pass
            Ns_cmp = [n for n in Ns_cmp if n >= 5]
            if not Ns_cmp:
                st.warning("Please enter one or more valid integers for N.")
            else:
                rng = np.random.default_rng(seed_compare)
                results = []
                for Ncmp in Ns_cmp:
                    looks_cmp = build_looks_for_N(Ncmp)
                    smin_cmp = min_successes_for_posterior_threshold(a0, b0, Ncmp, p0, theta_final)
                    if smin_cmp is None:
                        continue
                    x_min_cmp = compute_interim_futility_cutoffs(a0, b0, Ncmp, looks_cmp, p0, theta_final, c_futility)
                    design_cmp = dict(
                        N_total=Ncmp, looks=looks_cmp, a0=a0, b0=b0, p0=p0,
                        theta_final=theta_final, theta_interim=float(theta_interim),
                        c_futility=c_futility, allow_early_success=allow_early_success,
                        s_min_final=smin_cmp, x_min_to_continue_by_look=x_min_cmp
                    )
                    if enable_safety:
                        design_cmp["safety"] = dict(a_t0=a_t0, b_t0=b_t0, q_max=q_max, theta_tox=theta_tox)

                    Ueff = rng.uniform(size=(n_sims_compare, Ncmp))
                    Utox = rng.uniform(size=(n_sims_compare, Ncmp))

                    r_p0 = simulate_design_with_safety(design_cmp, p_eff=p0, p_tox=q_good_cmp, U_eff=Ueff, U_tox=Utox)
                    r_p1 = simulate_design_with_safety(design_cmp, p_eff=p1, p_tox=q_good_cmp, U_eff=Ueff, U_tox=Utox)
                    r_p1_bad = simulate_design_with_safety(design_cmp, p_eff=p1, p_tox=q_bad_cmp, U_eff=Ueff, U_tox=Utox)

                    results.append({
                        "N": Ncmp,
                        "looks": looks_cmp,
                        "s_min_final": smin_cmp,
                        "Type I @p0,q_good": r_p0["reject_rate"],
                        "Power @p1,q_good":  r_p1["reject_rate"],
                        "ESS @p0": r_p0["ess"],
                        "ESS @p1": r_p1["ess"],
                        "Early stop @p1": r_p1["early_stop_rate"],
                        "Early succ @p1": r_p1["early_succ_rate"],
                        "Early fut @p1":  r_p1["early_fut_rate"],
                        "Early safety @p1": r_p1["early_safety_rate"],
                        "Safety stop @p1,q_bad (any stage)": r_p1_bad["safety_stop_prob"],
                    })

                if not results:
                    st.warning("None of the N values produced a feasible final rule. Try relaxing θ_final or adjusting looks.")
                else:
                    df_cmp = pd.DataFrame(results).sort_values("N").reset_index(drop=True)
                    st.dataframe(df_cmp)

                    # Quick comparison charts: Power and ESS
                    if _HAS_PLOTLY:
                        fig_pow = px.bar(df_cmp, x="N", y="Power @p1,q_good", title="Power by N")
                        st.plotly_chart(fig_pow, use_container_width=True)
                        fig_ess = px.bar(df_cmp, x="N", y="ESS @p1", title="ESS @ p₁ by N")
                        st.plotly_chart(fig_ess, use_container_width=True)
                    else:
                        st.subheader("Power by N")
                        st.bar_chart(df_cmp.set_index("N")[["Power @p1,q_good"]])
                        st.subheader("ESS @ p₁ by N")
                        st.bar_chart(df_cmp.set_index("N")[["ESS @p1"]])

st.write("---")
st.write("### Methodological glossary (quick reference)")
st.markdown(
    """
- **p₀ (null rate)** – Highest response rate you'd still call unacceptable.  
- **p₁ (target rate)** – A response rate worth declaring success if supported by the data.  
- **θ_final** – Posterior threshold at the final look: success if P(p>p₀|data) ≥ θ_final.  
- **θ_interim** – Posterior threshold to allow early success at interims (optional; often ≥ θ_final).  
- **PPoS** – Predictive Probability of Success: chance (over future data) the study will meet the final rule.  
- **c_futility** – At interims, stop if PPoS < c_futility.  
- **Safety rule** – Stop if P(q>q_max|data) ≥ θ_tox, where q is the SAE rate.  
- **ESS (Expected Sample Size)** – Average number of patients enrolled before stopping.  
- **Early stop / success / futility / safety** – Fractions of trials that stop at an interim for the indicated reason.
"""
)
st.caption("Tip: Use stricter θ_interim and/or larger c_futility for more early stopping; adjust safety θ_tox and q_max to match DSMB preferences.")
