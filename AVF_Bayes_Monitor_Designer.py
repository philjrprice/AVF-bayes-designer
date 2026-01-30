 
# AVF_Bayes_Monitor_Designer_runin_safety_v2.py
# Streamlit app for single-arm Bayesian monitored design (binary endpoint)
# Rapid screener + deep-dive simulation WITH optional safety monitoring
# + Threshold Tuner for efficacy thresholds
# + Comprehensive OC Explorer (p,q grid with heatmaps and scenarios)
# + Export buttons for settings and results (JSON bundle + ZIP of CSVs)
#
from __future__ import annotations
import io
import json
import zipfile
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple
from scipy.stats import beta
from scipy.special import betaln, comb

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

SCHEMA_VERSION = "screen_v4_runin_splitlooks_tuner_oc_export"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CORE BAYESIAN UTILITIES                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def beta_posterior_params(a0: float, b0: float, x: int, n: int) -> Tuple[float, float]:
    return a0 + x, b0 + (n - x)

def posterior_prob_p_greater_than(p_cut: float, a_post: float, b_post: float) -> float:
    return 1.0 - beta.cdf(p_cut, a_post, b_post)

def min_successes_for_posterior_threshold(a0: float, b0: float, N: int, p0: float, theta_final: float) -> Optional[int]:
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
    if y < 0 or y > m:
        return -np.inf
    return np.log(comb(m, y)) + betaln(y + a, m - y + b) - betaln(a, b)

def beta_binomial_cdf_upper_tail(y_min: int, m: int, a: float, b: float) -> float:
    if y_min <= 0:
        return 1.0
    if y_min > m:
        return 0.0
    ys = np.arange(y_min, m + 1)
    logs = np.array([log_beta_binomial_pmf(int(y), m, a, b) for y in ys])
    mlog = np.max(logs)
    return float(np.exp(mlog) * np.sum(np.exp(logs - mlog)))

def predictive_prob_of_final_success(a0: float, b0: float, N_total: int, x_curr: int, n_curr: int, p0: float, theta_final: float) -> float:
    a_post, b_post = beta_posterior_params(a0, b0, x_curr, n_curr)
    m_remain = N_total - n_curr
    s_min = min_successes_for_posterior_threshold(a0, b0, N_total, p0, theta_final)
    if s_min is None:
        return 0.0
    y_needed = s_min - x_curr
    return beta_binomial_cdf_upper_tail(y_needed, m_remain, a_post, b_post)

def compute_interim_futility_cutoffs(a0: float, b0: float, N_total: int, looks_eff: List[int], p0: float, theta_final: float, c_futility: float) -> Dict[int, Optional[int]]:
    cutoffs: Dict[int, Optional[int]] = {}
    for n in looks_eff:
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
# ║ LOOK SCHEDULE UTILITIES                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_percent_list(s: str) -> List[float]:
    vals: List[float] = []
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
    vals: List[int] = []
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

def build_looks_with_runin(N: int, run_in: int, mode: str, k_total: Optional[int] = None, perc_str: Optional[str] = None, ns_str: Optional[str] = None) -> List[int]:
    looks: List[int] = []
    run_in = int(max(0, min(run_in, N - 1)))
    if run_in > 0:
        looks.append(run_in)

    if mode == "None (final only)":
        return looks

    if mode == "Equal‑spaced (choose total looks incl. run‑in)":
        K = int(k_total or 0)
        K_rem = max(0, K - (1 if run_in > 0 else 0))
        if K_rem > 0 and (N - run_in) > 0:
            for i in range(1, K_rem + 1):
                n_i = run_in + int(np.floor(i * (N - run_in) / (K_rem + 1)))
                if 0 < n_i < N:
                    looks.append(n_i)
    elif mode == "Custom percentages of remaining":
        fracs = parse_percent_list(perc_str or "")
        for f in fracs:
            n_i = run_in + int(np.floor(f * (N - run_in)))
            if 0 < n_i < N:
                looks.append(n_i)
    elif mode == "Custom absolute Ns":
        ns = parse_n_list(ns_str or "")
        for n_i in ns:
            if run_in > 0 and n_i == run_in:
                pass
            if run_in < n_i < N:
                looks.append(int(n_i))

    looks = [int(x) for x in looks if 0 < x < N]
    looks = sorted(list(dict.fromkeys(looks)))
    return looks

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ SIMULATION ENGINES                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def simulate_design_eff_only(design: Dict, p: float, U: np.ndarray) -> Dict:
    N = design["N_total"]
    looks_eff = design["looks_eff"]
    run_in_eff = design.get("run_in_eff", 0)
    a0 = design["a0"]; b0 = design["b0"]
    p0 = design["p0"]; theta_final = design["theta_final"]
    theta_interim = design.get("theta_interim", theta_final)
    c_fut = design["c_futility"]
    allow_early = design["allow_early_success"]
    s_min_final = design["s_min_final"]
    x_min_to_continue = design["x_min_to_continue_by_look_eff"]

    n_sims = U.shape[0]
    X = (U[:, :N] < p).astype(np.int16)

    cum_x = np.zeros(n_sims, dtype=np.int32)
    n_curr = 0
    active = np.ones(n_sims, dtype=bool)
    success = np.zeros(n_sims, dtype=bool)
    final_n = np.zeros(n_sims, dtype=np.int32)

    stop_by_look_counts = np.zeros(len(looks_eff), dtype=np.int64)
    early_succ_by_look = np.zeros(len(looks_eff), dtype=np.int64)
    early_fut_by_look = np.zeros(len(looks_eff), dtype=np.int64)

    if run_in_eff > 0:
        cum_x[active] += np.sum(X[active, :run_in_eff], axis=1)
        n_curr = run_in_eff

    for li, look_n in enumerate(looks_eff):
        add = look_n - n_curr
        if add > 0:
            cum_x[active] += np.sum(X[active, n_curr:look_n], axis=1)
            n_curr = look_n

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
    early_succ_rate = early_succ_by_look.sum() / n_sims
    early_fut_rate = early_fut_by_look.sum() / n_sims
    early_stop_rate = early_succ_rate + early_fut_rate
    unique_ns, counts = np.unique(final_n, return_counts=True)
    stop_dist = pd.DataFrame({"N_stop": unique_ns, "Probability": counts / n_sims}).sort_values("N_stop")
    return {
        "reject_rate": float(reject_rate),
        "ess": float(ess),
        "stop_probs_by_look_eff": (stop_by_look_counts / n_sims).tolist(),
        "stop_dist": stop_dist,
        "early_succ_by_look_eff": (early_succ_by_look / n_sims).tolist(),
        "early_fut_by_look_eff": (early_fut_by_look / n_sims).tolist(),
        "early_succ_rate": float(early_succ_rate),
        "early_fut_rate": float(early_fut_rate),
        "early_stop_rate": float(early_stop_rate),
    }


def simulate_design_joint(design: Dict, p_eff: float, p_tox: float, U_eff: np.ndarray, U_tox: np.ndarray) -> Dict:
    N = design["N_total"]
    looks_eff = design["looks_eff"]
    looks_saf = design.get("looks_saf", [])
    a0 = design["a0"]; b0 = design["b0"]
    p0 = design["p0"]; theta_final = design["theta_final"]
    theta_interim = design.get("theta_interim", theta_final)
    c_fut = design["c_futility"]
    allow_early = design["allow_early_success"]
    s_min_final = design["s_min_final"]
    x_min_to_continue = design["x_min_to_continue_by_look_eff"]
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

    eff_early_succ_by_look = np.zeros(len(looks_eff), dtype=np.int64)
    eff_early_fut_by_look = np.zeros(len(looks_eff), dtype=np.int64)
    saf_stop_by_look = np.zeros(len(looks_saf) + 1, dtype=np.int64)

    events = sorted(list(dict.fromkeys(list(looks_eff) + list(looks_saf) + [N])))

    for ev in events:
        if ev > n_curr and active.any():
            cum_x[active] += np.sum(X_eff[active, n_curr:ev], axis=1)
            if tox is not None:
                cum_t[active] += np.sum(X_tox[active, n_curr:ev], axis=1)
            n_curr = ev

        if tox is not None and ev in looks_saf and active.any():
            a_t = tox["a_t0"] + (cum_t[active] if cum_t is not None else 0)
            b_t = tox["b_t0"] + (n_curr - (cum_t[active] if cum_t is not None else 0))
            prob_tox_high = 1.0 - beta.cdf(tox["q_max"], a_t, b_t)
            unsafe = (prob_tox_high >= tox["theta_tox"])
            if np.any(unsafe):
                idx_active = np.where(active)[0]
                idx = idx_active[unsafe]
                final_n[idx] = n_curr
                active[idx] = False
                li_s = looks_saf.index(ev)
                saf_stop_by_look[li_s] += unsafe.sum()
            if not active.any():
                break

        if ev in looks_eff and active.any():
            if allow_early:
                a_post, b_post = beta_posterior_params(a0, b0, cum_x[active], n_curr)
                post_probs = 1.0 - beta.cdf(p0, a_post, b_post)
                early_succ = (post_probs >= theta_interim)
                if np.any(early_succ):
                    idx_active = np.where(active)[0]
                    idx = idx_active[early_succ]
                    success[idx] = True
                    final_n[idx] = n_curr
                    active[idx] = False
                    li_e = looks_eff.index(ev)
                    eff_early_succ_by_look[li_e] += early_succ.sum()
                if not active.any():
                    break

            x_min = x_min_to_continue.get(ev, None)
            li_e = looks_eff.index(ev)
            if x_min is None:
                idx = np.where(active)[0]
                if idx.size > 0:
                    final_n[idx] = n_curr
                    active[idx] = False
                    eff_early_fut_by_look[li_e] += idx.size
            else:
                need_continue = cum_x[active] >= x_min
                idx_all_active = np.where(active)[0]
                idx_stop = idx_all_active[~need_continue]
                if idx_stop.size > 0:
                    final_n[idx_stop] = n_curr
                    active[idx_stop] = False
                    eff_early_fut_by_look[li_e] += idx_stop.size
            if not active.any():
                break

    if active.any():
        if n_curr < N:
            cum_x[active] += np.sum(X_eff[active, n_curr:N], axis=1)
            if tox is not None:
                cum_t[active] += np.sum(X_tox[active, n_curr:N], axis=1)
            n_curr = N

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
                saf_stop_by_look[-1] += unsafe.sum()

        if active.any():
            succ_final = (cum_x[active] >= s_min_final)
            idx_active = np.where(active)[0]
            success[idx_active[succ_final]] = True
            final_n[idx_active] = N

    reject_rate = success.mean()
    ess = final_n.mean()

    n_sims = U_eff.shape[0]
    eff_early_succ_rate = eff_early_succ_by_look.sum() / n_sims
    eff_early_fut_rate = eff_early_fut_by_look.sum() / n_sims
    saf_early_rate = saf_stop_by_look[:-1].sum() / n_sims
    any_safety_rate = saf_stop_by_look.sum() / n_sims
    early_stop_rate = eff_early_succ_rate + eff_early_fut_rate + saf_early_rate

    unique_ns, counts = np.unique(final_n, return_counts=True)
    stop_dist = pd.DataFrame({"N_stop": unique_ns, "Probability": counts / n_sims}).sort_values("N_stop")

    return {
        "reject_rate": float(reject_rate),
        "ess": float(ess),
        "safety_stop_prob": float(any_safety_rate),
        "stop_dist": stop_dist,
        "eff_early_succ_by_look": (eff_early_succ_by_look / n_sims).tolist(),
        "eff_early_fut_by_look": (eff_early_fut_by_look / n_sims).tolist(),
        "saf_by_look": (saf_stop_by_look / n_sims).tolist(),
        "eff_early_succ_rate": float(eff_early_succ_rate),
        "eff_early_fut_rate": float(eff_early_fut_rate),
        "saf_early_rate": float(saf_early_rate),
        "early_stop_rate": float(early_stop_rate),
    }

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ SCREENING PIPELINE                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def shortlist_designs(param_grid: List[Dict], n_sims_small: int, seed: int, U: Optional[np.ndarray] = None):
    rng = np.random.default_rng(seed)
    if U is None:
        Nmax = max([g["N_total"] for g in param_grid])
        U = rng.uniform(size=(n_sims_small, Nmax))
    rows = []
    designs_built = []

    for g in param_grid:
        N = g["N_total"]
        looks_eff = g["looks_eff"]
        a0 = g["a0"]; b0 = g["b0"]
        p0 = g["p0"]; p1 = g["p1"]
        theta_final = g["theta_final"]; c_futility = g["c_futility"]
        theta_interim = g.get("theta_interim", theta_final)
        allow_early = g["allow_early_success"]

        s_min = min_successes_for_posterior_threshold(a0, b0, N, p0, theta_final)
        if s_min is None:
            continue
        x_min_to_continue = compute_interim_futility_cutoffs(a0, b0, N, looks_eff, p0, theta_final, c_futility)
        design = dict(
            N_total=N, looks_eff=looks_eff, a0=a0, b0=b0, p0=p0,
            theta_final=theta_final, theta_interim=theta_interim,
            c_futility=c_futility, allow_early_success=allow_early,
            s_min_final=s_min, x_min_to_continue_by_look_eff=x_min_to_continue,
            p1=p1,
            run_in_eff=g.get("run_in_eff", 0)
        )
        res_p0 = simulate_design_eff_only(design, p0, U[:, :N])
        res_p1 = simulate_design_eff_only(design, p1, U[:, :N])

        rows.append({
            "N_total": N,
            "looks_eff": looks_eff,
            "run_in_eff": design["run_in_eff"],
            "theta_final": theta_final,
            "theta_interim": theta_interim,
            "c_futility": c_futility,
            "allow_early_success": allow_early,
            "Type I error @ p0": res_p0["reject_rate"],
            "Power @ p1": res_p1["reject_rate"],
            "ESS @ p0": res_p0["ess"],
            "ESS @ p1": res_p1["ess"],
            "Early stop @ p0 (any)": res_p0["early_stop_rate"],
            "Early success @ p0": res_p0["early_succ_rate"],
            "Early futility @ p0": res_p0["early_fut_rate"],
            "s_min_final": s_min,
            "x_min_to_continue_eff": x_min_to_continue,
            "_design_eff_only": design,
            "_eff_bylook_succ_p0": res_p0["early_succ_by_look_eff"],
            "_eff_bylook_fut_p0": res_p0["early_fut_by_look_eff"],
        })
        designs_built.append(design)

    df = pd.DataFrame(rows)
    return df, designs_built

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ PLOTTING                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_lines(df: pd.DataFrame, x: str, y: str, title: str):
    if _HAS_PLOTLY:
        fig = px.line(df, x=x, y=y, title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader(title)
        st.line_chart(df[[x, y]].set_index(x))

def plot_stacked_bylook(labels: List[str], stacks: Dict[str, List[float]], title: str):
    df = pd.DataFrame({"look": labels, **stacks})
    if _HAS_PLOTLY:
        fig = go.Figure()
        for k, v in stacks.items():
            fig.add_bar(x=df["look"], y=v, name=k)
        fig.update_layout(barmode="stack", title=title, xaxis_title="Interim look (n)", yaxis_title="Fraction of trials")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader(title)
        st.dataframe(df.set_index("look"))

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ STREAMLIT UI                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="Bayesian Single‑Arm Designer (Binary) — Split Schedules + Tuner + OC + Export",
    layout="wide"
)

st.title("Bayesian Single‑Arm Monitored Study Designer (Binary Endpoint)")
st.caption("Design single‑arm trials with Bayesian interim monitoring for **efficacy** and **safety**. Now with **separate run‑ins**, **split schedules**, a **Threshold Tuner**, a **Comprehensive OC Explorer**, and **Export**.")

with st.expander("What this tool does (in simple terms)"):
    st.markdown(
        """
**Run‑in** delays the first interim but all run‑in patients are included in every posterior and in the final analysis. Run‑in is treated as **Look #1** for that domain. Threshold Tuner helps you find **θ_final** and **θ_interim** that meet α/power goals and optimize ESS/early stops. OC Explorer evaluates the design across a grid of (p,q) scenarios.
        """
    )

# ── Sidebar: inputs (keys added so Tuner can apply values) ─────────────────
st.sidebar.header("1) Set your efficacy targets")
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    p0 = st.number_input(
        "Null response rate (p₀)",
        min_value=0.0, max_value=1.0, value=st.session_state.get("p0", 0.20), step=0.01, format="%.2f", key="p0",
        help="Highest response rate still considered clinically unacceptable."
    )
    a0 = st.number_input(
        "Prior a₀ (efficacy)",
        min_value=0.0, value=st.session_state.get("a0", 1.0), step=0.5, key="a0",
        help="Efficacy prior Beta(a₀, b₀). Use (1,1) for uninformative. Increase a₀+b₀ to strengthen prior."
    )
    theta_final = st.number_input(
        "Final success threshold (θ_final)",
        min_value=0.5, max_value=0.999, value=st.session_state.get("theta_final", 0.95), step=0.01, format="%.3f", key="theta_final",
        help="At the final look, declare success if P(p>p₀ | data) ≥ θ_final."
    )
with col_sb2:
    p1 = st.number_input(
        "Target response rate (p₁)",
        min_value=0.0, max_value=1.0, value=st.session_state.get("p1", 0.40), step=0.01, format="%.2f", key="p1",
        help="A rate you consider clinically promising; power is evaluated here."
    )
    b0 = st.number_input(
        "Prior b₀ (efficacy)",
        min_value=0.0, value=st.session_state.get("b0", 1.0), step=0.5, key="b0",
        help="Efficacy prior Beta(a₀, b₀). Mean=a₀/(a₀+b₀); ESS≈a₀+b₀."
    )
    c_futility = st.number_input(
        "Futility cutoff (PPoS)",
        min_value=0.0, max_value=0.5, value=st.session_state.get("c_futility", 0.05), step=0.01, format="%.3f", key="c_futility",
        help="At interims, stop if the predictive probability of *final* success is below this."
    )
with st.sidebar.expander("About your efficacy prior"):
    if (a0 + b0) > 0:
        st.write(f"• Prior mean = **{a0/(a0+b0):.3f}**, prior ESS ≈ **{a0+b0:.1f}**.")
    else:
        st.write("• Prior mean undefined (a₀+b₀=0). Consider Beta(1,1).")

allow_early_success = st.sidebar.checkbox(
    "Allow early success at interim looks?",
    value=st.session_state.get("allow_early_success", False), key="allow_early_success",
    help="If checked, at interims stop early for success when P(p>p₀ | data) ≥ θ_interim."
)

theta_interim = st.sidebar.number_input(
    "Interim success threshold (θ_interim)",
    min_value=0.5, max_value=0.999, value=float(st.session_state.get("theta_interim", st.session_state.get("theta_final", 0.95))), step=0.01, format="%.3f", key="theta_interim",
    help="Recommended ≥ θ_final for stricter early success (used only if early success is enabled)."
)

# ── Safety monitoring ───────────────────────────────────────────────────────
st.sidebar.header("2) Safety monitoring (optional)")
enable_safety = st.sidebar.checkbox(
    "Enable safety (toxicity) monitoring?",
    value=st.session_state.get("enable_safety", True), key="enable_safety",
    help="Monitor an SAE rate with a Beta prior; stop if it likely exceeds q_max."
)
if enable_safety:
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        a_t0 = st.number_input(
            "Safety prior a_t0", min_value=0.0, value=st.session_state.get("a_t0", 1.0), step=0.5, key="a_t0",
            help="Safety prior Beta(a_t0, b_t0). Mean=a_t0/(a_t0+b_t0); ESS≈a_t0+b_t0."
        )
        q_max = st.number_input(
            "Max acceptable SAE (q_max)", min_value=0.0, max_value=1.0, value=st.session_state.get("q_max", 0.15), step=0.01, format="%.2f", key="q_max",
            help="If P(q>q_max | data) is high, the study stops for safety."
        )
    with col_s2:
        b_t0 = st.number_input(
            "Safety prior b_t0", min_value=0.0, value=st.session_state.get("b_t0", 9.0), step=0.5, key="b_t0",
            help="Increase a_t0+b_t0 for stronger prior. Beta(1,9) ⇒ mean ~0.10."
        )
        theta_tox = st.number_input(
            "Safety stop threshold (θ_tox)",
            min_value=0.5, max_value=0.999, value=st.session_state.get("theta_tox", 0.90), step=0.01, format="%.3f", key="theta_tox",
            help="Stop when P(q>q_max | data) ≥ θ_tox."
        )
    with st.sidebar.expander("About your safety prior"):
        if (a_t0 + b_t0) > 0:
            st.write(f"• Prior mean = **{a_t0/(a_t0+b_t0):.3f}**, prior ESS ≈ **{a_t0+b_t0:.1f}**.")
        else:
            st.write("• Prior mean undefined (a_t0+b_t0=0). Consider Beta(1,9) for ~10% prior mean.")
else:
    a_t0 = b_t0 = q_max = theta_tox = None

# ── Look schedules (separate) ───────────────────────────────────────────────
st.sidebar.header("3) Efficacy interim look schedule")
run_in_eff = st.sidebar.number_input(
    "Efficacy run‑in (first look)", min_value=0, max_value=400, value=st.session_state.get("run_in_eff", 0), step=1, key="run_in_eff",
    help="First Rₑ patients accrued before any efficacy interims are evaluated. Rₑ counts toward all posteriors."
)
looks_eff_mode = st.sidebar.selectbox(
    "Efficacy look timing",
    options=["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"],
    index=st.session_state.get("looks_eff_mode_idx", 1), key="looks_eff_mode_idx",
    help="Run‑in is treated as the first look when Rₑ>0."
)

k_looks_eff = perc_eff_str = ns_eff_str = None
if ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"][looks_eff_mode] == "Equal‑spaced (choose total looks incl. run‑in)":
    k_looks_eff = st.sidebar.slider(
        "Total number of efficacy looks (incl. run‑in if >0)",
        min_value=1, max_value=8, value=st.session_state.get("k_looks_eff", 2), step=1, key="k_looks_eff",
        help="If Rₑ>0, Look#1 is at Rₑ; remaining K−1 are spaced on N−Rₑ."
    )
elif ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"][looks_eff_mode] == "Custom percentages of remaining":
    perc_eff_str = st.sidebar.text_input(
        "Efficacy percentages of remaining (comma‑separated)", value=st.session_state.get("perc_eff_str", "33,67"), key="perc_eff_str",
        help="Example: 25,50,75 (applied to remaining N after run‑in)."
    )
elif ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"][looks_eff_mode] == "Custom absolute Ns":
    ns_eff_str = st.sidebar.text_input(
        "Efficacy interim sample sizes N (comma‑separated)", value=st.session_state.get("ns_eff_str", ""), key="ns_eff_str",
        help="Absolute Ns. Run‑in (if >0) acts as first look; Ns ≤ run‑in are ignored."
    )

st.sidebar.header("4) Safety interim look schedule (independent)")
run_in_saf = st.sidebar.number_input(
    "Safety run‑in (first look)", min_value=0, max_value=400, value=st.session_state.get("run_in_saf", 0), step=1, key="run_in_saf",
    help="First Rₛ patients accrued before any safety interims are evaluated. Rₛ counts toward all posteriors."
)
looks_saf_mode = st.sidebar.selectbox(
    "Safety look timing",
    options=["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"],
    index=st.session_state.get("looks_saf_mode_idx", 1), key="looks_saf_mode_idx",
    help="Run‑in is treated as the first look when Rₛ>0."
)

k_looks_saf = perc_saf_str = ns_saf_str = None
if ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"][looks_saf_mode] == "Equal‑spaced (choose total looks incl. run‑in)":
    k_looks_saf = st.sidebar.slider(
        "Total number of safety looks (incl. run‑in if >0)",
        min_value=1, max_value=8, value=st.session_state.get("k_looks_saf", 2), step=1, key="k_looks_saf",
        help="If Rₛ>0, Look#1 is at Rₛ; remaining K−1 are spaced on N−Rₛ."
    )
elif ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"][looks_saf_mode] == "Custom percentages of remaining":
    perc_saf_str = st.sidebar.text_input(
        "Safety percentages of remaining (comma‑separated)", value=st.session_state.get("perc_saf_str", "33,67"), key="perc_saf_str",
        help="Applied to the remaining N after safety run‑in."
    )
elif ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"][looks_saf_mode] == "Custom absolute Ns":
    ns_saf_str = st.sidebar.text_input(
        "Safety interim sample sizes N (comma‑separated)", value=st.session_state.get("ns_saf_str", ""), key="ns_saf_str",
        help="Absolute Ns. Run‑in (if >0) acts as first look; Ns ≤ run‑in are ignored."
    )

# ── Screening grid ──────────────────────────────────────────────────────────
st.sidebar.header("5) Rapid Screener settings")
N_min, N_max = st.sidebar.slider(
    "Range of maximum sample sizes to test",
    min_value=10, max_value=400, value=st.session_state.get("N_range", (30, 120)), step=1, key="N_range",
    help="We'll scan through this range to find feasible, efficient efficacy designs."
)
N_step = st.sidebar.number_input(
    "Step size for N grid",
    min_value=1, max_value=50, value=st.session_state.get("N_step", 5), step=1, key="N_step",
    help="Larger steps scan fewer N values (faster)."
)
n_sims_small = st.sidebar.number_input(
    "Screening simulations per design",
    min_value=100, max_value=200000, value=st.session_state.get("n_sims_small", 5000), step=500, key="n_sims_small",
    help="Higher = more precision; deep dive is where precision matters most."
)
alpha_max = st.sidebar.number_input(
    "Max Type I error (α) allowed",
    min_value=0.0, max_value=0.5, value=st.session_state.get("alpha_max", 0.10), step=0.01, format="%.2f", key="alpha_max",
    help="Keep only designs with false‑positive rate ≤ this."
)
power_min = st.sidebar.number_input(
    "Min power at p₁",
    min_value=0.0, max_value=1.0, value=st.session_state.get("power_min", 0.80), step=0.01, format="%.2f", key="power_min",
    help="Keep only designs with success rate at p₁ ≥ this."
)
seed = st.sidebar.number_input(
    "Random seed",
    min_value=1, value=st.session_state.get("seed", 2026), step=1, key="seed",
    help="Controls reproducibility."
)

# Build look schedules
Ns = list(range(N_min, N_max + 1, N_step))

# decode selectbox indices back to labels
_eff_modes = ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"]
_saf_modes = ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"]

looks_eff_mode_label = _eff_modes[st.session_state.get("looks_eff_mode_idx", 1)]
looks_saf_mode_label = _saf_modes[st.session_state.get("looks_saf_mode_idx", 1)]

param_grid = []
for N in Ns:
    looks_eff = build_looks_with_runin(
        N=N, run_in=run_in_eff, mode=looks_eff_mode_label,
        k_total=st.session_state.get("k_looks_eff", None), perc_str=st.session_state.get("perc_eff_str", None), ns_str=st.session_state.get("ns_eff_str", None)
    )
    param_grid.append({
        "N_total": N,
        "looks_eff": looks_eff,
        "a0": a0, "b0": b0, "p0": p0, "p1": p1,
        "theta_final": theta_final, "theta_interim": float(theta_interim),
        "c_futility": c_futility, "allow_early_success": allow_early_success,
        "run_in_eff": int(run_in_eff)
    })

# Helper safe cols

def _safe_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

# ── Rapid Screener ──────────────────────────────────────────────────────────
st.write("### 1) Rapid Screener (efficacy-only)")
st.caption("We evaluate many candidate designs (efficacy‑only) and keep those meeting α/power criteria. Outputs include **early-stop rates** under p₀.")

@st.cache_data(show_spinner=False)
def _screen(param_grid, n_sims_small, seed, schema_version: str):
    rng = np.random.default_rng(seed)
    Nmax = max([g["N_total"] for g in param_grid])
    U = rng.uniform(size=(n_sims_small, Nmax))
    df, designs = shortlist_designs(param_grid, n_sims_small, seed, U)
    _ = schema_version
    return df, designs

df_screen, designs_built = _screen(param_grid, n_sims_small, seed, SCHEMA_VERSION)
if df_screen.empty:
    st.warning("No viable designs found (final rule may be impossible). Try relaxing θ_final or expanding N.")
else:
    df_ok = df_screen[(df_screen["Type I error @ p0"] <= alpha_max) & (df_screen["Power @ p1"] >= power_min)].copy()
    cols_to_show = ["N_total", "run_in_eff", "looks_eff", "theta_final", "c_futility", "Type I error @ p0", "Power @ p1", "ESS @ p0", "Early stop @ p0 (any)", "Early success @ p0", "Early futility @ p0"]
    if df_ok.empty:
        st.info("No designs met both α and power. Showing the full screening table instead (for inspection).")
        st.dataframe(df_screen[_safe_cols(df_screen, cols_to_show)].sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True))
        df_to_select_from = df_screen.sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True)
    else:
        df_ranked = df_ok.sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True)
        st.dataframe(df_ranked[_safe_cols(df_ranked, cols_to_show)].head(15))
        st.success(f"Found {len(df_ok)} candidates meeting your criteria. Showing top 15 ranked by low ESS @ p₀.")
        df_to_select_from = df_ranked

    # ── Deep Dive (enter N) ───────────────────────────────────────────────
    st.write("### 2) Deep Dive on a Design (enter N)")
    st.caption("We run **joint** simulations (efficacy + optional safety) for precise operating characteristics and early‑stop breakdowns.")

    N_select = st.number_input("Enter an N (max sample size) to deep‑dive", min_value=Ns[0], max_value=Ns[-1], value=st.session_state.get("N_select", max(60, Ns[0])), step=1, key="N_select")

    looks_eff_sel = build_looks_with_runin(N_select, run_in_eff, looks_eff_mode_label, k_total=st.session_state.get("k_looks_eff", None), perc_str=st.session_state.get("perc_eff_str", None), ns_str=st.session_state.get("ns_eff_str", None))
    looks_saf_sel = build_looks_with_runin(N_select, run_in_saf, looks_saf_mode_label, k_total=st.session_state.get("k_looks_saf", None), perc_str=st.session_state.get("perc_saf_str", None), ns_str=st.session_state.get("ns_saf_str", None))

    s_min_sel = min_successes_for_posterior_threshold(a0, b0, N_select, p0, theta_final)
    if s_min_sel is None:
        st.error("The final success rule is infeasible at this N. Relax θ_final, adjust prior, or increase N.")
    else:
        x_min_to_continue_sel = compute_interim_futility_cutoffs(a0, b0, N_select, looks_eff_sel, p0, theta_final, c_futility)
        design_sel = dict(N_total=N_select, a0=a0, b0=b0, p0=p0, p1=p1, theta_final=theta_final, theta_interim=float(theta_interim), c_futility=c_futility, allow_early_success=allow_early_success, s_min_final=s_min_sel, looks_eff=looks_eff_sel, looks_saf=looks_saf_sel, run_in_eff=int(run_in_eff), run_in_saf=int(run_in_saf), x_min_to_continue_by_look_eff=x_min_to_continue_sel)
        if enable_safety:
            design_sel["safety"] = dict(a_t0=a_t0, b_t0=b_t0, q_max=q_max, theta_tox=theta_tox)

        with st.expander("Chosen Design Summary", expanded=True):
            st.subheader("Final Rule")
            st.write(f"• **N_total**: {design_sel['N_total']}")
            st.write(f"• **Final success threshold θ_final**: {design_sel['theta_final']}")
            st.write(f"• **Required successes at final look (s_min_final)**: {design_sel['s_min_final']}")

            st.subheader("Interim Rules — Efficacy")
            st.write(f"• **Run‑in (eff)**: {design_sel['run_in_eff']} → Look #1")
            st.write(f"• **Efficacy looks at N =** {design_sel['looks_eff']}")
            st.write(f"• **Early success allowed:** {design_sel['allow_early_success']}")
            st.write(f"• **θ_interim:** {design_sel['theta_interim']}")
            st.write("• **Continue thresholds** (need ≥ x cures at look n to continue):")
            st.dataframe(pd.DataFrame.from_dict(design_sel["x_min_to_continue_by_look_eff"], orient="index", columns=["x ≥ to continue"]))

            if enable_safety:
                st.subheader("Interim Rules — Safety")
                st.write(f"• **Run‑in (safety)**: {design_sel['run_in_saf']} → Look #1")
                st.write(f"• **Safety looks at N =** {design_sel['looks_saf']}")
                st.write(f"• **Safety rule:** stop if P(q>q_max | data) ≥ θ_tox")
                st.write(f"• **q_max / θ_tox:** {q_max:.2f} / {theta_tox:.3f}")
                st.write("• Safety prior Beta(a_t0, b_t0): " f"{a_t0 if a_t0 is not None else '-'}, {b_t0 if b_t0 is not None else '-'}")

            show_code = st.checkbox("Show underlying Python design dict", value=False)
            if show_code:
                st.code(repr(design_sel), language="python")

        st.write("#### Deep‑dive simulation settings")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            n_sims_deep = st.number_input("Simulations to run", min_value=2000, max_value=800000, value=st.session_state.get("n_sims_deep", 150000), step=5000, key="n_sims_deep")
        with col_d2:
            seed_deep = st.number_input("Random seed (deep dive)", min_value=1, value=st.session_state.get("seed_deep", seed + 1), step=1, key="seed_deep")

        st.write("#### Toxicity scenarios (for joint analysis)")
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            q_good = st.number_input("q_good (typical/benign toxicity)", min_value=0.0, max_value=1.0, value=st.session_state.get("q_good", 0.10), step=0.01, key="q_good")
        with col_q2:
            q_bad = st.number_input("q_bad (concerning/high toxicity)", min_value=0.0, max_value=1.0, value=st.session_state.get("q_bad", 0.20), step=0.01, key="q_bad")
        with col_q3:
            q_for_OC = st.number_input("Fixed q for OC/ESS curves", min_value=0.0, max_value=1.0, value=st.session_state.get("q_for_OC", 0.10), step=0.01, key="q_for_OC")

        if st.button("Run DEEP‑DIVE simulation (efficacy + safety)", key="btn_deep"):
            rng = np.random.default_rng(seed_deep)
            Ueff = rng.uniform(size=(n_sims_deep, design_sel["N_total"]))
            Utox = rng.uniform(size=(n_sims_deep, design_sel["N_total"]))

            res_p0_qgood = simulate_design_joint(design_sel, p_eff=p0, p_tox=q_good, U_eff=Ueff, U_tox=Utox)
            res_p1_qgood = simulate_design_joint(design_sel, p_eff=p1, p_tox=q_good, U_eff=Ueff, U_tox=Utox)
            res_p1_qbad  = simulate_design_joint(design_sel, p_eff=p1, p_tox=q_bad,  U_eff=Ueff, U_tox=Utox)

            # Store for export
            st.session_state["deep_results"] = dict(res_p0_qgood=res_p0_qgood, res_p1_qgood=res_p1_qgood, res_p1_qbad=res_p1_qbad)
            st.session_state["deep_design"] = design_sel

            st.write("##### Point estimates (joint, with safety if enabled)")
            cols = st.columns(6 if enable_safety else 5)
            cols[0].metric("Type I @ p₀, q_good", f"{res_p0_qgood['reject_rate']:.3f}")
            cols[1].metric("Power @ p₁, q_good", f"{res_p1_qgood['reject_rate']:.3f}")
            cols[2].metric("ESS @ p₀, q_good", f"{res_p0_qgood['ess']:.1f}")
            cols[3].metric("ESS @ p₁, q_good", f"{res_p1_qgood['ess']:.1f}")
            cols[4].metric("Early stop @ p₁ (any)", f"{res_p1_qgood['early_stop_rate']:.3f}")
            if enable_safety:
                cols[5].metric("P(Safety stop) @ p₁, q_bad", f"{res_p1_qbad['safety_stop_prob']:.3f}")

            cols2 = st.columns(4 if enable_safety else 3)
            cols2[0].metric("… early success @ p₁", f"{res_p1_qgood['eff_early_succ_rate']:.3f}")
            cols2[1].metric("… early futility @ p₁", f"{res_p1_qgood['eff_early_fut_rate']:.3f}")
            if enable_safety:
                cols2[2].metric("… early safety @ p₁", f"{res_p1_qgood['saf_early_rate']:.3f}")
                cols2[3].metric("Early stop (any) @ p₁", f"{res_p1_qgood['early_stop_rate']:.3f}")
            else:
                cols2[2].metric("Early stop (any) @ p₁", f"{res_p1_qgood['early_stop_rate']:.3f}")

            st.write("###### By‑look details @ p₁, q_good (fractions of trials)")
            if len(looks_eff_sel) > 0:
                lbl_eff = [str(n) for n in looks_eff_sel]
                plot_stacked_bylook(labels=lbl_eff, stacks={"Early success (eff)": res_p1_qgood["eff_early_succ_by_look"], "Early futility (eff)": res_p1_qgood["eff_early_fut_by_look"]}, title="Efficacy early decisions by interim look")
            if enable_safety and len(looks_saf_sel) > 0:
                lbl_saf = [str(n) for n in looks_saf_sel]
                saf_arr = res_p1_qgood["saf_by_look"]
                saf_interims = saf_arr[:-1] if len(saf_arr) == len(looks_saf_sel) + 1 else saf_arr
                plot_stacked_bylook(labels=lbl_saf, stacks={"Early safety stops": saf_interims}, title="Safety early stops by interim look")

            st.write("##### Sample‑size distribution")
            st.write("At the null (p₀, q_good):")
            st.dataframe(res_p0_qgood["stop_dist"]) 
            st.write("At the target (p₁, q_good):")
            st.dataframe(res_p1_qgood["stop_dist"]) 

            st.write("##### OC & ESS vs efficacy p (toxicity held fixed)")
            p_grid_min = st.number_input("OC curve p‑min", min_value=0.0, max_value=1.0, value=max(0.0, p0 - 0.15), step=0.01, key="p_grid_min")
            p_grid_max = st.number_input("OC curve p‑max", min_value=0.0, max_value=1.0, value=min(1.0, p1 + 0.20), step=0.01, key="p_grid_max")
            n_grid = st.slider("Number of points on the grid", min_value=5, max_value=40, value=15, step=1, key="n_grid")
            ps = np.linspace(p_grid_min, p_grid_max, n_grid)
            oc, ess_curve = [], []
            for pp in ps:
                r = simulate_design_joint(design_sel, p_eff=pp, p_tox=q_for_OC, U_eff=Ueff, U_tox=Utox)
                oc.append(r["reject_rate"])
                ess_curve.append(r["ess"])
            df_oc = pd.DataFrame({"p": ps, "Reject_Prob": oc, "ESS": ess_curve})
            plot_lines(df_oc, x="p", y="Reject_Prob", title=f"Operating Characteristic vs p (toxicity fixed at q={q_for_OC:.2f})")
            plot_lines(df_oc, x="p", y="ESS", title=f"Expected sample size vs p (toxicity fixed at q={q_for_OC:.2f})")

    # ── Compare panel ───────────────────────────────────────────────────────
    st.write("### 3) Compare multiple Ns (deep‑dive, joint with safety)")
    with st.expander("Open compare panel"):
        ns_str_compare = st.text_input("Enter N values (comma-separated)", value=st.session_state.get("ns_str_compare", "60,70,80"), key="ns_str_compare")
        n_sims_compare = st.number_input("Simulations per design (compare run)", min_value=5000, max_value=400000, value=st.session_state.get("n_sims_compare", 80000), step=5000, key="n_sims_compare")
        seed_compare = st.number_input("Random seed (compare run)", min_value=1, value=st.session_state.get("seed_compare", seed + 2), step=1, key="seed_compare")
        q_good_cmp = st.number_input("q_good for compare run", min_value=0.0, max_value=1.0, value=st.session_state.get("q_good_cmp", 0.10), step=0.01, key="q_good_cmp")
        q_bad_cmp = st.number_input("q_bad for compare run", min_value=0.0, max_value=1.0, value=st.session_state.get("q_bad_cmp", 0.20), step=0.01, key="q_bad_cmp")

        def _build_eff(N):
            return build_looks_with_runin(N, run_in_eff, looks_eff_mode_label, k_total=st.session_state.get("k_looks_eff", None), perc_str=st.session_state.get("perc_eff_str", None), ns_str=st.session_state.get("ns_eff_str", None))
        def _build_saf(N):
            return build_looks_with_runin(N, run_in_saf, looks_saf_mode_label, k_total=st.session_state.get("k_looks_saf", None), perc_str=st.session_state.get("perc_saf_str", None), ns_str=st.session_state.get("ns_saf_str", None))

        if st.button("Run compare", key="btn_compare"):
            Ns_cmp: List[int] = []
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
                dfs = []
                for Ncmp in Ns_cmp:
                    looks_eff_cmp = _build_eff(Ncmp)
                    looks_saf_cmp = _build_saf(Ncmp)
                    smin_cmp = min_successes_for_posterior_threshold(a0, b0, Ncmp, p0, theta_final)
                    if smin_cmp is None:
                        continue
                    x_min_cmp = compute_interim_futility_cutoffs(a0, b0, Ncmp, looks_eff_cmp, p0, theta_final, c_futility)
                    design_cmp = dict(N_total=Ncmp, looks_eff=looks_eff_cmp, looks_saf=looks_saf_cmp, run_in_eff=int(run_in_eff), run_in_saf=int(run_in_saf), a0=a0, b0=b0, p0=p0, theta_final=theta_final, theta_interim=float(theta_interim), c_futility=c_futility, allow_early_success=allow_early_success, s_min_final=smin_cmp, x_min_to_continue_by_look_eff=x_min_cmp)
                    if enable_safety:
                        design_cmp["safety"] = dict(a_t0=a_t0, b_t0=b_t0, q_max=q_max, theta_tox=theta_tox)
                    Ueff = rng.uniform(size=(n_sims_compare, Ncmp))
                    Utox = rng.uniform(size=(n_sims_compare, Ncmp))
                    r_p0 = simulate_design_joint(design_cmp, p_eff=p0, p_tox=q_good_cmp, U_eff=Ueff, U_tox=Utox)
                    r_p1 = simulate_design_joint(design_cmp, p_eff=p1, p_tox=q_good_cmp, U_eff=Ueff, U_tox=Utox)
                    r_p1_bad = simulate_design_joint(design_cmp, p_eff=p1, p_tox=q_bad_cmp, U_eff=Ueff, U_tox=Utox)
                    row = dict(N=Ncmp, looks_eff=looks_eff_cmp, looks_saf=looks_saf_cmp, s_min_final=smin_cmp, **{"Type I @p0,q_good": r_p0["reject_rate"], "Power @p1,q_good": r_p1["reject_rate"], "ESS @p0": r_p0["ess"], "ESS @p1": r_p1["ess"], "Early stop (any) @p1": r_p1["early_stop_rate"], "Early succ @p1": r_p1["eff_early_succ_rate"], "Early fut @p1": r_p1["eff_early_fut_rate"], "Early safety @p1": r_p1.get("saf_early_rate", 0.0), "Safety stop @p1,q_bad (any stage)": r_p1_bad.get("safety_stop_prob", 0.0)})
                    results.append(row)
                if not results:
                    st.warning("None of the N values produced a feasible final rule. Try relaxing θ_final or adjusting looks.")
                else:
                    df_cmp = pd.DataFrame(results).sort_values("N").reset_index(drop=True)
                    st.dataframe(df_cmp)
                    st.session_state["compare_df"] = df_cmp
                    if _HAS_PLOTLY:
                        fig_pow = px.bar(df_cmp, x="N", y="Power @p1,q_good", title="Power by N")
                        st.plotly_chart(fig_pow, use_container_width=True)
                        fig_ess = px.bar(df_cmp, x="N", y="ESS @p1", title="ESS @ p₁ by N")
                        st.plotly_chart(fig_ess, use_container_width=True)

    # ── Threshold Tuner (new) ──────────────────────────────────────────────
    st.write("### 4) Threshold Tuner (find θ_final / θ_interim to meet α/power and optimize)")
    with st.expander("Open Threshold Tuner"):
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            tf_min = st.number_input("θ_final min", 0.50, 0.999, value=0.90, step=0.01, format="%.3f")
            tf_max = st.number_input("θ_final max", 0.50, 0.999, value=max(0.96, theta_final), step=0.01, format="%.3f")
            tf_step = st.number_input("θ_final step", 0.001, 0.1, value=0.01, step=0.001, format="%.3f")
        with col_t2:
            ti_min = st.number_input("θ_interim min", 0.50, 0.999, value=max(theta_final, 0.90), step=0.01, format="%.3f")
            ti_max = st.number_input("θ_interim max", 0.50, 0.999, value=0.995, step=0.01, format="%.3f")
            ti_step = st.number_input("θ_interim step", 0.001, 0.1, value=0.01, step=0.001, format="%.3f")
        with col_t3:
            n_sims_tuner = st.number_input("Sims per candidate", min_value=2000, max_value=200000, value=40000, step=2000)
            seed_tuner = st.number_input("Random seed (tuner)", min_value=1, value=seed + 3, step=1)
            objective = st.selectbox("Optimize for", options=["Min ESS @ p0", "Min ESS @ p1", "Max Power @ p1", "Max Early stops @ p0"], index=0)
            alpha_cap = st.number_input("Constraint: max α", 0.0, 0.5, value=alpha_max, step=0.01, format="%.2f")
            power_floor = st.number_input("Constraint: min power", 0.0, 1.0, value=power_min, step=0.01, format="%.2f")

        st.caption("We re‑use your current N and efficacy look schedule; safety has no effect on this tuner (efficacy‑only).")

        if st.button("Run Tuner", key="btn_tuner"):
            rng = np.random.default_rng(seed_tuner)
            U = rng.uniform(size=(n_sims_tuner, N_select))
            tf_vals = np.arange(tf_min, tf_max + 1e-9, tf_step)
            ti_vals = np.arange(ti_min, ti_max + 1e-9, ti_step)

            rows = []
            for tf in tf_vals:
                s_min = min_successes_for_posterior_threshold(a0, b0, N_select, p0, tf)
                if s_min is None:
                    continue
                x_min_map = compute_interim_futility_cutoffs(a0, b0, N_select, looks_eff_sel, p0, tf, c_futility)
                for ti in ti_vals:
                    design_tmp = dict(N_total=N_select, looks_eff=looks_eff_sel, a0=a0, b0=b0, p0=p0, theta_final=tf, theta_interim=float(ti), c_futility=c_futility, allow_early_success=allow_early_success, s_min_final=s_min, x_min_to_continue_by_look_eff=x_min_map, run_in_eff=run_in_eff)
                    r0 = simulate_design_eff_only(design_tmp, p0, U)
                    r1 = simulate_design_eff_only(design_tmp, p1, U)
                    if r0["reject_rate"] <= alpha_cap and r1["reject_rate"] >= power_floor:
                        obj = {
                            "Min ESS @ p0": r0["ess"],
                            "Min ESS @ p1": r1["ess"],
                            "Max Power @ p1": -r1["reject_rate"],  # negate for ascending sort
                            "Max Early stops @ p0": -r0["early_stop_rate"],
                        }[objective]
                        rows.append(dict(theta_final=tf, theta_interim=ti, type1=r0["reject_rate"], power=r1["reject_rate"], ESS_p0=r0["ess"], ESS_p1=r1["ess"], early_any_p0=r0["early_stop_rate"], early_succ_p0=r0["early_succ_rate"], early_fut_p0=r0["early_fut_rate"], objective=obj))
            if not rows:
                st.warning("No candidate thresholds met the α/power constraints in the search ranges.")
            else:
                df_tune = pd.DataFrame(rows).sort_values("objective").reset_index(drop=True)
                st.dataframe(df_tune.head(20))
                st.session_state["tuner_df"] = df_tune

                # Apply best to sidebar
                best = df_tune.iloc[0]
                if st.button("Apply best (θ_final, θ_interim) to sidebar", key="btn_apply_tuner"):
                    st.session_state["theta_final"] = float(best["theta_final"]) 
                    st.session_state["theta_interim"] = float(best["theta_interim"]) 
                    st.success(f"Applied θ_final={best['theta_final']:.3f}, θ_interim={best['theta_interim']:.3f} to sidebar.")

    # ── Comprehensive OC Explorer (new) ─────────────────────────────────────
    st.write("### 5) Comprehensive OC Explorer (grid over efficacy p and toxicity q)")
    with st.expander("Open OC Explorer"):
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            p_min = st.number_input("p_min", 0.0, 1.0, value=max(0.0, p0 - 0.20), step=0.01)
            p_max = st.number_input("p_max", 0.0, 1.0, value=min(1.0, p1 + 0.25), step=0.01)
            p_points = st.slider("# of p points", 3, 25, 9, 1)
        with col_o2:
            q_min = st.number_input("q_min", 0.0, 1.0, value=max(0.0, (q_max or 0.10) - 0.10), step=0.01)
            q_max_g = st.number_input("q_max (grid)", 0.0, 1.0, value=min(1.0, (q_max or 0.10) + 0.20), step=0.01)
            q_points = st.slider("# of q points", 3, 25, 7, 1)
        col_o3, col_o4 = st.columns(2)
        with col_o3:
            n_sims_oc = st.number_input("Sims per grid cell", min_value=2000, max_value=200000, value=50000, step=2000)
        with col_o4:
            seed_oc = st.number_input("Random seed (OC grid)", min_value=1, value=seed + 4, step=1)

        st.caption("We use your current design (N, thresholds, run‑ins, and look schedules).")

        if st.button("Run OC Explorer", key="btn_oc"):
            if s_min_sel is None:
                st.error("Deep‑dive design is infeasible — please adjust θ_final or N.")
            else:
                rng = np.random.default_rng(seed_oc)
                Ueff = rng.uniform(size=(n_sims_oc, N_select))
                Utox = rng.uniform(size=(n_sims_oc, N_select))
                ps = np.linspace(p_min, p_max, p_points)
                qs = np.linspace(q_min, q_max_g, q_points)

                oc_rows = []
                for pp in ps:
                    for qq in qs:
                        r = simulate_design_joint(design_sel, p_eff=pp, p_tox=qq, U_eff=Ueff, U_tox=Utox)
                        oc_rows.append(dict(p=pp, q=qq, reject_rate=r["reject_rate"], ess=r["ess"], safety_stop_any=r["safety_stop_prob"], early_any=r["early_stop_rate"], early_eff_succ=r["eff_early_succ_rate"], early_eff_fut=r["eff_early_fut_rate"], early_safety=r["saf_early_rate"]))
                df_grid = pd.DataFrame(oc_rows)
                st.dataframe(df_grid)
                st.session_state["oc_grid_df"] = df_grid

                if _HAS_PLOTLY:
                    # Heatmaps over p (x) and q (y)
                    def _heat(df, z, title):
                        piv = df.pivot(index="q", columns="p", values=z)
                        fig = px.imshow(piv.values, x=piv.columns.round(3), y=piv.index.round(3), aspect="auto", color_continuous_scale="Viridis", origin="lower", title=title)
                        st.plotly_chart(fig, use_container_width=True)
                    _heat(df_grid, "reject_rate", "Pr(Declare efficacy) over (p,q)")
                    _heat(df_grid, "ess", "Expected sample size over (p,q)")
                    _heat(df_grid, "safety_stop_any", "Pr(Safety stop at any stage) over (p,q)")

                # Pre-defined scenario breakpoints
                scenarios = [
                    (p0, st.session_state.get("q_good", 0.10), "Null eff, benign tox"),
                    (p1, st.session_state.get("q_good", 0.10), "Target eff, benign tox"),
                    ((p0 + p1)/2, (st.session_state.get("q_good", 0.10) + st.session_state.get("q_bad", 0.20))/2, "Borderline eff & tox"),
                    (min(1.0, p1 + 0.15), min(1.0, st.session_state.get("q_bad", 0.20) + 0.05), "High eff, high tox"),
                ]
                rows_scen = []
                for (pp, qq, name) in scenarios:
                    r = simulate_design_joint(design_sel, p_eff=pp, p_tox=qq, U_eff=Ueff, U_tox=Utox)
                    rows_scen.append(dict(scenario=name, p=pp, q=qq, reject_rate=r["reject_rate"], ess=r["ess"], safety_stop_any=r["safety_stop_prob"], early_any=r["early_stop_rate"]))
                df_scen = pd.DataFrame(rows_scen)
                st.write("**Scenario breakpoints (quick view):**")
                st.dataframe(df_scen)
                st.session_state["oc_scenarios_df"] = df_scen

    # ── Export buttons (settings + results) ─────────────────────────────────
    st.write("### 6) Export settings and results")
    st.caption("Download a JSON bundle and/or a ZIP containing CSVs for screening, deep‑dive, compare, and OC explorer.")

    # Assemble JSON bundle
    export_bundle = {
        "design": {
            "N_total": int(N_select),
            "looks_eff": [int(x) for x in looks_eff_sel],
            "looks_saf": [int(x) for x in looks_saf_sel],
            "run_in_eff": int(run_in_eff),
            "run_in_saf": int(run_in_saf),
            "prior_efficacy": {"a0": float(a0), "b0": float(b0)},
            "prior_safety": ({"a_t0": float(a_t0), "b_t0": float(b_t0)} if enable_safety else None),
            "p0": float(p0), "p1": float(p1),
            "theta_final": float(theta_final), "theta_interim": float(theta_interim),
            "c_futility": float(c_futility), "allow_early_success": bool(allow_early_success),
        },
        "screening_table": df_screen.to_dict(orient="list") if not df_screen.empty else None,
        "deep_dive": st.session_state.get("deep_results", None),
        "compare": st.session_state.get("compare_df", None),
        "tuner": st.session_state.get("tuner_df", None),
        "oc_grid": st.session_state.get("oc_grid_df", None),
        "oc_scenarios": st.session_state.get("oc_scenarios_df", None),
    }

    json_bytes = json.dumps(export_bundle, default=lambda o: o if isinstance(o, (int, float, str, bool, type(None))) else str(o)).encode("utf-8")
    st.download_button("Download JSON bundle", data=json_bytes, file_name="design_and_results.json", mime="application/json")

    # ZIP of CSVs
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Screening table
        if not df_screen.empty:
            zf.writestr("screening.csv", df_screen.to_csv(index=False))
        # Deep dive stop dists + summary
        deep = st.session_state.get("deep_results", None)
        if deep is not None:
            for name in ["res_p0_qgood", "res_p1_qgood", "res_p1_qbad"]:
                if name in deep and isinstance(deep[name], dict) and "stop_dist" in deep[name]:
                    df_sd = deep[name]["stop_dist"]
                    if isinstance(df_sd, pd.DataFrame):
                        zf.writestr(f"deep_{name}_stopdist.csv", df_sd.to_csv(index=False))
            # Deep dive summary metrics
            rows = []
            for k, v in deep.items():
                rows.append({"scenario": k, **{kk: vv for kk, vv in v.items() if kk != "stop_dist"}})
            zf.writestr("deep_summary.csv", pd.DataFrame(rows).to_csv(index=False))
        # Compare
        if st.session_state.get("compare_df", None) is not None:
            zf.writestr("compare.csv", st.session_state["compare_df"].to_csv(index=False))
        # Tuner
        if st.session_state.get("tuner_df", None) is not None:
            zf.writestr("tuner_candidates.csv", st.session_state["tuner_df"].to_csv(index=False))
        # OC explorer grid and scenarios
        if st.session_state.get("oc_grid_df", None) is not None:
            zf.writestr("oc_grid.csv", st.session_state["oc_grid_df"].to_csv(index=False))
        if st.session_state.get("oc_scenarios_df", None) is not None:
            zf.writestr("oc_scenarios.csv", st.session_state["oc_scenarios_df"].to_csv(index=False))

    st.download_button("Download ZIP (CSVs)", data=buf.getvalue(), file_name="design_results_csv.zip", mime="application/zip")

st.write("---")
st.write("#### Notes")
st.markdown("- Threshold Tuner uses **efficacy-only** simulation to rapidly search θ-values. You can widen the ranges/steps or increase sims for higher precision.")
st.markdown("- OC Explorer runs full **joint** simulations on a (p,q) grid; set sims per cell high enough for your precision needs.")
