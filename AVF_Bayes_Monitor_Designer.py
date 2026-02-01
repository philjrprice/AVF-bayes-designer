# AVF_Bayes_Monitor_Designer_v3_1_5_OCcurves_NO_TUNER.py
# Streamlit app for single-arm Bayesian monitored design (binary endpoint)
# v3.1.5q: Restores rich Deep‑Dive tables/plots safely.
# Adds "Compute OC/ESS Curves (fixed q)" in Deep‑Dive.
# Keeps single-line guard to avoid indentation errors.
from __future__ import annotations
import io
import json
import zipfile
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import beta
from scipy.special import betaln, comb
# PDFs
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
# Plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False
SCHEMA_VERSION = "v3_1_5q_indent_guard_plus_preview"
# --- UI state helpers (keep panels open after a run) ---
def _get_flag(name: str, default: bool = False) -> bool:
    import streamlit as st
    if name not in st.session_state:
        st.session_state[name] = default
    return st.session_state[name]
def _set_flag(name: str, value: bool = True):
    import streamlit as st
    st.session_state[name] = value
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ CORE BAYESIAN UTILITIES                                                  ║
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
    return math.log(comb(m, y)) + betaln(y + a, m - y + b) - betaln(a, b)
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
def build_looks_with_runin(
    N: int,
    run_in: int,
    mode_label: str,
    k_total: Optional[int] = None,
    perc_str: Optional[str] = None,
    ns_str: Optional[str] = None,
    step_every: Optional[int] = None,
) -> List[int]:
    looks: List[int] = []
    run_in = int(max(0, min(run_in, N - 1)))
    if run_in > 0:
        looks.append(run_in)
    if mode_label == "None (final only)":
        return looks
    if mode_label == "Equal‑spaced (choose total looks incl. run‑in)":
        K = int(k_total or 0)
        K_rem = max(0, K - (1 if run_in > 0 else 0))
        if K_rem > 0 and (N - run_in) > 0:
            for i in range(1, K_rem + 1):
                n_i = run_in + int(np.floor(i * (N - run_in) / (K_rem + 1)))
                if 0 < n_i < N:
                    looks.append(n_i)
    elif mode_label == "Custom percentages of remaining":
        fracs = parse_percent_list(perc_str or "")
        for f in fracs:
            n_i = run_in + int(np.floor(f * (N - run_in)))
            if 0 < n_i < N:
                looks.append(n_i)
    elif mode_label == "Custom absolute Ns":
        ns = parse_n_list(ns_str or "")
        for n_i in ns:
            if run_in > 0 and n_i == run_in:
                pass
            if run_in < n_i < N:
                looks.append(int(n_i))
    elif mode_label == "Look every N after run‑in":
        step = int(step_every or 0)
        if step > 0 and (N - run_in) > 0:
            k = 1
            while True:
                n_i = run_in + k * step
                if n_i >= N:
                    break
                looks.append(int(n_i))
                k += 1
    looks = [int(x) for x in looks if 0 < x < N]
    looks = sorted(list(dict.fromkeys(looks)))
    return looks
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ MONTE CARLO SE/CI HELPERS                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def se_prop(p_hat: float, n: int) -> float:
    p_hat = max(0.0, min(1.0, float(p_hat)))
    return math.sqrt(max(p_hat * (1 - p_hat), 1e-12) / max(1, n))
def ci95_prop(p_hat: float, n: int) -> Tuple[float, float]:
    se = se_prop(p_hat, n)
    return max(0.0, p_hat - 1.96 * se), min(1.0, p_hat + 1.96 * se)
def ess_se_from_stopdist(df_stop: pd.DataFrame, n: int) -> float:
    if df_stop is None or len(df_stop) == 0:
        return float('nan')
    mean = float((df_stop["N_stop"] * df_stop["Probability"]).sum())
    var = float((((df_stop["N_stop"] - mean) ** 2) * df_stop["Probability"]).sum())
    return float(math.sqrt(max(var, 0.0) / max(n, 1)))
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ SIMULATION ENGINES                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def simulate_design_eff_only(design: Dict, p: float, U: np.ndarray) -> Dict:
    N = design["N_total"]
    looks_eff = design["looks_eff"]
    looks_fut = design.get("looks_fut", looks_eff)
    run_in_eff = design.get("run_in_eff", 0)
    a0 = design["a0"]; b0 = design["b0"]
    p0 = design["p0"]; theta_final = design["theta_final"]
    theta_interim = design.get("theta_interim", theta_final)
    allow_early = design["allow_early_success"]
    s_min_final = design["s_min_final"]
    x_min_to_continue = design["x_min_to_continue_by_look_fut"]
    n_sims = U.shape[0]
    X = (U[:, :N] < p).astype(np.int16)
    cum_x = np.zeros(n_sims, dtype=np.int32)
    n_curr = 0
    active = np.ones(n_sims, dtype=bool)
    success = np.zeros(n_sims, dtype=bool)
    final_n = np.zeros(n_sims, dtype=np.int32)
    eff_early_succ_by_look = np.zeros(len(looks_eff), dtype=np.int64)
    fut_early_by_look = np.zeros(len(looks_fut), dtype=np.int64)
    if run_in_eff > 0:
        cum_x[active] += np.sum(X[active, :run_in_eff], axis=1)
        n_curr = run_in_eff
    events = sorted(list(dict.fromkeys(list(looks_eff) + list(looks_fut) + [N])))
    for ev in events:
        if ev > n_curr and active.any():
            cum_x[active] += np.sum(X[active, n_curr:ev], axis=1)
            n_curr = ev
        # Early success at efficacy looks
        if ev in looks_eff and allow_early and active.any():
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
        # Futility at futility looks
        if ev in looks_fut and active.any():
            x_min = x_min_to_continue.get(ev, None)
            li_f = looks_fut.index(ev)
            if x_min is None:
                idx = np.where(active)[0]
                if idx.size > 0:
                    final_n[idx] = n_curr
                    active[idx] = False
                    fut_early_by_look[li_f] += idx.size
            else:
                need_continue = cum_x[active] >= x_min
                idx_all = np.where(active)[0]
                idx_stop = idx_all[~need_continue]
                if idx_stop.size > 0:
                    final_n[idx_stop] = n_curr
                    active[idx_stop] = False
                    fut_early_by_look[li_f] += idx_stop.size
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
    early_succ_rate = eff_early_succ_by_look.sum() / n_sims
    early_fut_rate = fut_early_by_look.sum() / n_sims
    early_stop_rate = early_succ_rate + early_fut_rate
    unique_ns, counts = np.unique(final_n, return_counts=True)
    stop_dist = pd.DataFrame({"N_stop": unique_ns, "Probability": counts / n_sims}).sort_values("N_stop")
    return {
        "reject_rate": float(reject_rate),
        "ess": float(ess),
        "stop_dist": stop_dist,
        "early_success_by_look_eff": (eff_early_succ_by_look / n_sims).tolist(),
        "early_fut_by_look_eff": (fut_early_by_look / n_sims).tolist(),
        "eff_early_succ_rate": float(early_succ_rate),
        "eff_early_fut_rate": float(early_fut_rate),
        "early_stop_rate": float(early_stop_rate),
    }
def simulate_design_joint(design: Dict, p_eff: float, p_tox: float, U_eff: np.ndarray, U_tox: np.ndarray) -> Dict:
    N = design["N_total"]
    looks_eff = design["looks_eff"]
    looks_fut = design.get("looks_fut", looks_eff)
    looks_saf = design.get("looks_saf", [])
    a0 = design["a0"]; b0 = design["b0"]
    p0 = design["p0"]; theta_final = design["theta_final"]
    theta_interim = design.get("theta_interim", theta_final)
    allow_early = design["allow_early_success"]
    s_min_final = design["s_min_final"]
    x_min_to_continue = design["x_min_to_continue_by_look_fut"]
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
    eff_early_fut_by_look = np.zeros(len(looks_fut), dtype=np.int64)
    saf_stop_by_look = np.zeros(len(looks_saf) + 1, dtype=np.int64)  # +1 for final
    events = sorted(list(dict.fromkeys(list(looks_eff) + list(looks_fut) + list(looks_saf) + [N])))
    for ev in events:
        if ev > n_curr and active.any():
            cum_x[active] += np.sum(X_eff[active, n_curr:ev], axis=1)
            if tox is not None:
                cum_t[active] += np.sum(X_tox[active, n_curr:ev], axis=1)
            n_curr = ev
        # Safety checks
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
        # Early success
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
        # Futility
        if ev in looks_fut and active.any():
            x_min = x_min_to_continue.get(ev, None)
            li_f = looks_fut.index(ev)
            if x_min is None:
                idx = np.where(active)[0]
                if idx.size > 0:
                    final_n[idx] = n_curr
                    active[idx] = False
                    eff_early_fut_by_look[li_f] += idx.size
            else:
                need_continue = cum_x[active] >= x_min
                idx_all = np.where(active)[0]
                idx_stop = idx_all[~need_continue]
                if idx_stop.size > 0:
                    final_n[idx_stop] = n_curr
                    active[idx_stop] = False
                    eff_early_fut_by_look[li_f] += idx_stop.size
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
    eff_early_succ_rate = eff_early_succ_by_look.sum() / n_sims
    eff_early_fut_rate = eff_early_fut_by_look.sum() / n_sims
    saf_early_rate = saf_stop_by_look[:-1].sum() / n_sims
    any_safety_rate = saf_stop_by_look.sum() / n_sims
    early_stop_rate = eff_early_succ_rate + eff_early_fut_rate + saf_early_rate
    unique_ns, counts = np.unique(final_n, return_counts=True)
    stop_dist = pd.DataFrame({"N_stop": unique_ns, "Probability": counts / n_sims}).sort_values("N_stop")
    fut_by_look_vec = (eff_early_fut_by_look / n_sims).tolist()
    return {
        "reject_rate": float(reject_rate),
        "ess": float(ess),
        "safety_stop_prob": float(any_safety_rate),
        "stop_dist": stop_dist,
        "eff_early_succ_by_look": (eff_early_succ_by_look / n_sims).tolist(),
        "eff_early_fut_by_look": fut_by_look_vec,  # compat
        "fut_early_by_look": fut_by_look_vec,      # clearer alias
        "saf_by_look": (saf_stop_by_look / n_sims).tolist(),
        "eff_early_succ_rate": float(eff_early_succ_rate),
        "eff_early_fut_rate": float(eff_early_fut_rate),
        "saf_early_rate": float(saf_early_rate),
        "early_stop_rate": float(early_stop_rate),
    }
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ PLOTTING HELPERS (OC/ESS preview)                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def _normalize_oc_ess_df(df_curves: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns to expected names: p, reject_rate, ess."""
    df = df_curves.copy()
    ren = {}
    if 'reject_rate' not in df.columns:
        if 'reject rate' in df.columns:
            ren['reject rate'] = 'reject_rate'
        if 'Pr(declare efficacy)' in df.columns:
            ren['Pr(declare efficacy)'] = 'reject_rate'
    if 'ess' not in df.columns and 'ESS' in df.columns:
        ren['ESS'] = 'ess'
    if 'p' not in df.columns and 'p_eff' in df.columns:
        ren['p_eff'] = 'p'
    if ren:
        df = df.rename(columns=ren)
    return df
def _render_oc_ess_preview(df_curves: pd.DataFrame, q_fixed: Optional[float] = None):
    """Render dual-axis OC/ESS vs p using plotly (called via one-line guard)."""
    if not _HAS_PLOTLY:
        st.dataframe(df_curves, use_container_width=True)
        return
    df_plot = _normalize_oc_ess_df(df_curves)
    required = {'p', 'reject_rate', 'ess'}
    if df_plot.empty or not required.issubset(set(df_plot.columns)):
        st.warning("OC/ESS curves: results are empty or columns are missing; showing raw table instead of a plot.")
        st.dataframe(df_curves, use_container_width=True)
        return
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df_plot['p'], y=df_plot['reject_rate'],
                             mode='lines+markers', name='Pr(declare efficacy)'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_plot['p'], y=df_plot['ess'],
                             mode='lines+markers', name='ESS'), secondary_y=True)
    fig.update_yaxes(title_text='Pr(declare efficacy)', secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text='ESS', secondary_y=True)
    fig.update_xaxes(title_text='Efficacy rate p')
    title_q = f"OC/ESS vs p at fixed q={q_fixed:.2f}" if (q_fixed is not None) else "OC/ESS vs p (fixed q)"
    fig.update_layout(title=title_q)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_plot, use_container_width=True)
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ SCREENING (FAST)                                                         ║
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
        looks_fut = g.get("looks_fut", looks_eff)
        a0 = g["a0"]; b0 = g["b0"]
        p0 = g["p0"]; p1 = g["p1"]
        theta_final = g["theta_final"]; c_futility = g["c_futility"]
        theta_interim = g.get("theta_interim", theta_final)
        allow_early = g["allow_early_success"]
        s_min = min_successes_for_posterior_threshold(a0, b0, N, p0, theta_final)
        if s_min is None:
            continue
        x_min_to_continue = compute_interim_futility_cutoffs(a0, b0, N, looks_fut, p0, theta_final, c_futility)
        design = dict(
            N_total=N, looks_eff=looks_eff, looks_fut=looks_fut, a0=a0, b0=b0, p0=p0,
            theta_final=theta_final, theta_interim=theta_interim,
            c_futility=c_futility, allow_early_success=allow_early,
            s_min_final=s_min, x_min_to_continue_by_look_fut=x_min_to_continue,
            p1=p1, run_in_eff=g.get("run_in_eff", 0)
        )
        res_p0 = simulate_design_eff_only(design, p0, U[:, :N])
        res_p1 = simulate_design_eff_only(design, p1, U[:, :N])
        rows.append({
            "N_total": N,
            "looks_eff": looks_eff,
            "Efficacy evals (incl. final)": (looks_eff + [N]),
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
            "Early success @ p0": res_p0.get("eff_early_succ_rate", res_p0.get("early_succ_rate", float("nan"))),
            "Early futility @ p0": res_p0.get("eff_early_fut_rate", res_p0.get("early_fut_rate", float("nan"))),
            "s_min_final": s_min,
            "x_min_to_continue_eff": x_min_to_continue,
        })
        designs_built.append(design)
    df = pd.DataFrame(rows)
    return df, designs_built
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ STREAMLIT UI — Header & Sidebar                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.set_page_config(page_title="Bayesian Single‑Arm Designer (Binary) — v3.1.5q", layout="wide")
st.title("Bayesian Single‑Arm Monitored Study Designer (Binary Endpoint) — v3.1.5q")
st.caption("Deep‑Dive: restored OC/ESS preview & detailed tables; keeps single‑line guard; OC Explorer unchanged.")
# Sidebar — Efficacy
st.sidebar.header("1) Efficacy target and prior")
col1, col2 = st.sidebar.columns(2)
with col1:
    p0 = st.number_input("Null response rate (p₀)", 0.0, 1.0, 0.20, 0.01, format="%.2f", key='p0',
                         help="The highest response rate that you would still consider not clinically worthwhile.")
    a0 = st.number_input("Prior a₀ (efficacy)", 0.0, None, 1.0, 0.5, key='a0',
                         help="Beta(a₀,b₀) prior for response. Increase a₀+b₀ for a stronger prior.")
    theta_final_default = 0.95 if 'theta_final' not in st.session_state else float(st.session_state['theta_final'])
    theta_final = st.number_input("Final success threshold (θ_final)", 0.5, 0.999, theta_final_default, 0.01, format="%.3f", key='theta_final',
                         help="At the final look, declare success if P(p>p₀ | all data) ≥ θ_final.")
with col2:
    p1 = st.number_input("Target (promising) response rate (p₁)", 0.0, 1.0, 0.40, 0.01, format="%.2f", key='p1',
                         help="A response rate you would be happy to see. Power is evaluated here.")
    b0 = st.number_input("Prior b₀ (efficacy)", 0.0, None, 1.0, 0.5, key='b0',
                         help="Beta(a₀,b₀) prior for response. Prior mean=a₀/(a₀+b₀); ESS≈a₀+b₀.")
    c_futility_default = 0.05 if 'c_futility' not in st.session_state else float(st.session_state['c_futility'])
    c_futility = st.number_input("Futility cutoff (PPoS)", 0.0, 0.5, c_futility_default, 0.01, format="%.3f", key='c_futility',
                         help="At interim looks, stop for futility if Predictive Probability of **final** success < this value.")
with st.sidebar.expander("About your efficacy prior"):
    if (a0 + b0) > 0:
        st.write(f"Prior mean = **{a0/(a0+b0):.3f}**, prior ESS ≈ **{a0+b0:.1f}**.")
    else:
        st.write("Prior mean undefined (a₀+b₀=0). Consider Beta(1,1).")
allow_early_success = st.sidebar.checkbox("Allow early success at interim looks?", value=st.session_state.get('allow_early_success', False), key='allow_early_success',
    help="If checked, you may stop early for success when the posterior reaches θ_interim.")
if 'theta_interim' not in st.session_state:
    st.session_state['theta_interim'] = float(theta_final)
theta_interim = st.sidebar.number_input("Interim success threshold (θ_interim)", 0.5, 0.999, float(st.session_state['theta_interim']), 0.01, format="%.3f", key='theta_interim',
    help="Often set near θ_final to avoid overly optimistic early claims.")
# Sidebar — Safety
st.sidebar.header("2) Optional safety monitoring")
enable_safety = st.sidebar.checkbox("Enable safety monitoring?", True, key='enable_safety', help="Monitor serious adverse events (SAE) with a Beta prior; stop if SAE rate likely exceeds q_max.")
if enable_safety:
    c1, c2 = st.sidebar.columns(2)
    with c1:
        a_t0 = st.number_input("Safety prior a_t0", 0.0, None, 1.0, 0.5, key='a_t0',
                               help="Beta(a_t0,b_t0) prior for SAE rate.")
        q_max = st.number_input("Max acceptable SAE rate (q_max)", 0.0, 1.0, 0.15, 0.01, format="%.2f", key='q_max',
                                help="If the SAE rate is likely above q_max, the trial stops for safety.")
    with c2:
        b_t0 = st.number_input("Safety prior b_t0", 0.0, None, 9.0, 0.5, key='b_t0',
                               help="Increase a_t0+b_t0 for a stronger safety prior (e.g., Beta(1,9) → mean≈0.10).")
        theta_tox = st.number_input("Safety stop threshold (θ_tox)", 0.5, 0.999, 0.90, 0.01, format="%.3f", key='theta_tox',
                                    help="Stop when P(q>q_max | data) ≥ θ_tox at interims or final.")
    with st.sidebar.expander("About your safety prior"):
        if (a_t0 is not None) and (b_t0 is not None) and (a_t0 + b_t0) > 0:
            st.write(f"Prior mean = **{a_t0/(a_t0+b_t0):.3f}**, prior ESS ≈ **{a_t0+b_t0:.1f}**.")
        else:
            st.write("Prior mean undefined (a_t0+b_t0=0). Consider Beta(1,9) or similar.")
else:
    a_t0 = b_t0 = q_max = theta_tox = None
# Sidebar — Look schedules
st.sidebar.header("3) When to check the data (look schedule)")
run_in_eff = st.sidebar.number_input("Run‑in for efficacy (patients enrolled before first look)", 0, 400, 0, 1, key='run_in_eff',
    help="Patients enrolled before the first efficacy look; they count for all decisions.")
looks_eff_mode_label = st.sidebar.selectbox(
    "Efficacy look timing",
    ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"],
    index=1, key='eff_mode',
    help="Choose how to place efficacy interim looks."
)
k_looks_eff = perc_eff_str = ns_eff_str = None
if looks_eff_mode_label == "Equal‑spaced (choose total looks incl. run‑in)":
    k_looks_eff = st.sidebar.slider("Total efficacy looks (including run‑in)", 1, 8, 2, 1, key='k_eff',
        help="For example, 2 looks → roughly at 1/3 and 2/3 of the maximum N.")
elif looks_eff_mode_label == "Custom percentages of remaining":
    perc_eff_str = st.sidebar.text_input("Efficacy look percentages (comma)", "33,67", key='perc_eff',
        help="Example: 25,50,75 (percent of the planned maximum N)")
elif looks_eff_mode_label == "Custom absolute Ns":
    ns_eff_str = st.sidebar.text_input("Efficacy look sample sizes N (comma)", "", key='ns_eff',
        help="Example: 20,40 (each must be less than the maximum N)")
st.sidebar.header("4) Safety look schedule")
run_in_saf = st.sidebar.number_input("Run‑in for safety", 0, 400, 0, 1, key='run_in_saf',
    help="Patients enrolled before the first safety look; they count for safety decisions.")
looks_saf_mode_label = st.sidebar.selectbox("Safety look timing",
    ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns", "Look every N after run‑in"],
    index=1, key='saf_mode',
    help="Choose how to place safety interim looks.")
k_looks_saf = perc_saf_str = ns_saf_str = step_saf = None
if looks_saf_mode_label == "Equal‑spaced (choose total looks incl. run‑in)":
    k_looks_saf = st.sidebar.slider("Total safety looks (including run‑in)", 1, 8, 2, 1, key='k_saf')
elif looks_saf_mode_label == "Custom percentages of remaining":
    perc_saf_str = st.sidebar.text_input("Safety look percentages (comma)", "33,67", key='perc_saf')
elif looks_saf_mode_label == "Custom absolute Ns":
    ns_saf_str = st.sidebar.text_input("Safety look sample sizes N (comma)", "", key='ns_saf')
elif looks_saf_mode_label == "Look every N after run‑in":
    step_saf = st.sidebar.number_input("Look every N participants (after safety run‑in)", 1, 400, 10, 1, key='step_saf',
        help="First safety look is after the run‑in; then look every N participants. Final analysis still occurs at max N.")
# Sidebar — Futility look schedule
st.sidebar.header("4b) Futility look schedule")
use_fut_same = st.sidebar.checkbox(
    "Use same schedule as efficacy? (default)",
    True,
    key='fut_same',
    help="If checked, futility looks occur at the same Ns as efficacy looks (backward-compatible). Uncheck to customize a separate futility schedule."
)
if use_fut_same:
    run_in_fut = int(st.session_state.get('run_in_eff', run_in_eff))
    looks_fut_mode_label = "Same as efficacy"
    k_looks_fut = perc_fut_str = ns_fut_str = step_fut = None
else:
    run_in_fut = st.sidebar.number_input(
        "Run‑in for futility",
        0, 400, int(st.session_state.get('run_in_eff', run_in_eff)), 1, key='run_in_fut',
        help="Patients enrolled before the first futility look; they count for futility decisions."
    )
    looks_fut_mode_label = st.sidebar.selectbox(
        "Futility look timing",
        ["Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns", "Look every N after run‑in"],
        index=1, key='fut_mode',
        help="Choose how to place futility interim looks."
    )
    k_looks_fut = perc_fut_str = ns_fut_str = step_fut = None
    if looks_fut_mode_label == "Equal‑spaced (choose total looks incl. run‑in)":
        k_looks_fut = st.sidebar.slider("Total futility looks (including run‑in)", 1, 8, 2, 1, key='k_fut')
    elif looks_fut_mode_label == "Custom percentages of remaining":
        perc_fut_str = st.sidebar.text_input("Futility look percentages (comma)", "33,67", key='perc_fut')
    elif looks_fut_mode_label == "Custom absolute Ns":
        ns_fut_str = st.sidebar.text_input("Futility look sample sizes N (comma)", "", key='ns_fut')
    elif looks_fut_mode_label == "Look every N after run‑in":
        step_fut = st.sidebar.number_input("Look every N participants (after futility run‑in)", 1, 400, 10, 1, key='step_fut',
            help="First futility look is after the run‑in; then look every N participants. Final analysis still occurs at max N.")
# Sidebar — Screener settings
st.sidebar.header("5) Rapid Screener")
N_min, N_max = st.sidebar.slider("Range of maximum N values to test", 10, 400, (30, 120), 1, key='Nrange',
    help="We'll scan this range to find feasible, efficient designs.")
N_step = st.sidebar.number_input("Step between N values", 1, 50, 5, 1, key='Nstep', help="Larger steps scan fewer N values (faster).")
n_sims_small = st.sidebar.number_input("Simulations per design (screening)", 100, 200000, 5000, 500, key='sims_small',
    help="Higher = more precision; the Deep Dive is where precision matters most.")
alpha_max = st.sidebar.number_input("Max Type I error (α) allowed", 0.0, 0.5, 0.10, 0.01, format="%.2f", key='alpha_max',
    help="Keep only designs with false-positive rate ≤ this.")
power_min = st.sidebar.number_input("Min power at p₁", 0.0, 1.0, 0.80, 0.01, format="%.2f", key='power_min',
    help="Keep only designs with success rate at p₁ ≥ this.")
seed = st.sidebar.number_input("Random seed", 1, None, 2026, 1, key='seed', help="Controls reproducibility.")
# Build candidate grid for Screener
Ns = list(range(N_min, N_max + 1, N_step))
param_grid = []
for N in Ns:
    looks_eff = build_looks_with_runin(N, run_in_eff, looks_eff_mode_label, k_total=k_looks_eff, perc_str=perc_eff_str, ns_str=ns_eff_str)
    looks_fut = looks_eff if st.session_state.get('fut_same', True) or looks_fut_mode_label == 'Same as efficacy' else build_looks_with_runin(
        N, run_in_fut, looks_fut_mode_label, k_total=k_looks_fut, perc_str=perc_fut_str, ns_str=ns_fut_str, step_every=step_fut
    )
    param_grid.append(dict(N_total=N, looks_eff=looks_eff, looks_fut=looks_fut,
                           a0=a0, b0=b0, p0=p0, p1=p1,
                           theta_final=theta_final, theta_interim=float(theta_interim),
                           c_futility=c_futility, allow_early_success=allow_early_success,
                           run_in_eff=int(run_in_eff)))
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 1) Rapid Screener (efficacy‑only)                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
@st.cache_data(show_spinner=False)
def _screen(param_grid, n_sims_small, seed, schema_version: str):
    rng = np.random.default_rng(seed)
    Nmax = max([g["N_total"] for g in param_grid]) if param_grid else 0
    if Nmax <= 0:
        return pd.DataFrame(), []
    U = rng.uniform(size=(n_sims_small, Nmax))
    df, designs = shortlist_designs(param_grid, n_sims_small, seed, U)
    _ = schema_version
    return df, designs
df_screen, designs_built = _screen(param_grid, n_sims_small, seed, SCHEMA_VERSION)
st.write("### 1) Rapid Screener (efficacy‑only)")
if df_screen.empty:
    st.warning("No viable designs found. Try relaxing θ_final or increasing N.")
else:
    df_ok = df_screen[(df_screen["Type I error @ p0"] <= alpha_max) & (df_screen["Power @ p1"] >= power_min)].copy()
    cols = ["N_total", "run_in_eff", "Efficacy evals (incl. final)", "theta_final", "c_futility", "Type I error @ p0", "Power @ p1", "ESS @ p0", "Early stop @ p0 (any)"]
    table_df = df_ok if not df_ok.empty else df_screen
    # Backward-compat: synthesize column if absent (old caches)
    if 'Efficacy evals (incl. final)' not in table_df.columns and 'looks_eff' in table_df.columns and 'N_total' in table_df.columns:
        try:
            table_df = table_df.copy()
            table_df['Efficacy evals (incl. final)'] = table_df.apply(lambda r: list(r['looks_eff']) + [int(r['N_total'])], axis=1)
        except Exception:
            table_df = table_df.copy()
            table_df['Efficacy evals (incl. final)'] = table_df['looks_eff']
    st.dataframe(table_df[cols].sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True), use_container_width=True)
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 2) Compare multiple Ns (deep‑dive, joint with safety)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.write("### 2) Compare multiple N values (deep‑dive, joint with safety)")
with st.expander("Open compare panel", expanded=False):
    ns_str_compare = st.text_input("Enter N values (comma)", "60,70,80", key='cmp_ns_str', help="We'll construct designs with the SAME rule settings you set above.")
    n_sims_compare = st.number_input("Simulations per design (compare run)", 5000, 400000, 80000, 5000, key='cmp_sims')
    seed_compare = st.number_input("Random seed (compare run)", 1, None, seed + 2, 1, key='cmp_seed')
    q_good_cmp = st.number_input("q_good (typical/benign toxicity)", 0.0, 1.0, 0.10, 0.01, key='cmp_qgood')
    q_bad_cmp = st.number_input("q_bad (concerning/high toxicity)", 0.0, 1.0, 0.20, 0.01, key='cmp_qbad')
    def _build_eff(N):
        return build_looks_with_runin(N, run_in_eff, looks_eff_mode_label, k_total=k_looks_eff, perc_str=perc_eff_str, ns_str=ns_eff_str)
    def _build_saf(N):
        return build_looks_with_runin(N, run_in_saf, looks_saf_mode_label, k_total=k_looks_saf, perc_str=perc_saf_str, ns_str=ns_saf_str, step_every=step_saf)
    def _build_fut(N):
        if st.session_state.get('fut_same', True) or looks_fut_mode_label == 'Same as efficacy':
            return build_looks_with_runin(N, run_in_eff, looks_eff_mode_label, k_total=k_looks_eff, perc_str=perc_eff_str, ns_str=ns_eff_str)
        else:
            return build_looks_with_runin(N, int(st.session_state.get('run_in_fut', run_in_fut)), looks_fut_mode_label,
                                          k_total=k_looks_fut, perc_str=perc_fut_str, ns_str=ns_fut_str, step_every=step_fut)
    if st.button("Run compare", key='run_compare'):
        Ns_cmp = []
        for tok in ns_str_compare.split(','):
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
            rows = []
            for Ncmp in Ns_cmp:
                looks_eff_cmp = _build_eff(Ncmp)
                looks_saf_cmp = _build_saf(Ncmp)
                looks_fut_cmp = _build_fut(Ncmp)
                smin_cmp = min_successes_for_posterior_threshold(a0, b0, Ncmp, p0, theta_final)
                if smin_cmp is None:
                    continue
                x_min_cmp = compute_interim_futility_cutoffs(a0, b0, Ncmp, looks_fut_cmp, p0, theta_final, c_futility)
                design_cmp = dict(N_total=Ncmp, looks_eff=looks_eff_cmp, looks_saf=looks_saf_cmp, looks_fut=looks_fut_cmp,
                                  run_in_eff=int(run_in_eff), run_in_saf=int(run_in_saf), a0=a0, b0=b0,
                                  p0=p0, theta_final=theta_final, theta_interim=float(theta_interim),
                                  c_futility=c_futility, allow_early_success=allow_early_success,
                                  s_min_final=smin_cmp, x_min_to_continue_by_look_fut=x_min_cmp)
                if enable_safety:
                    design_cmp["safety"] = dict(a_t0=a_t0, b_t0=b_t0, q_max=q_max, theta_tox=theta_tox)
                Ueff = rng.uniform(size=(n_sims_compare, Ncmp))
                Utox = rng.uniform(size=(n_sims_compare, Ncmp))
                r_p0 = simulate_design_joint(design_cmp, p_eff=p0, p_tox=q_good_cmp, U_eff=Ueff, U_tox=Utox)
                r_p1 = simulate_design_joint(design_cmp, p_eff=p1, p_tox=q_good_cmp, U_eff=Ueff, U_tox=Utox)
                r_p1_bad = simulate_design_joint(design_cmp, p_eff=p1, p_tox=q_bad_cmp, U_eff=Ueff, U_tox=Utox)
                rows.append({
                    "N": Ncmp,
                    "looks_eff": looks_eff_cmp,
                    "looks_saf": looks_saf_cmp,
                    "s_min_final": smin_cmp,
                    "Efficacy evals (incl. final)": (looks_eff_cmp + [Ncmp]),
                    "Safety evals (incl. final)": (looks_saf_cmp + [Ncmp]),
                    **{
                        "Type I @p0,q_good": r_p0["reject_rate"], "Power @p1,q_good": r_p1["reject_rate"],
                        "ESS @p0": r_p0["ess"], "ESS @p1": r_p1["ess"],
                        "Early stop (any) @p1": r_p1["early_stop_rate"],
                        "Early succ @p1": r_p1["eff_early_succ_rate"],
                        "Early fut @p1": r_p1["eff_early_fut_rate"],
                        "Early safety @p1": r_p1.get("saf_early_rate", 0.0),
                        "Safety stop @p1,q_bad (any stage)": r_p1_bad.get("safety_stop_prob", 0.0),
                    }
                })
            if not rows:
                st.warning("None of the N values produced a feasible final rule. Try relaxing θ_final or adjusting looks.")
            else:
                df_cmp = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)
                st.dataframe(df_cmp, use_container_width=True)
                st.session_state['compare_df'] = df_cmp
                if _HAS_PLOTLY and not df_cmp.empty:
                    fig_cmp = go.Figure()
                    fig_cmp.add_bar(x=df_cmp['N'], y=df_cmp['Early succ @p1'], name='Early success @p1,q_good')
                    fig_cmp.add_bar(x=df_cmp['N'], y=df_cmp['Early fut @p1'], name='Early futility @p1,q_good')
                    if 'Early safety @p1' in df_cmp.columns:
                        fig_cmp.add_bar(x=df_cmp['N'], y=df_cmp['Early safety @p1'], name='Early safety @p1,q_good')
                    if 'Safety stop @p1,q_bad (any stage)' in df_cmp.columns:
                        fig_cmp.add_bar(x=df_cmp['N'], y=df_cmp['Safety stop @p1,q_bad (any stage)'], name='Safety stop (any) @p1,q_bad')
                    fig_cmp.update_layout(barmode='group', title='Stop proportions by reason vs N (final always evaluated at max N)', xaxis_title='N', yaxis_title='Proportion')
                    st.plotly_chart(fig_cmp, use_container_width=True)
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 3) Deep Dive (joint efficacy + safety)                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.write("### 3) Deep Dive (joint efficacy + safety)")
N_select = st.number_input("Select a maximum sample size N to deep‑dive", Ns[0] if Ns else 5, Ns[-1] if Ns else 400, max(60, Ns[0] if Ns else 60), 1, key='N_select',
    help="Choose an N from your screening range (or adjacent values) to inspect in detail.")
looks_eff_sel = build_looks_with_runin(N_select, run_in_eff, looks_eff_mode_label, k_total=k_looks_eff, perc_str=perc_eff_str, ns_str=ns_eff_str)
looks_saf_sel = build_looks_with_runin(N_select, run_in_saf, looks_saf_mode_label, k_total=k_looks_saf, perc_str=perc_saf_str, ns_str=ns_saf_str, step_every=step_saf)
s_min_sel = min_successes_for_posterior_threshold(a0, b0, N_select, p0, theta_final)
if s_min_sel is None:
    st.error("Final rule infeasible at this N. Relax θ_final / adjust prior / increase N.")
else:
    looks_fut_sel = looks_eff_sel if st.session_state.get('fut_same', True) or looks_fut_mode_label == 'Same as efficacy' else build_looks_with_runin(
        N_select, int(st.session_state.get('run_in_fut', run_in_eff)), looks_fut_mode_label, k_total=k_looks_fut, perc_str=perc_fut_str, ns_str=ns_fut_str, step_every=step_fut
    )
    x_min_to_continue_sel = compute_interim_futility_cutoffs(a0, b0, N_select, looks_fut_sel, p0, theta_final, c_futility)
    design_sel = dict(N_total=N_select, a0=a0, b0=b0, p0=p0, p1=p1, theta_final=theta_final,
                      theta_interim=float(theta_interim), c_futility=c_futility,
                      allow_early_success=allow_early_success, s_min_final=s_min_sel,
                      looks_eff=looks_eff_sel, looks_saf=looks_saf_sel, looks_fut=looks_fut_sel,
                      run_in_eff=int(run_in_eff), run_in_saf=int(run_in_saf),
                      run_in_fut=int(st.session_state.get('run_in_fut', run_in_eff)),
                      x_min_to_continue_by_look_fut=x_min_to_continue_sel)
    if enable_safety:
        design_sel["safety"] = dict(a_t0=a_t0, b_t0=b_t0, q_max=q_max, theta_tox=theta_tox)
    # Expanded design summary
    with st.expander("Chosen Design Summary (plain language)", expanded=True):
        Ntot = design_sel['N_total']
        eff_all = (design_sel["looks_eff"] + [Ntot]) if len(design_sel["looks_eff"])>0 else [Ntot]
        saf_all = (design_sel.get("looks_saf", []) + [Ntot]) if len(design_sel.get("looks_saf", []))>0 else [Ntot]
        fut_all = (design_sel.get("looks_fut", []) + [Ntot]) if len(design_sel.get("looks_fut", []))>0 else [Ntot]
        st.markdown("**Design snapshot**")
        st.markdown("• **Planned maximum participants (N):** " + str(Ntot))
        st.markdown("• **Run‑in (eff/saf):** " + f"{design_sel.get('run_in_eff',0)} / {design_sel.get('run_in_saf',0)}")
        st.markdown("• **Look points — efficacy (including final):** " + ", ".join(str(x) for x in eff_all))
        st.markdown("• **Look points — safety (including final):** " + ", ".join(str(x) for x in saf_all))
        st.markdown("• **Look points — futility (including final):** " + ", ".join(str(x) for x in fut_all) + " *(no futility decision at final; shown for visibility)*")
        st.markdown("• **Efficacy prior Beta(a₀,b₀):** " + f"{design_sel['a0']:.3g}, {design_sel['b0']:.3g}")
        if enable_safety:
            st.markdown("• **Safety prior Beta(a_t0,b_t0):** " + f"{a_t0:.3g}, {b_t0:.3g}")
        st.markdown("• **Null/Target response (p₀ / p₁):** " + f"{p0:.2f} / {p1:.2f}")
        st.markdown("• **Key thresholds:** " + f"θ_final={design_sel['theta_final']:.3f}, θ_interim={design_sel['theta_interim']:.3f}, c_futility={design_sel['c_futility']:.3f}")
        if enable_safety:
            st.markdown("• **Safety thresholds:** " + f"q_max={q_max:.2f}, θ_tox={theta_tox:.3f}")
        st.markdown("• **Final rule (minimum successes needed):** " + f"s_min_final={design_sel['s_min_final']}")
        st.caption("Futility thresholds (x responders needed to CONTINUE at each futility look):")
        st.dataframe(pd.DataFrame.from_dict(design_sel["x_min_to_continue_by_look_fut"], orient="index", columns=["x ≥ to continue"]))
    # Deep-dive simulation settings
    st.write("#### Deep‑dive simulation settings")
    colD1, colD2, colD3 = st.columns(3)
    with colD1:
        n_sims_deep = st.number_input("Simulations to run (deep dive)", 2000, 800000, 150000, 5000, key='deep_sims',
            help="More simulations → tighter precision; slower.")
    with colD2:
        seed_deep = st.number_input("Random seed (deep dive)", 1, None, seed + 1, 1, key='deep_seed',
            help="Change to re-simulate independently.")
    with colD3:
        show_ci = st.checkbox("Show 95% CI bands on metrics", value=True, key='deep_ci')
    colQ1, colQ2 = st.columns(2)
    with colQ1:
        q_good = st.number_input("q_good (typical/benign toxicity)", 0.0, 1.0, 0.10, 0.01, key='q_good')
    with colQ2:
        q_bad = st.number_input("q_bad (concerning/high toxicity)", 0.0, 1.0, 0.20, 0.01, key='q_bad')
    if st.button("Run DEEP‑DIVE (efficacy + safety)", key='run_deep'):
        rng = np.random.default_rng(seed_deep)
        Ueff = rng.uniform(size=(n_sims_deep, design_sel["N_total"]))
        Utox = rng.uniform(size=(n_sims_deep, design_sel["N_total"]))
        res_p0_qgood = simulate_design_joint(design_sel, p_eff=p0, p_tox=q_good, U_eff=Ueff, U_tox=Utox)
        res_p1_qgood = simulate_design_joint(design_sel, p_eff=p1, p_tox=q_good, U_eff=Ueff, U_tox=Utox)
        res_p1_qbad = simulate_design_joint(design_sel, p_eff=p1, p_tox=q_bad, U_eff=Ueff, U_tox=Utox)
        st.session_state["deep_results"] = dict(res_p0_qgood=res_p0_qgood, res_p1_qgood=res_p1_qgood, res_p1_qbad=res_p1_qbad)
        st.session_state["deep_design"] = design_sel
    # Display metrics if present
    deep = st.session_state.get("deep_results", None)
    if deep:
        def se_p(p): return se_prop(p, n_sims_deep)
        cols = st.columns(6 if enable_safety else 5)
        t1 = deep['res_p0_qgood']['reject_rate']; cols[0].metric("Type I @ p₀, q_good", f"{t1:.3f}", f"±{1.96*se_p(t1):.3f}" if show_ci else None)
        pwr = deep['res_p1_qgood']['reject_rate']; cols[1].metric("Power @ p₁, q_good", f"{pwr:.3f}", f"±{1.96*se_p(pwr):.3f}" if show_ci else None)
        sd = deep['res_p1_qgood']['stop_dist']
        if isinstance(sd, pd.DataFrame) and not sd.empty:
            Nvals = sd['N_stop'].to_numpy(); probs = sd['Probability'].to_numpy()
            meanN = (Nvals*probs).sum(); varN = (((Nvals-meanN)**2*probs).sum()); seN = math.sqrt(varN/max(1,n_sims_deep))
        else:
            seN = 0.0
        cols[3].metric("ESS @ p₁, q_good", f"{deep['res_p1_qgood']['ess']:.1f}", f"±{1.96*seN:.1f}" if show_ci else None)
        cols[2].metric("ESS @ p₀, q_good", f"{deep['res_p0_qgood']['ess']:.1f}")
        estop = deep['res_p1_qgood']['early_stop_rate']; cols[4].metric("Early stop (any) @ p₁", f"{estop:.3f}", f"±{1.96*se_p(estop):.3f}" if show_ci else None)
        if enable_safety:
            sprob = deep['res_p1_qbad']['safety_stop_prob']; cols[5].metric("P(Safety stop) @ p₁, q_bad", f"{sprob:.3f}", f"±{1.96*se_p(sprob):.3f}" if show_ci else None)
        # ----- New: Deep‑Dive details (tables & plots) -----
        with st.expander("Deep‑Dive: Detailed tables & plots", expanded=True):
            c1, c2, c3 = st.columns(3)
            # Stopping distributions
            if isinstance(deep['res_p1_qgood'].get('stop_dist', None), pd.DataFrame):
                c1.markdown("**Stop distribution @ p₁, q_good**")
                c1.dataframe(deep['res_p1_qgood']['stop_dist'], use_container_width=True)
            if isinstance(deep['res_p0_qgood'].get('stop_dist', None), pd.DataFrame):
                c2.markdown("**Stop distribution @ p₀, q_good**")
                c2.dataframe(deep['res_p0_qgood']['stop_dist'], use_container_width=True)
            if enable_safety and isinstance(deep['res_p1_qbad'].get('stop_dist', None), pd.DataFrame):
                c3.markdown("**Stop distribution @ p₁, q_bad**")
                c3.dataframe(deep['res_p1_qbad']['stop_dist'], use_container_width=True)
            # By-look breakdown (efficacy & futility)
            looks_eff = design_sel['looks_eff']
            looks_fut = design_sel['looks_fut']
            # Success by look
            eff_succ = pd.DataFrame({"Look (eff)": looks_eff,
                                     "Early success prob": deep['res_p1_qgood'].get('eff_early_succ_by_look', [0.0]*len(looks_eff))})
            # Futility by look
            fut_prob = pd.DataFrame({"Look (fut)": looks_fut,
                                     "Early futility prob": deep['res_p1_qgood'].get('fut_early_by_look', [0.0]*len(looks_fut))})
            st.markdown("**By‑look breakdown @ p₁, q_good**")
            st.dataframe(eff_succ, use_container_width=True)
            st.dataframe(fut_prob, use_container_width=True)
            # Safety by look (+ final)
            if enable_safety:
                saf_vec = deep['res_p1_qgood'].get('saf_by_look', [])
                if saf_vec:
                    labels_saf = [f"{n}" for n in design_sel.get('looks_saf', [])] + ["FINAL"]
                    df_saf = pd.DataFrame({"Safety look": labels_saf, "Stop for safety": saf_vec})
                    st.dataframe(df_saf, use_container_width=True)
            # Stacked bar: reasons for early stopping @ p1,q_good
            if _HAS_PLOTLY:
                succ_r = deep['res_p1_qgood'].get('eff_early_succ_rate', 0.0)
                fut_r = deep['res_p1_qgood'].get('eff_early_fut_rate', 0.0)
                saf_r = deep['res_p1_qgood'].get('saf_early_rate', 0.0)
                fig_r = go.Figure()
                fig_r.add_bar(x=['@p1,q_good'], y=[succ_r], name='Early success')
                fig_r.add_bar(x=['@p1,q_good'], y=[fut_r], name='Early futility')
                if enable_safety:
                    fig_r.add_bar(x=['@p1,q_good'], y=[saf_r], name='Early safety')
                fig_r.update_layout(barmode='stack', title='Reasons for early stop @ p1,q_good',
                                    yaxis_title='Proportion', xaxis_title='Scenario')
                st.plotly_chart(fig_r, use_container_width=True)
        # ---- Compute OC/ESS curves at fixed q (and preview) ----
        with st.expander("Compute OC/ESS Curves (fixed q)", expanded=False):
            cA, cB = st.columns(2)
            with cA:
                p_min_c = st.number_input("Curves: p_min", 0.0, 1.0, max(0.0, p0-0.15), 0.01, key='curve_pmin')
                p_max_c = st.number_input("Curves: p_max", 0.0, 1.0, min(1.0, p1+0.15), 0.01, key='curve_pmax')
                p_points_c = st.slider("Curves: number of p points", 3, 25, 9, 1, key='curve_ppoints')
            with cB:
                q_fixed = st.number_input("Fixed q for curves", 0.0, 1.0, q_good, 0.01, key='curve_qfixed')
                n_sims_curves = st.number_input("Simulations per p point", 2000, 200000, max(20000, int(n_sims_deep/5)), 1000, key='curve_sims')
            if st.button("Compute OC/ESS curves", key='btn_curves'):
                rng = np.random.default_rng(seed_deep + 11)
                ps = np.linspace(p_min_c, p_max_c, p_points_c)
                Ueff = rng.uniform(size=(n_sims_curves, design_sel["N_total"]))
                Utox = rng.uniform(size=(n_sims_curves, design_sel["N_total"]))
                rows = []
                for pp in ps:
                    r = simulate_design_joint(design_sel, p_eff=pp, p_tox=q_fixed, U_eff=Ueff, U_tox=Utox)
                    rows.append(dict(p=pp, reject_rate=r['reject_rate'], ess=r['ess']))
                df_curves = pd.DataFrame(rows)
                st.session_state['oc_ess_curves_df'] = df_curves
                st.session_state['oc_ess_q_fixed'] = float(q_fixed)
        # Parse‑proof guard: preview curves if already computed earlier (single render per run)
        if _HAS_PLOTLY and 'oc_ess_curves_df' in st.session_state:
            _render_oc_ess_preview(st.session_state['oc_ess_curves_df'], st.session_state.get('oc_ess_q_fixed'))
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 4) OC Explorer (always‑on heatmap) + Automated Interpretation            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.write("### 4) OC Explorer")
with st.expander("Open OC Explorer", expanded=True):
    if 'deep_design' not in st.session_state:
        st.info("Run a deep‑dive first to lock a design.")
    else:
        design_sel = st.session_state['deep_design']
        colO1, colO2 = st.columns(2)
        with colO1:
            p_min = st.number_input("Grid: p_min", 0.0, 1.0, max(0.0, st.session_state.get('p0', p0)-0.20), 0.01, key='oc_pmin', help="Start of efficacy (p) grid.")
            p_max = st.number_input("Grid: p_max", 0.0, 1.0, min(1.0, st.session_state.get('p1', p1)+0.25), 0.01, key='oc_pmax', help="End of efficacy (p) grid.")
            p_points = st.slider("Number of p points", 3, 25, 9, 1, key='oc_ppoints')
        with colO2:
            q_min = st.number_input("Grid: q_min", 0.0, 1.0, max(0.0, (st.session_state.get('q_max', 0.10) or 0.10)-0.10), 0.01, key='oc_qmin', help="Start of toxicity (q) grid.")
            q_max_g = st.number_input("Grid: q_max", 0.0, 1.0, min(1.0, (st.session_state.get('q_max', 0.10) or 0.10)+0.20), 0.01, key='oc_qmax', help="End of toxicity (q) grid.")
            q_points = st.slider("Number of q points", 3, 25, 7, 1, key='oc_qpoints')
        n_sims_oc = st.number_input("Simulations per cell", 2000, 200000, 50000, 2000, key='oc_sims', help="Higher = more precise but slower.")
        seed_oc = st.number_input("Random seed (OC)", 1, None, st.session_state.get('seed', 2026)+4, 1, key='oc_seed')
        if st.button("Run OC Explorer", key='run_oc'):
            rng = np.random.default_rng(seed_oc)
            Ueff = rng.uniform(size=(n_sims_oc, design_sel['N_total']))
            Utox = rng.uniform(size=(n_sims_oc, design_sel['N_total']))
            ps = np.linspace(p_min, p_max, p_points)
            qs = np.linspace(q_min, q_max_g, q_points)
            oc_rows = []
            for pp in ps:
                for qq in qs:
                    r = simulate_design_joint(design_sel, p_eff=pp, p_tox=qq, U_eff=Ueff, U_tox=Utox)
                    oc_rows.append(dict(p=pp, q=qq,
                        reject_rate=r['reject_rate'], ess=r['ess'],
                        safety_stop_any=r['safety_stop_prob'], early_any=r['early_stop_rate']))
            df_grid = pd.DataFrame(oc_rows)
            st.session_state['oc_grid_df'] = df_grid
            nice = df_grid.copy()
            nice.rename(columns={'reject_rate': 'Pr(declare efficacy)', 'ess': 'ESS', 'safety_stop_any': 'Pr(safety stop)', 'early_any': 'Pr(early stop any)'}, inplace=True)
            nice = nice[['p','q','Pr(declare efficacy)','ESS','Pr(safety stop)','Pr(early stop any)']]
            nice = nice.sort_values(['q','p']).reset_index(drop=True)
            st.dataframe(nice, use_container_width=True)
            if _HAS_PLOTLY:
                piv = df_grid.pivot(index='q', columns='p', values='reject_rate')
                fig = px.imshow(piv.values, x=[f"{x:.2f}" for x in piv.columns], y=[f"{y:.2f}" for y in piv.index],
                                aspect='auto', origin='lower', color_continuous_scale='Viridis',
                                labels=dict(color='Probability of declaring efficacy'),
                                title='Probability of declaring efficacy across (p,q)')
                fig.update_xaxes(title_text='Efficacy rate p')
                fig.update_yaxes(title_text='Toxicity rate q')
                st.plotly_chart(fig, use_container_width=True)
        # Automated interpretation
        st.write("#### Automated Interpretation & Design Diagnostics")
        Ntot = design_sel['N_total']
        pow_target = st.session_state.get('power_min', 0.80)
        saf_warn_thr = 0.25
        if 'oc_grid_df' in st.session_state:
            df_grid = st.session_state['oc_grid_df']
            grp = df_grid.groupby('q')['reject_rate'].apply(lambda s: (s >= pow_target).sum()/len(s))
            robust_qs = grp[grp >= 0.6].index.tolist()
            fragile_qs = grp[grp < 0.6].index.tolist()
            saf_grp = df_grid.groupby('q')['safety_stop_any'].mean()
            high_safety_qs = saf_grp[saf_grp > saf_warn_thr].index.tolist()
            ess_mean = df_grid.groupby('p')['ess'].mean()
            ess_spike = (ess_mean > 0.9 * Ntot).sum() / len(ess_mean) > 0.3
            bullets = []
            if robust_qs:
                bullets.append(f"Design maintains power≥{pow_target:.2f} across most p for q≤{max(robust_qs):.2f}.")
            else:
                bullets.append("Power region is narrow; few (p,q) cells reach target power.")
            if high_safety_qs:
                bullets.append(f"Safety stopping becomes frequent (> {saf_warn_thr:.0%}) at q≥{min(high_safety_qs):.2f}.")
            else:
                bullets.append("Safety-stop probability remains moderate across evaluated q.")
            bullets.append("ESS is " + ("often near max N" if ess_spike else "well below max N for much of the grid") + ".")
            for b in bullets:
                st.write("• " + b)
            warn_msgs = []
            if fragile_qs and min(fragile_qs) <= (q_max_g if 'q_max_g' in locals() else df_grid['q'].max()):
                warn_msgs.append("Power falls below target for many p at moderate q levels.")
            if high_safety_qs:
                warn_msgs.append("Safety stop rate high in upper‑q region — consider stricter θ_tox or fewer safety looks.")
            if ess_spike:
                warn_msgs.append("ESS frequently approaches N — consider adding futility or relaxing θ_final.")
            if warn_msgs:
                st.warning("\n".join(["⚠ " + m for m in warn_msgs]))
            with st.expander("Extended interpretation (details)"):
                robust_text = (f"q ≤ {max(robust_qs):.2f}" if len(robust_qs) > 0 else "no robust q-levels")
                fragile_text = (f"q ≥ {min(fragile_qs):.2f}" if len(fragile_qs) > 0 else "no fragile q-levels")
                safety_q = (min(high_safety_qs) if len(high_safety_qs) > 0 else q_max_g)
                ess_txt = ('Large portions of the grid require near‑max N.' if ess_spike else 'ESS stays moderate for many scenarios.')
                body = ("**Robust region (indicative):** " + robust_text + "; "
                        "**Fragile region:** " + fragile_text + ".\n\n"
                        "**Safety:** Above q≈" + f"{safety_q:.2f}" + ", safety stops become common.\n\n"
                        "**ESS:** " + ess_txt)
                st.markdown(body)
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ PDF EXPORT HELPERS                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def make_design_pdf(design: Dict, deep_results: Optional[Dict], compare_df: Optional[pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    x0, y = 2*cm, H - 2*cm
    def line(text: str, dy: float = 0.6*cm, bold: bool = False):
        nonlocal y
        if y < 3*cm:
            c.showPage(); y = H - 2*cm
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11 if bold else 10)
        c.drawString(x0, y, text)
        y -= dy
    line("Bayesian Single-Arm Design – Summary (v3.1.5q)", bold=True)
    line("")
    line("Design settings", bold=True)
    if isinstance(design, dict):
        line(f"N_total: {design.get('N_total')}")
        line(f"Look points — efficacy: {design.get('looks_eff')} + [FINAL]")
        line(f"Look points — futility: {design.get('looks_fut')} + [FINAL]")
        if design.get('looks_saf') is not None:
            line(f"Look points — safety: {design.get('looks_saf')} + [FINAL]")
        line(f"Run-in (eff/saf/fut): {design.get('run_in_eff',0)} / {design.get('run_in_saf',0)} / {design.get('run_in_fut', design.get('run_in_eff',0))}")
        line(f"Final evaluation at N: {design.get('N_total')}")
        line(f"Efficacy Beta(a0,b0): {design.get('a0')}, {design.get('b0')}")
        if 'safety' in design:
            s = design['safety']
            line(f"Safety Beta(a_t0,b_t0): {s.get('a_t0')}, {s.get('b_t0')}")
            line(f"q_max / θ_tox: {s.get('q_max')} / {s.get('theta_tox')}")
        line(f"p0 / p1: {design.get('p0')} / {design.get('p1')}")
        line(f"θ_final / θ_interim / c_futility: {design.get('theta_final')} / {design.get('theta_interim')} / {design.get('c_futility')}")
        line(f"s_min_final: {design.get('s_min_final')}")
    if deep_results is not None and isinstance(deep_results, dict):
        line("")
        line("Deep-dive key metrics", bold=True)
        for k, r in deep_results.items():
            if isinstance(r, dict):
                rr = r.get('reject_rate', float('nan'))
                ess = r.get('ess', float('nan'))
                estop = r.get('early_stop_rate', float('nan'))
                line(f"{k}: Pr(declare eff)={rr:.3f}, ESS={ess:.1f}, Early(any)={estop:.3f}")
    if compare_df is not None and isinstance(compare_df, pd.DataFrame) and not compare_df.empty:
        line("")
        line("Compare by N (first 10)", bold=True)
        for _, row in compare_df.head(10).iterrows():
            line(f"N={row['N']}, Pow={row['Power @p1,q_good']:.3f}, ESS1={row['ESS @p1']:.1f}")
    c.showPage()
    c.save()
    return buf.getvalue()
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 5) Export settings and results (JSON / CSVs / PDF)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.write("### 5) Export settings and results (JSON / CSVs / PDF)")
export_bundle = {
    "design": st.session_state.get('deep_design', {}),
    "screening_table": None if 'df_screen' not in locals() or df_screen.empty else df_screen.to_dict(orient='list'),
    "deep_dive": st.session_state.get('deep_results', None),
    "compare": st.session_state.get('compare_df', None),
    "oc_grid": None if 'oc_grid_df' not in st.session_state else st.session_state['oc_grid_df'].to_dict(orient='list'),
    "oc_ess_curves": None if 'oc_ess_curves_df' not in st.session_state else st.session_state['oc_ess_curves_df'].to_dict(orient='list'),
}
json_bytes = json.dumps(export_bundle, default=lambda o: o if isinstance(o, (int,float,str,bool,type(None))) else str(o)).encode('utf-8')
st.download_button("Download JSON bundle", data=json_bytes, file_name="design_and_results_v3_1_5q.json", mime="application/json")
buf = io.BytesIO()
with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
    if 'df_screen' in locals() and not df_screen.empty:
        zf.writestr("screening.csv", df_screen.to_csv(index=False))
    deep = st.session_state.get('deep_results', None)
    if isinstance(deep, dict):
        rows = []
        for k,v in deep.items():
            if isinstance(v, dict) and 'stop_dist' in v and isinstance(v['stop_dist'], pd.DataFrame):
                zf.writestr(f"deep_{k}_stopdist.csv", v['stop_dist'].to_csv(index=False))
            if isinstance(v, dict):
                rows.append({"scenario": k, **{kk: vv for kk, vv in v.items() if kk != 'stop_dist'}})
        if rows:
            zf.writestr("deep_summary.csv", pd.DataFrame(rows).to_csv(index=False))
    if 'compare_df' in st.session_state:
        zf.writestr("compare.csv", st.session_state['compare_df'].to_csv(index=False))
    if 'oc_grid_df' in st.session_state:
        zf.writestr("oc_grid.csv", st.session_state['oc_grid_df'].to_csv(index=False))
    if 'oc_ess_curves_df' in st.session_state:
        zf.writestr("oc_ess_curves.csv", st.session_state['oc_ess_curves_df'].to_csv(index=False))
st.download_button("Download ZIP (CSVs)", data=buf.getvalue(), file_name="design_results_v3_1_5q_csv.zip", mime="application/zip")
if st.button("Download protocol‑ready PDF", key='download_pdf'):
    design_pdf = st.session_state.get('deep_design', None)
    pdf_bytes = make_design_pdf(design_pdf or {}, st.session_state.get('deep_results', None), st.session_state.get('compare_df', None))
    st.download_button("Click to download PDF", data=pdf_bytes, file_name="design_summary_v3_1_5q.pdf", mime="application/pdf")
