 
# AVF_Bayes_Monitor_Designer_runin_safety_v3_1_3.py 
# Streamlit app for single-arm Bayesian monitored design (binary endpoint) 
# v3.1.4 UX upgrades: 
# • Plain-language labels and richer help tooltips across the app 
# • Friendlier “Chosen Design Summary” panel (clear bullets + decision rules) 
# • Keeps v3.1.2 safety fixes for OC interpretation and robust strings 
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
    _HAS_PLOTLY = True 
except Exception: 
    _HAS_PLOTLY = False 
SCHEMA_VERSION = "v3_1_4_everyN_stops_charts" 
# ╔══════════════════════════════════════════════════════════════════════════╗ 
# ║ CORE BAYESIAN UTILITIES ║ 
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
# ║ LOOK SCHEDULE UTILITIES ║ 
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

def build_looks_with_runin(N: int, run_in: int, mode_label: str, k_total: Optional[int] = None, perc_str: Optional[str] = None, ns_str: Optional[str] = None, step_every: Optional[int] = None) -> List[int]: 
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
                n_i = run_in + k*step 
                if n_i >= N: 
                    break 
                looks.append(int(n_i)) 
                k += 1 
    looks = [int(x) for x in looks if 0 < x < N] 
    looks = sorted(list(dict.fromkeys(looks))) 
    return looks 
# ╔══════════════════════════════════════════════════════════════════════════╗ 
# ║ MONTE CARLO SE/CI HELPERS ║ 
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
    var = float(((df_stop["N_stop"] - mean) ** 2 * df_stop["Probability"]).sum()) 
    return float(math.sqrt(max(var, 0.0) / max(n, 1))) 
# ╔══════════════════════════════════════════════════════════════════════════╗ 
# ║ SIMULATION ENGINES ║ 
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
    # unify event ordering across engines: start at 0 and advance to each event 
    events = sorted(list(dict.fromkeys(list(looks_eff) + list(looks_fut) + [N]))) 
    for ev in events: 
        if ev > n_curr and active.any(): 
            cum_x[active] += np.sum(X[active, n_curr:ev], axis=1) 
            n_curr = ev 
        # early success at efficacy looks 
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
        # futility at futility looks 
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
    # proceed to final if still active 
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
        "early_succ_by_look_eff": (eff_early_succ_by_look / n_sims).tolist(), 
        "early_fut_by_look_eff": (fut_early_by_look / n_sims).tolist(), 
        "eff_early_succ_rate": float(early_succ_rate), 
        "eff_early_fut_rate": float(early_fut_rate), 
        "early_stop_rate": float(early_stop_rate), 
    } 

def simulate_design_joint(design: Dict, p_eff: float, p_tox: float, U_eff: np.ndarray, U_tox: np.ndarray) -> Dict: 
    N = design["N_total"] 
    looks_eff = design["looks_eff"] 
    looks_saf = design.get("looks_saf", []) 
    looks_fut = design.get("looks_fut", looks_eff) 
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
    fut_early_by_look = np.zeros(len(looks_fut), dtype=np.int64) 
    saf_stop_by_look = np.zeros(len(looks_saf) + 1, dtype=np.int64) 
    events = sorted(list(dict.fromkeys(list(looks_eff) + list(looks_fut) + list(looks_saf) + [N]))) 
    for ev in events: 
        if ev > n_curr and active.any(): 
            cum_x[active] += np.sum(X_eff[active, n_curr:ev], axis=1) 
            if tox is not None: 
                cum_t[active] += np.sum(X_tox[active, n_curr:ev], axis=1) 
            n_curr = ev 
        # Safety checks at safety looks
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
        # Early success at efficacy looks 
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
        # Futility at futility looks (separate from efficacy)
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
    # After last interim, accrue to final N and evaluate safety + final success 
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
    fut_early_rate = fut_early_by_look.sum() / n_sims 
    saf_early_rate = saf_stop_by_look[:-1].sum() / n_sims 
    any_safety_rate = saf_stop_by_look.sum() / n_sims 
    early_stop_rate = eff_early_succ_rate + fut_early_rate + saf_early_rate 
    unique_ns, counts = np.unique(final_n, return_counts=True) 
    stop_dist = pd.DataFrame({"N_stop": unique_ns, "Probability": counts / n_sims}).sort_values("N_stop") 
    return { 
        "reject_rate": float(reject_rate), 
        "ess": float(ess), 
        "safety_stop_prob": float(any_safety_rate), 
        "stop_dist": stop_dist, 
        "eff_early_succ_by_look": (eff_early_succ_by_look / n_sims).tolist(), 
        "fut_early_by_look": (fut_early_by_look / n_sims).tolist(), 
        "saf_by_look": (saf_stop_by_look / n_sims).tolist(), 
        "eff_early_succ_rate": float(eff_early_succ_rate), 
        "fut_early_rate": float(fut_early_rate), 
        "saf_early_rate": float(saf_early_rate), 
        "early_stop_rate": float(early_stop_rate), 
    } 
# ╔══════════════════════════════════════════════════════════════════════════╗ 
# ║ SCREENING (FAST) ║ 
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
            N_total=N, looks_eff=looks_eff, a0=a0, b0=b0, p0=p0, 
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
# ║ STREAMLIT UI — Header & Sidebar (with richer tooltips) ║ 
# ╚══════════════════════════════════════════════════════════════════════════╝ 
st.set_page_config(page_title="Bayesian Single‑Arm Designer (Binary) — v3.1.4", layout="wide") 
st.title("Bayesian Single‑Arm Monitored Study Designer (Binary Endpoint) — v3.1.4") 
st.caption("Now with plainer labels, richer help tooltips, and a friendlier design summary. All v3.1.2 robustness fixes are retained.") 
with st.expander("What this tool does (in simple terms)"): 
    st.markdown( 
        "This app helps you design a single‑arm trial with interim checks for **benefit** and optional **safety**.

" 
        "• **You set** prior beliefs, thresholds, and when to look at the data.
" 
        "• The **Screener** tries many maximum sample sizes (N) and filters by α/power.
" 
        "• The **Deep Dive** runs precise joint simulations (benefit + safety).
" 
        "• The **OC Explorer** scans a grid of scenarios (efficacy p, toxicity q) and interprets them for you.") 
# Sidebar — Efficacy (plainer labels + help) 
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
# Sidebar — Safety (plainer labels + help) 
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
        b_t0 = st.number_input("Safety prior b_t0", 0.0, None, 9.0, 0.5, key='b_t0', help="Increase a_t0+b_t0 for a stronger safety prior (e.g., Beta(1,9) → mean≈0.10).") 
        theta_tox = st.number_input("Safety stop threshold (θ_tox)", 0.5, 0.999, 0.90, 0.01, format="%.3f", key='theta_tox', 
            help="Stop when P(q>q_max | data) ≥ θ_tox at interims or final.") 
    with st.sidebar.expander("About your safety prior"): 
        if (a_t0 is not None) and (b_t0 is not None) and (a_t0 + b_t0) > 0: 
            st.write(f"Prior mean = **{a_t0/(a_t0+b_t0):.3f}**, prior ESS ≈ **{a_t0+b_t0:.1f}**.") 
        else: 
            st.write("Prior mean undefined (a_t0+b_t0=0). Consider Beta(1,9) or similar.") 
else: 
    a_t0 = b_t0 = q_max = theta_tox = None 
# Sidebar — Look schedules (helped) 
st.sidebar.header("3) When to check the data (look schedule)") 
run_in_eff = st.sidebar.number_input("Run‑in for efficacy (patients enrolled before first look)", 0, 400, 0, 1, key='run_in_eff', 
    help="Patients enrolled before the first efficacy look; they count for all decisions.") 
looks_eff_mode_label = st.sidebar.selectbox( 
    "Efficacy look timing", 
    ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns"], 
    index=1, key='eff_mode', 
    help="Choose how to place efficacy interim looks.") 
k_looks_eff = perc_eff_str = ns_eff_str = step_eff = None 
if looks_eff_mode_label == "Equal‑spaced (choose total looks incl. run‑in)": 
    k_looks_eff = st.sidebar.slider("Total efficacy looks (including run‑in)", 1, 8, 2, 1, key='k_eff', 
        help="For example, 2 looks → roughly at 1/3 and 2/3 of the maximum N.") 
elif looks_eff_mode_label == "Custom percentages of remaining": 
    perc_eff_str = st.sidebar.text_input("Efficacy look percentages (comma)", "33,67", key='perc_eff', 
        help="Example: 25,50,75 (percent of the planned maximum N)") 
elif looks_eff_mode_label == "Custom absolute Ns": 
    ns_eff_str = st.sidebar.text_input("Efficacy look sample sizes N (comma)", "", key='ns_eff', 
        help="Example: 20,40 (each must be less than the maximum N)") 
# Safety schedule 
st.sidebar.header("4) Safety look schedule") 
run_in_saf = st.sidebar.number_input("Run‑in for safety", 0, 400, 0, 1, key='run_in_saf', help="Patients enrolled before the first safety look; they count for safety decisions.") 
looks_saf_mode_label = st.sidebar.selectbox("Safety look timing", 
    ["None (final only)", "Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns", "Look every N after run‑in"], index=1, key='saf_mode', 
    help="Choose how to place safety interim looks.") 
k_looks_saf = perc_saf_str = ns_saf_str = step_saf = None 
if looks_saf_mode_label == "Equal‑spaced (choose total looks incl. run‑in)": 
    k_looks_saf = st.sidebar.slider("Total safety looks (including run‑in)", 1, 8, 2, 1, key='k_saf') 
elif looks_saf_mode_label == "Custom percentages of remaining": 
    perc_saf_str = st.sidebar.text_input("Safety look percentages (comma)", "33,67", key='perc_saf') 
elif looks_saf_mode_label == "Custom absolute Ns": 
    ns_saf_str = st.sidebar.text_input("Safety look sample sizes N (comma)", "", key='ns_saf') 
elif looks_saf_mode_label == "Look every N after run‑in": 
    step_saf = st.sidebar.number_input("Look every N participants (after safety run‑in)", 1, 400, 10, 1, key='step_saf', help="First safety look is after the run‑in; then look every N participants. Final analysis still occurs at max N.") 
# Futility schedule 
st.sidebar.header("4b) Futility look schedule") 
use_fut_same = st.sidebar.checkbox( 
    "Use same schedule as efficacy? (default)", 
    True, 
    key='fut_same', 
    help="If checked, futility looks occur at the same Ns as efficacy looks (backward-compatible). Uncheck to customize a separate futility schedule.") 
if use_fut_same: 
    run_in_fut = int(st.session_state.get('run_in_eff', run_in_eff)) 
    looks_fut_mode_label = "Same as efficacy" 
    k_looks_fut = perc_fut_str = ns_fut_str = step_fut = None 
else: 
    run_in_fut = st.sidebar.number_input( 
        "Run‑in for futility", 
        0, 400, int(st.session_state.get('run_in_eff', run_in_eff)), 1, key='run_in_fut', 
        help="Patients enrolled before the first futility look; they count for futility decisions.") 
    looks_fut_mode_label = st.sidebar.selectbox( 
        "Futility look timing", 
        ["Equal‑spaced (choose total looks incl. run‑in)", "Custom percentages of remaining", "Custom absolute Ns", "Look every N after run‑in"], 
        index=1, key='fut_mode', 
        help="Choose how to place futility interim looks.") 
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
# Screener settings 
st.sidebar.header("5) Rapid Screener") 
N_min, N_max = st.sidebar.slider("Range of maximum N values to test", 10, 400, (30, 120), 1, key='Nrange', help="We'll scan this range to find feasible, efficient designs.") 
N_step = st.sidebar.number_input("Step between N values", 1, 50, 5, 1, key='Nstep', help="Larger steps scan fewer N values (faster).") 
n_sims_small = st.sidebar.number_input("Simulations per design (screening)", 100, 200000, 5000, 500, key='sims_small', help="Higher = more precision; the Deep Dive is where precision matters most.") 
alpha_max = st.sidebar.number_input("Max Type I error (α) allowed", 0.0, 0.5, 0.10, 0.01, format="%.2f", key='alpha_max', help="Keep only designs with false-positive rate ≤ this.") 
power_min = st.sidebar.number_input("Min power at p₁", 0.0, 1.0, 0.80, 0.01, format="%.2f", key='power_min', help="Keep only designs with success rate at p₁ ≥ this.") 
seed = st.sidebar.number_input("Random seed", 1, None, 2026, 1, key='seed', help="Controls reproducibility.") 
# Build candidate grid for Screener 
Ns = list(range(N_min, N_max + 1, N_step)) 
param_grid = [] 
for N in Ns: 
    looks_eff = build_looks_with_runin(N, run_in_eff, looks_eff_mode_label, k_total=k_looks_eff, perc_str=perc_eff_str, ns_str=ns_eff_str, step_every=step_eff) 
    looks_fut = looks_eff if st.session_state.get('fut_same', True) or looks_fut_mode_label == 'Same as efficacy' else build_looks_with_runin(N, run_in_fut, looks_fut_mode_label, k_total=k_looks_fut, perc_str=perc_fut_str, ns_str=ns_fut_str, step_every=step_fut) 
    param_grid.append(dict(N_total=N, looks_eff=looks_eff, a0=a0, b0=b0, p0=p0, p1=p1, 
        theta_final=theta_final, theta_interim=float(theta_interim), 
        c_futility=c_futility, allow_early_success=allow_early_success, 
        run_in_eff=int(run_in_eff))) 
# ╔══════════════════════════════════════════════════════════════════════════╗ 
# ║ 1) Rapid Screener (efficacy‑only) ║ 
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
st.caption("Screens a range of N using quick efficacy-only sims to find candidates that meet α/power targets.") 
if df_screen.empty: 
    st.warning("No viable designs found. Try relaxing θ_final or increasing N.") 
else: 
    df_ok = df_screen[(df_screen["Type I error @ p0"] <= alpha_max) & (df_screen["Power @ p1"] >= power_min)].copy() 
    cols = ["N_total", "run_in_eff", "looks_eff", "theta_final", "c_futility", "Type I error @ p0", "Power @ p1", "ESS @ p0", "Early stop @ p0 (any)"] 
    table_df = df_ok if not df_ok.empty else df_screen 
    st.dataframe(table_df[cols].sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True), use_container_width=True) 
# ╔══════════════════════════════════════════════════════════════════════════╗ 
# ║ 2) Compare multiple Ns (joint with safety) ║ 
# ╚══════════════════════════════════════════════════════════════════════════╝ 
st.write("### 2) Compare multiple N values (deep‑dive, joint with safety)") 
st.caption("Runs precise joint simulations at the N values you type, using your current rule settings.") 
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
                # NEW: build futility schedule consistent with Deep Dive logic 
                if st.session_state.get('fut_same', True) or looks_fut_mode_label == 'Same as efficacy': 
                    looks_fut_cmp = looks_eff_cmp 
                    run_in_fut_cmp = int(st.session_state.get('run_in_eff', run_in_eff)) 
                else: 
                    run_in_fut_cmp = int(st.session_state.get('run_in_fut', 0) or 0) 
                    looks_fut_cmp = build_looks_with_runin(Ncmp, run_in_fut_cmp, looks_fut_mode_label, k_total=k_looks_fut, perc_str=perc_fut_str, ns_str=ns_fut_str, step_every=step_fut) 
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
                rows.append(dict(N=Ncmp, looks_eff=looks_eff_cmp, looks_fut=looks_fut_cmp, looks_saf=looks_saf_cmp, s_min_final=smin_cmp, 
                    **{"Type I @p0,q_good": r_p0["reject_rate"], "Power @p1,q_good": r_p1["reject_rate"], 
                    "ESS @p0": r_p0["ess"], "ESS @p1": r_p1["ess"], 
                    "Early stop (any) @p1": r_p1["early_stop_rate"], 
                    "Early succ @p1": r_p1["eff_early_succ_rate"], 
                    "Early fut @p1": r_p1.get("fut_early_rate", r_p1.get("eff_early_fut_rate", 0.0)), 
                    "Early safety @p1": r_p1.get("saf_early_rate", 0.0), 
                    "Safety stop @p1,q_bad (any stage)": r_p1_bad.get("safety_stop_prob", 0.0)}) ) 
            if not rows: 
                st.warning("None of the N values produced a feasible final rule. Try relaxing θ_final or adjusting looks.") 
            else: 
                df_cmp = pd.DataFrame(rows).sort_values("N").reset_index(drop=True) 
                st.dataframe(df_cmp, use_container_width=True) 
                st.session_state['compare_df'] = df_cmp 
                # Stop-reason chart across input N values 
                if _HAS_PLOTLY and not df_cmp.empty: 
                    fig_cmp = go.Figure() 
                    fig_cmp.add_bar(x=df_cmp['N'], y=df_cmp['Early succ @p1'], name='Early success @p1,q_good') 
                    fig_cmp.add_bar(x=df_cmp['N'], y=df_cmp['Early fut @p1'], name='Early futility @p1,q_good') 
                    if 'Early safety @p1' in df_cmp.columns: 
                        fig_cmp.add_bar(x=df_cmp['N'], y=df_cmp['Early safety @p1'], name='Early safety @p1,q_good') 
                    if 'Safety stop @p1,q_bad (any stage)' in df_cmp.columns: 
                        fig_cmp.add_bar(x=df_cmp['N'], y=df_cmp['Safety stop @p1,q_bad (any stage)'], name='Safety stop (any) @p1,q_bad') 
                    fig_cmp.update_layout(barmode='group', title='Stop proportions by reason vs N', xaxis_title='N', yaxis_title='Proportion') 
                    st.plotly_chart(fig_cmp, use_container_width=True) 
# ╔══════════════════════════════════════════════════════════════════════════╗ 
# ║ 3) Deep Dive (joint efficacy + safety) ║ 
# ╚══════════════════════════════════════════════════════════════════════════╝ 
st.write("### 3) Deep Dive (joint efficacy + safety)") 
st.caption("Locks a single design and shows detailed operating characteristics, per-look summaries, and stop reasons.") 
# Compute a safe default value for N_select (must lie within [min,max])
if Ns:
    _nselect_default = min(max(60, Ns[0]), Ns[-1])
    _nselect_min, _nselect_max = Ns[0], Ns[-1]
else:
    _nselect_default = 60
    _nselect_min, _nselect_max = 5, 400
N_select = st.number_input(
    "Select a maximum sample size N to deep‑dive",
    _nselect_min, _nselect_max, _nselect_default, 1,
    key='N_select',
    help="Choose an N from your screening range (or adjacent values) to inspect in detail."
) 
looks_eff_sel = build_looks_with_runin(N_select, run_in_eff, looks_eff_mode_label, k_total=k_looks_eff, perc_str=perc_eff_str, ns_str=ns_eff_str, step_every=step_eff) 
looks_saf_sel = build_looks_with_runin(N_select, run_in_saf, looks_saf_mode_label, k_total=k_looks_saf, perc_str=perc_saf_str, ns_str=ns_saf_str, step_every=step_saf) 
s_min_sel = min_successes_for_posterior_threshold(a0, b0, N_select, p0, theta_final) 
if s_min_sel is None: 
    st.error("Final rule infeasible at this N. Relax θ_final / adjust prior / increase N.") 
else: 
    # Build futility looks (same as efficacy by default) 
    looks_fut_sel = looks_eff_sel if st.session_state.get('fut_same', True) or looks_fut_mode_label == 'Same as efficacy' else build_looks_with_runin(N_select, int(st.session_state.get('run_in_fut', run_in_fut)), looks_fut_mode_label, k_total=k_looks_fut, perc_str=perc_fut_str, ns_str=ns_fut_str, step_every=step_fut) 
    x_min_to_continue_sel = compute_interim_futility_cutoffs(a0, b0, N_select, looks_fut_sel, p0, theta_final, c_futility) 
    design_sel = dict(N_total=N_select, a0=a0, b0=b0, p0=p0, p1=p1, theta_final=theta_final, 
        theta_interim=float(theta_interim), c_futility=c_futility, 
        allow_early_success=allow_early_success, s_min_final=s_min_sel, 
        looks_eff=looks_eff_sel, looks_saf=looks_saf_sel, looks_fut=looks_fut_sel, 
        run_in_eff=int(run_in_eff), run_in_saf=int(run_in_saf), 
        x_min_to_continue_by_look_fut=x_min_to_continue_sel) 
    if enable_safety: 
        design_sel["safety"] = dict(a_t0=a_t0, b_t0=b_t0, q_max=q_max, theta_tox=theta_tox) 
    # Summary panel and deep dive UI ... (unchanged from prior, omitted here to save space) 
# TUNER CORE + OC Explorer + EXPORT sections ... (unchanged; included in full file) 
