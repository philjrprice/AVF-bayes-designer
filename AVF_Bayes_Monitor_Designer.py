# app.py
# Streamlit app for single-arm Bayesian monitored design (binary endpoint)
# Rapid screener + deep-dive simulation WITH optional safety monitoring
#
# Author: M365 Copilot for Phil
# Features:
#  - Wide range of interim schedules: equal-spaced (1..8), custom percentages, or custom absolute Ns.
#  - Separate posterior success thresholds for interim early success (θ_interim) and final (θ_final).
#  - Optional safety (toxicity) monitoring via Beta–Binomial rule at interims and final.
#  - Plotly optional (falls back to Streamlit charts).
#  - Screening uses small sims with common random numbers; deep-dive uses larger sims.

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import beta
from scipy.special import betaln, comb

# ----- Plotly is optional (graceful fallback to Streamlit charts) -----
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


# ----------------------------
# Core math utilities
# ----------------------------

def beta_posterior_params(a0, b0, x, n):
    """Posterior Beta(a0+x, b0+n-x) after observing x successes out of n."""
    return a0 + x, b0 + (n - x)

def posterior_prob_p_greater_than(p_cut, a_post, b_post):
    """Compute P(p > p_cut | data) under Beta(a_post, b_post)."""
    return 1.0 - beta.cdf(p_cut, a_post, b_post)

def min_successes_for_posterior_threshold(a0, b0, N, p0, theta_final):
    """
    Smallest total successes s_min in [0, N] such that:
    P(p > p0 | Beta(a0+s, b0+N-s)) >= theta_final
    Monotone in s, so use binary search.
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
    if ans == N + 1:
        return None  # No number of successes can meet the threshold
    return int(ans)

def log_beta_binomial_pmf(y, m, a, b):
    """
    Log PMF of Beta-Binomial(m, a, b).
    P(Y=y) = C(m, y) * B(y+a, m-y+b) / B(a, b)
    """
    if y < 0 or y > m:
        return -np.inf
    return np.log(comb(m, y)) + betaln(y + a, m - y + b) - betaln(a, b)

def beta_binomial_cdf_upper_tail(y_min, m, a, b):
    """
    Sum_{y >= y_min} Beta-Binomial PMF(m, a, b).
    If y_min <= 0 -> 1
    If y_min > m -> 0
    """
    if y_min <= 0:
        return 1.0
    if y_min > m:
        return 0.0
    ys = np.arange(y_min, m + 1)
    logs = np.array([log_beta_binomial_pmf(int(y), m, a, b) for y in ys])
    mlog = np.max(logs)
    return float(np.exp(mlog) * np.sum(np.exp(logs - mlog)))

def predictive_prob_of_final_success(a0, b0, N_total, x_curr, n_curr, p0, theta_final):
    """
    Predictive probability (over future data) that final posterior criterion
    (P(p > p0 | final data) >= theta_final) will be met, given current x_curr, n_curr.
    Exact under Beta-Binomial predictive with posterior Beta(a0+x_curr, b0+n_curr-x_curr).
    """
    a_post, b_post = beta_posterior_params(a0, b0, x_curr, n_curr)
    m_remain = N_total - n_curr
    s_min = min_successes_for_posterior_threshold(a0, b0, N_total, p0, theta_final)
    if s_min is None:
        return 0.0
    y_needed = s_min - x_curr
    return beta_binomial_cdf_upper_tail(y_needed, m_remain, a_post, b_post)

def compute_interim_futility_cutoffs(a0, b0, N_total, looks, p0, theta_final, c_futility):
    """
    For each interim look n in looks (n < N_total), compute the minimum current successes x_min_to_continue
    such that PPoS >= c_futility. (Monotone in x.)
    Returns dict: look_n -> x_min_to_continue (None => never continue).
    """
    cutoffs = {}
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


# ----------------------------
# Simulation engines
# ----------------------------

def simulate_design(design, p, U):
    """
    Efficacy-only simulation (used in SCREENING for speed).
    Simulate one design under Bernoulli(p) with shared uniforms U (n_sims x N).
    Returns reject_rate, ess, stop_dist.
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
    stop_by_look_counts = np.zeros(len(looks), dtype=np.int64)

    for li, look_n in enumerate(looks):
        add = look_n - n_curr
        if add > 0:
            cum_x[active] += np.sum(X[active, n_curr:look_n], axis=1)
            n_curr = look_n

        # Early success (efficacy)
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
                stop_by_look_counts[li] += early_succ.sum()

        if not active.any():
            break

        # Futility (PPoS continue threshold)
        x_min = x_min_to_continue.get(look_n, None)
        if x_min is None:
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

    # Final
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

    unique_ns, counts = np.unique(final_n, return_counts=True)
    stop_dist = pd.DataFrame({"N_stop": unique_ns, "Probability": counts / n_sims}).sort_values("N_stop")

    return {"reject_rate": float(reject_rate),
            "ess": float(ess),
            "stop_probs_by_look": (stop_by_look_counts / n_sims).tolist(),
            "stop_dist": stop_dist}


def simulate_design_with_safety(design, p_eff, p_tox, U_eff, U_tox):
    """
    Joint efficacy + safety simulation (used in DEEP DIVE).
    Safety rule: stop if P(q > q_max | data) >= theta_tox at interims/final.
    Returns reject_rate (declare efficacy success), ess, safety_stop_prob, stop_dist.
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
    safety_stop_counts = np.zeros(len(looks) + 1, dtype=np.int64)  # per interim + final

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
                safety_stop_counts[li] += unsafe.sum()

        if not active.any():
            break

        # 2) Early efficacy success
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

        # 3) Futility by PPoS
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

    # Final for those still active
    if active.any():
        add = N - n_curr
        if add > 0:
            cum_x[active] += np.sum(X_eff[active, n_curr:N], axis=1)
            if tox is not None:
                cum_t[active] += np.sum(X_tox[active, n_curr:N], axis=1)
            n_curr = N

        # Safety at final
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

        # Efficacy final decision
        if active.any():
            succ_final = (cum_x[active] >= s_min_final)
            idx_active = np.where(active)[0]
            success[idx_active[succ_final]] = True
            final_n[idx_active] = N

    reject_rate = success.mean()
    ess = final_n.mean()
    safety_stop_prob = safety_stop_counts.sum() / n_sims if tox is not None else 0.0

    unique_ns, counts = np.unique(final_n, return_counts=True)
    stop_dist = pd.DataFrame({"N_stop": unique_ns, "Probability": counts / n_sims}).sort_values("N_stop")

    return {
        "reject_rate": float(reject_rate),
        "ess": float(ess),
        "safety_stop_prob": float(safety_stop_prob),
        "stop_dist": stop_dist
    }


# ----------------------------
# Screening (efficacy-only for speed)
# ----------------------------

def shortlist_designs(param_grid, n_sims_small, seed, U=None):
    """
    Screen designs quickly using small simulation with common random numbers U.
    Efficacy-only (no safety) to keep this stage fast.
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
            "s_min_final": s_min,
            "x_min_to_continue": x_min_to_continue,
            "_design": design
        })
        designs_built.append(design)

    df = pd.DataFrame(rows)
    return df, designs_built


# ----------------------------
# Plotting helper (Plotly or fallback)
# ----------------------------

def plot_lines(df, x, y, title):
    if _HAS_PLOTLY:
        fig = px.line(df, x=x, y=y, title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader(title)
        chart_df = df[[x, y]].set_index(x)
        st.line_chart(chart_df)


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Bayesian Single-Arm Design (Binary) – Rapid Screener & Simulator", layout="wide")

st.title("Single-Arm Bayesian Monitored Study Designer (Binary Endpoint)")

with st.expander("Assumptions & Notes", expanded=False):
    st.markdown(
        """
- **Endpoint**: binary (e.g., response yes/no).
- **Prior**: Beta(a₀, b₀). Posterior after x successes in n is Beta(a₀+x, b₀+n−x).
- **Final decision**: declare success if P(p > p₀ | data) ≥ θ_final.
- **Interim futility**: stop if predictive probability of final success (PPoS) < c_futility.
- **Predictive probability** is computed **exactly** under Beta–Binomial predictive distribution.
- **Early success at interims**: if enabled, stop early if P(p > p₀ | data) ≥ **θ_interim** (which can differ from θ_final).
- **Safety (optional)**: stop if P(q > q_max | data) ≥ θ_tox at interims/final.
- Rapid **screening** uses small simulations with **common random numbers** (efficacy-only for speed).
- **Deep dive**: larger joint simulations (efficacy + safety) for precise operating characteristics.
        """
    )

# Sidebar: global inputs
st.sidebar.header("Design Inputs")
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    p0 = st.number_input("Null rate p0", min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")
    a0 = st.number_input("Prior a₀ (efficacy)", min_value=0.0, value=1.0, step=0.5)
    theta_final = st.number_input("θ_final (posterior threshold at final)", min_value=0.5, max_value=0.999, value=0.95, step=0.01, format="%.3f")
with col_sb2:
    p1 = st.number_input("Target rate p1", min_value=0.0, max_value=1.0, value=0.40, step=0.01, format="%.2f")
    b0 = st.number_input("Prior b₀ (efficacy)", min_value=0.0, value=1.0, step=0.5)
    c_futility = st.number_input("c_futility (PPoS futility cutoff)", min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.3f")

allow_early_success = st.sidebar.checkbox("Allow early success at interims", value=False)
theta_interim = st.sidebar.number_input(
    "θ_interim (posterior threshold at interims)",
    min_value=0.5, max_value=0.999, value=float(theta_final), step=0.01, format="%.3f",
    help="Used only if 'Allow early success' is checked."
)

# ---- Safety monitoring options ----
st.sidebar.subheader("Safety Monitoring (optional)")
enable_safety = st.sidebar.checkbox("Enable safety monitoring", value=True)
if enable_safety:
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        a_t0 = st.number_input("Safety prior a_t0", min_value=0.0, value=1.0, step=0.5)
        q_max = st.number_input("q_max (max acceptable toxicity)", min_value=0.0, max_value=1.0, value=0.15, step=0.01, format="%.2f")
    with col_s2:
        b_t0 = st.number_input("Safety prior b_t0", min_value=0.0, value=9.0, step=0.5)
        theta_tox = st.number_input("θ_tox (toxicity stopping threshold)", min_value=0.5, max_value=0.999, value=0.90, step=0.01, format="%.3f")
else:
    a_t0 = b_t0 = q_max = theta_tox = None

# ---- Wider interim look options ----
st.sidebar.subheader("Interim Look Schedule")

looks_mode = st.sidebar.selectbox(
    "Choose interim look mode",
    options=[
        "None (final only)",
        "Equal-spaced (k looks)",
        "Custom percentages",
        "Custom absolute Ns"
    ],
    index=1
)

k_looks = None
perc_str = None
ns_str = None
if looks_mode == "Equal-spaced (k looks)":
    k_looks = st.sidebar.slider("Number of interims (k)", min_value=1, max_value=8, value=2, step=1,
                                help="Interims at ~i/(k+1) of N, e.g., k=2 → ~33% and ~67%.")
elif looks_mode == "Custom percentages":
    perc_str = st.sidebar.text_input("Interim percentages (comma-separated)", value="33,67",
                                     help="Example: 25,50,75")
elif looks_mode == "Custom absolute Ns":
    ns_str = st.sidebar.text_input("Interim Ns (comma-separated)", value="",
                                   help="Example: 20,40 (must be < N)")

# Screening controls
st.sidebar.subheader("Screening Grid")
N_min, N_max = st.sidebar.slider("Total N range", min_value=10, max_value=400, value=(30, 120), step=1)
N_step = st.sidebar.number_input("N step", min_value=1, max_value=50, value=5, step=1)

n_sims_small = st.sidebar.number_input("Screening sims per design", min_value=100, max_value=200000, value=5000, step=500)
alpha_max = st.sidebar.number_input("Max Type I error (α) allowed", min_value=0.0, max_value=0.5, value=0.10, step=0.01, format="%.2f")
power_min = st.sidebar.number_input("Min Power at p1", min_value=0.0, max_value=1.0, value=0.80, step=0.01, format="%.2f")
seed = st.sidebar.number_input("Random seed", min_value=1, value=2026, step=1)


# Build candidate grid
Ns = list(range(N_min, N_max + 1, N_step))

def parse_percent_list(s):
    if not s:
        return []
    vals = []
    for tok in s.split(","):
        tok = tok.strip().replace("%", "")
        if tok == "":
            continue
        try:
            v = float(tok) / 100.0
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    return vals

def parse_n_list(s):
    if not s:
        return []
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        try:
            v = int(round(float(tok)))
            vals.append(v)
        except Exception:
            pass
    return vals

def look_schedule(N, mode, k_looks=None, perc_str=None, ns_str=None):
    if mode == "None (final only)":
        looks = []
    elif mode == "Equal-spaced (k looks)":
        k = int(k_looks or 1)
        looks = [int(np.floor(i * N / (k + 1))) for i in range(1, k + 1)]
    elif mode == "Custom percentages":
        fracs = parse_percent_list(perc_str or "")
        looks = [int(np.floor(f * N)) for f in fracs]
    elif mode == "Custom absolute Ns":
        ns = parse_n_list(ns_str or "")
        looks = ns
    else:
        looks = []
    looks = [int(min(max(1, l), N - 1)) for l in looks if 0 < l < N]
    looks = sorted(list(dict.fromkeys(looks)))
    return looks

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

st.write("### 1) Rapid Screener")
st.caption("We evaluate many candidate Ns and look schedules quickly, keep those meeting α and power constraints, and rank by ESS under p0.")

# Cache uniforms and screening results to keep app responsive
@st.cache_data(show_spinner=False)
def _screen(param_grid, n_sims_small, seed):
    rng = np.random.default_rng(seed)
    Nmax = max([g["N_total"] for g in param_grid])
    U = rng.uniform(size=(n_sims_small, Nmax))
    df, designs = shortlist_designs(param_grid, n_sims_small, seed, U)
    return df, designs

df_screen, designs_built = _screen(param_grid, n_sims_small, seed)

if df_screen.empty:
    st.warning("No viable designs encountered in the screening set (or final criterion impossible). Consider adjusting thresholds or N range.")
else:
    # Apply constraints and sort
    df_ok = df_screen[
        (df_screen["Type I error @ p0"] <= alpha_max) &
        (df_screen["Power @ p1"] >= power_min)
    ].copy()

    if df_ok.empty:
        st.info("No candidates met the α and power constraints. Displaying the full screening table for reference.")
        st.dataframe(df_screen.sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True))
        df_to_select_from = df_screen.sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True)
    else:
        df_ranked = df_ok.sort_values(["ESS @ p0", "N_total"]).reset_index(drop=True)
        st.dataframe(df_ranked.head(15))
        st.success(f"Found {len(df_ok)} candidate designs meeting constraints. Showing top 15 by ESS @ p0.")
        df_to_select_from = df_ranked

    # Selection for deep dive
    st.write("### 2) Deep Dive on Selected Design")
    st.caption("Pick a row index from the (filtered) table above for a high-precision joint simulation (efficacy + safety).")
    idx = st.number_input("Row index (0-based from the filtered table above)", min_value=0, value=0, step=1)

    # Choose a design row
    if len(df_to_select_from) > 0:
        idx_used = int(np.clip(idx, 0, len(df_to_select_from) - 1))
        chosen = df_to_select_from.iloc[idx_used]
    else:
        chosen = None

    if chosen is not None:
        st.write("**Chosen design**")
        show_cols = ["N_total", "looks", "theta_final", "theta_interim", "c_futility",
                     "allow_early_success", "Type I error @ p0", "Power @ p1",
                     "ESS @ p0", "ESS @ p1", "s_min_final"]
        st.json({k: (int(chosen[k]) if isinstance(chosen[k], (np.integer,)) else chosen.get(k, None)) for k in show_cols})
        st.caption("Interim continue thresholds (x ≥ x_min to continue):")
        st.write(chosen["x_min_to_continue"])

        # Deep dive controls
        st.write("#### Deep-dive Simulation Settings")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            n_sims_deep = st.number_input("Deep-dive sims", min_value=2000, max_value=800000, value=150000, step=5000)
        with col_d2:
            seed_deep = st.number_input("Deep-dive seed", min_value=1, value=seed + 1, step=1)

        # Toxicity scenarios
        st.write("#### Toxicity scenarios for deep dive")
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            q_good = st.number_input("q_good (benign toxicity)", min_value=0.0, max_value=1.0, value=0.10, step=0.01)
        with col_q2:
            q_bad = st.number_input("q_bad (concerning toxicity)", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
        with col_q3:
            q_for_OC = st.number_input("q_for_OC (toxicity for OC/ESS curves)", min_value=0.0, max_value=1.0, value=0.10, step=0.01)

        # Design dict from chosen row; add safety config from UI
        design = dict(chosen["_design"])  # copy
        if enable_safety:
            design["safety"] = dict(a_t0=a_t0, b_t0=b_t0, q_max=q_max, theta_tox=theta_tox)
        # Use defensive access so older cached objects don't break
        p0_used = design.get("p0", p0)
        p1_used = design.get("p1", p1)

        # Deep-dive simulation
        if st.button("Run Deep-Dive Simulation (efficacy + safety)", type="primary"):
            rng = np.random.default_rng(seed_deep)
            Ueff = rng.uniform(size=(n_sims_deep, design["N_total"]))
            Utox = rng.uniform(size=(n_sims_deep, design["N_total"]))

            # Point metrics (with safety if enabled)
            res_p0_qgood = simulate_design_with_safety(design, p_eff=p0_used, p_tox=q_good, U_eff=Ueff, U_tox=Utox)
            res_p1_qgood = simulate_design_with_safety(design, p_eff=p1_used, p_tox=q_good, U_eff=Ueff, U_tox=Utox)
            res_p1_qbad  = simulate_design_with_safety(design, p_eff=p1_used, p_tox=q_bad,  U_eff=Ueff, U_tox=Utox)

            st.write("##### Point Estimates (joint, with safety if enabled)")
            cols = st.columns(5)
            cols[0].metric("Type I @ p0, q_good", f"{res_p0_qgood['reject_rate']:.3f}")
            cols[1].metric("Power @ p1, q_good", f"{res_p1_qgood['reject_rate']:.3f}")
            cols[2].metric("ESS @ p0, q_good", f"{res_p0_qgood['ess']:.1f}")
            cols[3].metric("ESS @ p1, q_good", f"{res_p1_qgood['ess']:.1f}")
            if enable_safety:
                cols[4].metric("Safety stop @ p1, q_bad", f"{res_p1_qbad['safety_stop_prob']:.3f}")

            st.write("##### Sample Size Distribution @ p0 (q_good)")
            st.dataframe(res_p0_qgood["stop_dist"])
            st.write("##### Sample Size Distribution @ p1 (q_good)")
            st.dataframe(res_p1_qgood["stop_dist"])

            # OC and ESS curves across p with fixed q_for_OC
            st.write("##### OC & ESS vs efficacy p (with toxicity fixed at q_for_OC)")
            p_grid_min = st.number_input("OC curve p-min", min_value=0.0, max_value=1.0, value=max(0.0, p0_used - 0.15), step=0.01)
            p_grid_max = st.number_input("OC curve p-max", min_value=0.0, max_value=1.0, value=min(1.0, p1_used + 0.20), step=0.01)
            n_grid = st.slider("Number of points on OC grid", min_value=5, max_value=40, value=15, step=1)

            ps = np.linspace(p_grid_min, p_grid_max, n_grid)
            oc = []
            ess_curve = []
            # reuse uniforms for speed/variance reduction
            for pp in ps:
                r = simulate_design_with_safety(design, p_eff=pp, p_tox=q_for_OC, U_eff=Ueff, U_tox=Utox)
                oc.append(r["reject_rate"])
                ess_curve.append(r["ess"])

            df_oc = pd.DataFrame({"p": ps, "Reject_Prob": oc, "ESS": ess_curve})
            plot_lines(df_oc, x="p", y="Reject_Prob", title=f"Operating Characteristic: P(Declare Success) vs p (q={q_for_OC:.2f})")
            plot_lines(df_oc, x="p", y="ESS", title=f"Expected Sample Size vs p (q={q_for_OC:.2f})")

            st.write("##### Exportable Design Summary")
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
                    "Safety: stop if P(q>q_max|data) ≥ θ_tox at interims or final; "
                    "Efficacy: early success if P(p>p0|data) ≥ θ_interim; "
                    "Futility: stop if PPoS(final success) < c_futility; "
                    "Final success if P(p>p0|final data) ≥ θ_final."
                )
            )
            st.code(repr(export), language="python")

st.write("---")
st.write("### Methodological Details (what the app computes)")
st.markdown(
    r"""
- **Posterior**: With prior \(p \sim \text{Beta}(a_0, b_0)\), and data \(X \sim \text{Binomial}(n, p)\),
  \(p \mid X=x \sim \text{Beta}(a_0 + x,\, b_0 + n - x)\).

- **Final success** requires \(\Pr(p > p_0 \mid a_0+x, b_0+n-x) \ge \theta_{\text{final}}\).
  For fixed \(N\), this gives a **minimum total successes** \(s_{\min}\) at the final look, found via binary search.

- **Predictive probability of success (PPoS)** at interim \(n\) with \(x\) successes:
  Let \(m = N - n\) remain. With posterior \(p \mid x \sim \text{Beta}(a_0+x, b_0+n-x)\),
  the predictive distribution for future successes \(Y\) is **Beta–Binomial**.
  If \(s_{\min}\) is the final boundary, we need \(Y \ge s_{\min}-x\).
  Thus \(\mathrm{PPoS} = \Pr(Y \ge s_{\min}-x)\), compared to \(c_{\text{futility}}\).

- **Safety**: With prior \(q \sim \text{Beta}(a_{t0}, b_{t0})\), at interims/final we stop
  if \(\Pr(q > q_{\max} \mid \text{data}) \ge \theta_{\text{tox}}\).
"""
)

st.caption("Tip: If you want the safety rule to be more/less sensitive, adjust θ_tox and/or q_max; you can also strengthen or weaken the safety prior.")


