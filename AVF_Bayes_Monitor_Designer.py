import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Specialized Priors")
st.markdown("Updated v16.1: Fixed Search Engine & Robust Lead-in Logic.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5)
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7)
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Efficacy Prior Strength")
prior_alpha = st.sidebar.slider("Eff Prior α", 1.0, 10.0, 1.0, step=0.5)
prior_beta = st.sidebar.slider("Eff Prior β", 1.0, 10.0, 1.0, step=0.5)

st.sidebar.header("🛡️ Safety Prior Strength")
s_prior_alpha = st.sidebar.slider("Saf Prior α", 1.0, 10.0, 1.0, step=0.5)
s_prior_beta = st.sidebar.slider("Saf Prior β", 1.0, 10.0, 1.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max Alpha", 0.005, 0.20, 0.01, step=0.005)
min_power = st.sidebar.slider("Min Power", 0.70, 0.99, 0.90)
min_safety_power = st.sidebar.slider("Min Safety Power", 0.70, 0.99, 0.95)

st.sidebar.markdown("---")
st.sidebar.header("🔬 Simulation Rigor")
n_sims = st.sidebar.select_slider("Simulations", options=[2000, 5000, 10000], value=2000)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Thresholds")
min_n_lead = st.sidebar.slider("Min N Before First Check", 5, 50, 20)
eff_conf = st.sidebar.slider("Efficacy Success Confidence", 0.70, 0.99, 0.85)
safety_conf = st.sidebar.slider("Safety Stop Confidence", 0.50, 0.99, 0.90)
fut_conf = st.sidebar.slider("Futility Stop Threshold", 0.01, 0.20, 0.05)
cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5)
n_range = st.sidebar.slider("N Search Range", 40, 150, (60, 100))

# --- STABLE ENGINE FROM V15 ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, e_conf, limit, cohort_sz, s_conf, f_conf, p_a, p_b, s_a, s_b, min_n):
    np.random.seed(42)
    p_eff, p_sae = np.clip(p_eff, 0.001, 0.999), np.clip(p_sae, 0.001, 0.999)
    
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    stops_n = np.full(sims, max_n)
    is_success, is_safety_stop, is_futility_stop, already_stopped = [np.zeros(sims, dtype=bool) for _ in range(4)]

    # Analysis sequence using Lead-in and Cohort logic
    look_points = sorted(list(set([min_n] + [n for n in range(min_n, max_n + 1, cohort_sz) if n <= max_n])))

    for n in look_points:
        active = ~already_stopped
        if not np.any(active): break
        c_s, c_tox = np.sum(outcomes[active, :n], axis=1), np.sum(saes[active, :n], axis=1)
        
        prob_eff = 1 - beta.cdf(hurdle, p_a + c_s, p_b + (n - c_s))
        prob_tox = 1 - beta.cdf(limit, s_a + c_tox, s_b + (n - c_tox))
        
        tox_trig, eff_trig = prob_tox > s_conf, prob_eff > e_conf
        fut_trig = (n >= max_n/2) & (prob_eff < f_conf)
        
        new_stops = active.copy(); new_stops[active] = (tox_trig | eff_trig | fut_trig)
        is_safety_stop[active & newly_mapped(active, tox_trig)] = True
        is_success[active & newly_mapped(active, eff_trig & ~tox_trig)] = True
        is_futility_stop[active & newly_mapped(active, fut_trig & ~tox_trig & ~eff_trig)] = True
        stops_n[new_stops & ~already_stopped] = n
        already_stopped[new_stops] = True

    remaining = ~already_stopped
    if np.any(remaining):
        f_s = np.sum(outcomes[remaining, :max_n], axis=1)
        is_success[remaining] = (1 - beta.cdf(hurdle, p_a + f_s, p_b + (max_n - f_s))) > e_conf
    
    return np.mean(is_success), np.mean(is_safety_stop), np.mean(stops_n), np.mean(is_futility_stop)

def newly_mapped(active, trig):
    m = np.zeros(len(active), dtype=bool); m[active] = trig; return m

# --- SEARCH ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    adj_start = max(n_range[0], min_n_lead)
    n_list = range(adj_start, n_range[1] + 1, 2)
    hurdle_options = np.linspace(p0, (p0 + p1)/2, 5)
    
    with st.spinner("Analyzing designs..."):
        for n in n_list:
            for h in hurdle_options:
                h = round(float(h), 3)
                # Corrected: Search now respects cohort and lead-in monitoring
                a, _, _, _ = run_fast_batch(n_sims, n, p0, 0.05, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta, min_n_lead)
                if a <= max_alpha:
                    p, _, _, _ = run_fast_batch(n_sims, n, p1, 0.05, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta, min_n_lead)
                    _, sp, _, _ = run_fast_batch(n_sims, n, p1, true_toxic_rate, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta, min_n_lead)
                    if p >= min_power and sp >= min_safety_power:
                        results.append({"N": n, "Hurdle": h, "Alpha": a, "Power": p, "Safety": sp})
                        break
    
    if results:
        st.session_state['best_design'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['used_params'] = {
            "p0": p0, "p1": p1, "safe_limit": safe_limit, "toxic_rate": true_toxic_rate,
            "eff_conf": eff_conf, "saf_conf": safety_conf, "fut_conf": fut_conf,
            "p_a": prior_alpha, "p_b": prior_beta, "s_a": s_prior_alpha, "s_b": s_prior_beta,
            "cohort": cohort_size, "min_n": min_n_lead, "sim_rigor": n_sims
        }
    else:
        st.error("❌ No design found. Check if Min N Lead-in is set too high for the Search Range.")

# --- RESULTS DISPLAY ---
if 'best_design' in st.session_state:
    best, up = st.session_state['best_design'], st.session_state['used_params']
    st.success(f"### ✅ Optimal Design Found: Max N={int(best['N'])} | Hurdle={best['Hurdle']}")
    
    if st.button("📈 Run Multi-Scenario Stress Test"):
        scenarios = [
            ("1. Super-Effective (+10%)", up['p1']+0.1, 0.05),
            ("2. On-Target", up['p1'], 0.05),
            ("3. Null (Alpha)", up['p0'], 0.05),
            ("4. Toxic Scenario", up['p1'], up['toxic_rate'])
        ]
        stress_data = []
        for name, pe, ps in scenarios:
            pw, stp, asn, fut = run_fast_batch(up['sim_rigor'], int(best['N']), pe, ps, best['Hurdle'], up['eff_conf'], up['safe_limit'], up['cohort'], up['saf_conf'], up['fut_conf'], up['p_a'], up['p_b'], up['s_a'], up['s_b'], up['min_n'])
            stress_data.append({"Scenario": name, "Success %": f"{pw:.1%}", "Safety Stop %": f"{stp:.1%}", "Avg N": f"{asn:.1f}"})
        st.table(pd.DataFrame(stress_data))
