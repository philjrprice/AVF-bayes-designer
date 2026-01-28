import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Stress-Tester")
st.markdown("Restored: Independent Confidence Sliders, OC Logic, and Guidance Tooltips.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, 
    help="The success rate of the current standard of care. If the drug performs at or below this, it is a failure.")

p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, 
    help="The 'Goal' success rate you hope the drug achieves.")

safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15, 
    help="Maximum allowable Serious Adverse Event rate. Exceeding this triggers a stop.")

true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30, 
    help="Hypothetical rate used to test if the monitor is strong enough to catch a dangerous drug.")

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.005, 0.20, 0.01, step=0.005, 
    help="Risk of a 'False Win.' 0.01 means a 1% chance of calling a failing drug a success.")

min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.90, 
    help="The probability of correctly identifying a successful drug.")

min_safety_power = st.sidebar.slider("Min Safety Power (Detection)", 0.70, 0.99, 0.95, 
    help="The probability the trial will stop if the drug is actually toxic.")

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Thresholds")
eff_conf = st.sidebar.slider("Efficacy Success Confidence", 0.70, 0.99, 0.85, 
    help="The probability threshold required to declare success. Higher = harder to stop early.")

safety_conf = st.sidebar.slider("Safety Stop Confidence", 0.50, 0.99, 0.90, 
    help="Confidence required to trigger a safety stop. Lower = more sensitive safety monitor.")

cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5, 
    help="How often the monitor checks data. Smaller = faster stops; Larger = easier logistics.")

n_range = st.sidebar.slider("N Search Range", 40, 150, (60, 100))

# --- STABLE VECTORIZED ENGINE ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, e_conf, limit, cohort_sz, s_conf):
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    stops_n = np.full(sims, max_n)
    is_success, is_safety_stop, is_futility_stop, already_stopped = [np.zeros(sims, dtype=bool) for _ in range(4)]

    for n in range(cohort_sz, max_n + 1, cohort_sz):
        active = ~already_stopped
        if not np.any(active): break
        c_s, c_tox = np.sum(outcomes[active, :n], axis=1), np.sum(saes[active, :n], axis=1)
        prob_eff = 1 - beta.cdf(hurdle, 1 + c_s, 1 + (n - c_s))
        prob_tox = 1 - beta.cdf(limit, 1 + c_tox, 1 + (n - c_tox))
        tox_trig, eff_trig = prob_tox > s_conf, prob_eff > e_conf
        fut_trig = (n >= max_n/2) & (prob_eff < 0.05)
        
        new_stops = active.copy(); new_stops[active] = (tox_trig | eff_trig | fut_trig)
        is_safety_stop[active & newly_mapped(active, tox_trig)] = True
        is_success[active & newly_mapped(active, eff_trig & ~tox_trig)] = True
        is_futility_stop[active & newly_mapped(active, fut_trig & ~tox_trig & ~eff_trig)] = True
        stops_n[new_stops & ~already_stopped] = n
        already_stopped[new_stops] = True

    remaining = ~already_stopped
    if np.any(remaining):
        f_s = np.sum(outcomes[remaining, :max_n], axis=1)
        is_success[remaining] = (1 - beta.cdf(hurdle, 1 + f_s, 1 + (max_n - f_s))) > e_conf
    return np.mean(is_success), np.mean(is_safety_stop), np.mean(stops_n), np.mean(is_futility_stop)

def newly_mapped(active, trig):
    m = np.zeros(len(active), dtype=bool); m[active] = trig; return m

# --- PHASE 1: SEARCH ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    n_list = list(range(n_range[0], n_range[1] + 1, 2))
    with st.spinner("Searching for optimal design..."):
        for n in n_list:
            for hurdle in [0.55, 0.60, 0.65]:
                alpha, _, _, _ = run_fast_batch(2000, n, p0, 0.05, hurdle, eff_conf, safe_limit, n, safety_conf)
                if alpha <= max_alpha:
                    pwr, _, _, _ = run_fast_batch(2000, n, p1, 0.05, hurdle, eff_conf, safe_limit, n, safety_conf)
                    _, tox_p, _, _ = run_fast_batch(2000, n, p1, true_toxic_rate, hurdle, eff_conf, safe_limit, n, safety_conf)
                    if pwr >= min_power and tox_p >= min_safety_power:
                        results.append({"N": n, "Hurdle": hurdle, "Alpha": alpha, "Power": pwr, "Safety": tox_p})
    if results:
        st.session_state['best_design'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['best_design_eff_conf'] = eff_conf
    else:
        st.error("No design found. Try relaxing Risk Standards.")

# --- PERSISTENT DISPLAY & PROTOCOL WINDOW ---
if 'best_design' in st.session_state:
    best = st.session_state['best_design']
    used_eff_conf = st.session
