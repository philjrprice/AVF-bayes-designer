import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt

# --- STABLE VERSION 15 RESTORATION WITH COHORT FIX ---
st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Specialized Priors")
st.markdown("Updated v22: Fixed Cohort-Induced Search Instability.")

# --- SIDEBAR: DESIGN GOALS (RESTORED V15) ---
with st.sidebar:
    st.header("🎯 Efficacy & Safety")
    p0 = st.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5)
    p1 = st.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7)
    safe_limit = st.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
    true_toxic_rate = st.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30)
    
    st.header("⚖️ Priors")
    p_a = st.slider("Eff Prior α", 1.0, 10.0, 1.0)
    p_b = st.slider("Eff Prior β", 1.0, 10.0, 1.0)
    
    st.header("📐 Risk Standards")
    max_alpha = st.slider("Max Alpha", 0.01, 0.30, 0.05)
    min_power = st.slider("Min Power", 0.50, 0.99, 0.80)
    
    st.header("⏱️ Adaptive Thresholds")
    eff_conf = st.slider("Success Confidence", 0.70, 0.99, 0.85)
    cohort_size = st.slider("Interim Cohort Size", 5, 25, 10)
    n_range = st.slider("N Search Range", 40, 200, (60, 120))

# --- STABILIZED SEARCH ENGINE ---
def run_simulation(sims, max_n, p_eff, hurdle, e_conf, cohort_sz, pa, pb):
    np.random.seed(42)
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    is_success = np.zeros(sims, dtype=bool)
    stopped = np.zeros(sims, dtype=bool)
    
    # Define a 'Floor' for looks to prevent N=20 jumps
    look_points = [n for n in range(cohort_sz, max_n + 1, cohort_sz)]
    if not look_points or look_points[-1] < max_n:
        look_points.append(max_n)

    for n in look_points:
        active = ~stopped
        if not np.any(active): break
        
        c_s = np.sum(outcomes[active, :n], axis=1)
        # Bayesian Success Check
        prob_eff = 1 - beta.cdf(hurdle, pa + c_s, pb + (n - c_s))
        
        eff_trig = prob_eff > e_conf
        idx = np.where(active)[0]
        is_success[idx[eff_trig]] = True
        stopped[idx[eff_trig]] = True

    return np.mean(is_success)

# --- SEARCH ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    # Broaden Hurdle Search to prevent 'No Results' at low cohort sizes
    hurdle_opts = np.linspace(p0, (p0+p1)/2, 10)
    n_list = range(n_range[0], n_range[1] + 1, 5)
    
    with st.spinner("Stabilizing design search..."):
        for n in n_list:
            for h in hurdle_opts:
                # Use simulation-accurate Alpha check
                alpha = run_simulation(2000, n, p0, h, eff_conf, cohort_size, p_a, p_b)
                if alpha <= max_alpha:
                    pwr = run_simulation(2000, n, p1, h, eff_conf, cohort_size, p_a, p_b)
                    if pwr >= min_power:
                        results.append({"N": n, "Hurdle": round(h, 3), "Alpha": alpha, "Power": pwr})
                        break
    
    if results:
        st.session_state['best'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.success(f"### ✅ Stable Design Found: Max N={int(st.session_state['best']['N'])}")
        st.write(st.session_state['best'])
    else:
        st.error("❌ No design found. Try lowering 'Success Confidence' or increasing 'Max Alpha'.")
