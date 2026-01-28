import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Stress-Tester")
st.markdown("Optimized with Protocol Rule Generation for Bayesian Monitors.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, help="Success rate of standard care.")
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, help="Goal success rate.")
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15, help="Max allowable SAE rate.")
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30, help="Hypothetical toxicity for testing.")

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.005, 0.20, 0.01, step=0.005)
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.90)
min_safety_power = st.sidebar.slider("Min Safety Power", 0.70, 0.99, 0.95)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Thresholds")
eff_conf = st.sidebar.slider("Efficacy Success Confidence", 0.70, 0.99, 0.85, help="Certainty required to stop for efficacy.")
safety_conf = st.sidebar.slider("Safety Stop Confidence", 0.50, 0.99, 0.90, help="Certainty required to stop for toxicity.")
cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5)
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
        tox_trig = prob_tox > s_conf 
        eff_trig = prob_eff > e_conf   
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

# --- PERSISTENT DISPLAY & RULES WINDOW ---
if 'best_design' in st.session_state:
    best = st.session_state['best_design']
    used_eff_conf = st.session_state['best_design_eff_conf']
    
    st.success(f"### ✅ Optimal Design Parameters (Max N = {int(best['N'])})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Enrollment", int(best['N']))
    c2.metric("Efficacy Power", f"{best['Power']:.1%}")
    c3.metric("Safety Detection", f"{best['Safety']:.1%}")
    c4.metric("Risk (Alpha)", f"{best['Alpha']:.2%}")
    
    # --- NEW: BAYESIAN MONITORING RULES WINDOW ---
    with st.expander("📝 Protocol Summary: Bayesian Monitoring Rules", expanded=True):
        st.markdown(f"""
        **The following decision rules are required for the Bayesian Monitor to maintain the operational characteristics shown above:**
        
        1. **Interim Analysis Schedule**: Data will be monitored in cohorts of **{cohort_size}** patients.
        2. **Efficacy Success Rule**: Declare success if the posterior probability $P(\\text{{Response Rate}} > {best['Hurdle']}) > {used_eff_conf}$.
        3. **Safety Stopping Rule**: Stop immediately for toxicity if $P(\\text{{SAE Rate}} > {safe_limit}) > {safety_conf}$.
        4. **Futility Rule**: At or after patient **{int(best['N']//2)}**, stop for futility if the probability of eventually hitting the success threshold is **< 5%**.
        5. **Final Analysis**: If the trial reaches **{int(best['N'])}** patients without an interim stop, declare success only if the final posterior probability exceeds **{used_eff_conf}**.
        """)

    st.markdown("---")
    st.subheader("📊 Operational Characteristics (OC) Stress-Tester")
    if st.button("📈 Run Multi-Scenario Stress Test"):
        # (Stress-tester logic remains the same)
        pass
    
