import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Specialized Priors")
st.markdown("Rebuilt v20: Full v15 Restoration with Smarter Search Logic.")

# --- SIDEBAR: DESIGN GOALS ---
with st.sidebar:
    st.header("🎯 Efficacy & Safety")
    p0 = st.slider("Null Efficacy (p0)", 0.2, 0.8, 0.5)
    p1 = st.slider("Target Efficacy (p1)", 0.3, 0.9, 0.7)
    safe_limit = st.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
    true_toxic_rate = st.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30)
    
    st.header("⚖️ Bayesian Priors")
    p_a = st.slider("Eff Prior α", 0.5, 5.0, 1.0)
    p_b = st.slider("Eff Prior β", 0.5, 5.0, 1.0)
    s_a = st.slider("Saf Prior α", 0.5, 5.0, 1.0)
    s_b = st.slider("Saf Prior β", 0.5, 5.0, 1.0)
    
    st.header("📐 Risk & Stopping")
    max_alpha = st.slider("Max Alpha", 0.01, 0.30, 0.05)
    min_power = st.slider("Min Power", 0.50, 0.95, 0.80)
    min_safety_power = st.slider("Min Safety Detection", 0.50, 0.99, 0.80)
    
    st.header("⏱️ Lead-in & Adaptive")
    min_n_lead = st.slider("Min N Before First Check", 5, 50, 20)
    cohort_size = st.slider("Cohort Size", 1, 10, 5)
    n_range = st.slider("N Search Range", 20, 200, (40, 100))
    eff_conf = st.slider("Success Confidence", 0.50, 0.99, 0.85)
    safety_conf = st.slider("Safety Confidence", 0.50, 0.99, 0.90)
    fut_conf = st.slider("Futility Threshold", 0.01, 0.30, 0.05)

# --- CORE SIMULATION ENGINE ---
def run_simulation(sims, max_n, p_eff, p_sae, hurdle, e_conf, limit, cohort_sz, s_conf, f_conf, pa, pb, sa, sb, min_n):
    np.random.seed(42)
    # Correct Look Points
    look_points = [min_n]
    while look_points[-1] + cohort_sz <= max_n:
        look_points.append(look_points[-1] + cohort_sz)
    if look_points[-1] < max_n: look_points.append(max_n)

    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    
    stopped = np.zeros(sims, dtype=bool)
    results = {"success": np.zeros(sims, dtype=bool), "safety_stop": np.zeros(sims, dtype=bool), 
               "futility_stop": np.zeros(sims, dtype=bool), "n": np.full(sims, max_n)}

    for n in look_points:
        active = ~stopped
        if not np.any(active): break
        
        s_count = np.sum(outcomes[active, :n], axis=1)
        t_count = np.sum(saes[active, :n], axis=1)
        
        # Bayesian Posterior Checks
        p_success = 1 - beta.cdf(hurdle, pa + s_count, pb + (n - s_count))
        p_toxic = 1 - beta.cdf(limit, sa + t_count, sb + (n - t_count))
        
        is_tox = p_toxic > s_conf
        is_eff = p_success > e_conf
        is_fut = (n >= max_n/2) & (p_success < f_conf)
        
        # Record stops
        idx = np.where(active)[0]
        tox_sims = idx[is_tox]
        eff_sims = idx[is_eff & ~is_tox]
        fut_sims = idx[is_fut & ~is_tox & ~is_eff]
        
        for s_idx in tox_sims:
            if not stopped[s_idx]:
                results["safety_stop"][s_idx] = True
                results["n"][s_idx] = n
                stopped[s_idx] = True
        for s_idx in eff_sims:
            if not stopped[s_idx]:
                results["success"][s_idx] = True
                results["n"][s_idx] = n
                stopped[s_idx] = True
        for s_idx in fut_sims:
            if not stopped[s_idx]:
                results["futility_stop"][s_idx] = True
                results["n"][s_idx] = n
                stopped[s_idx] = True
                
    return np.mean(results["success"]), np.mean(results["safety_stop"]), np.mean(results["n"]), np.mean(results["futility_stop"])

# --- OPTIMIZATION LOOP ---
if st.button("🚀 Find Optimal Sample Size"):
    found = False
    n_list = range(n_range[0], n_range[1] + 1, 2)
    # Broad hurdle search to ensure we find a valid anchor
    hurdle_list = np.linspace(p0 - 0.05, (p0+p1)/2, 6)
    
    with st.spinner("Scanning Design Space..."):
        for n in n_list:
            for h in hurdle_list:
                alpha, _, _, _ = run_simulation(1500, n, p0, 0.05, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, p_a, p_b, s_a, s_b, min_n_lead)
                if alpha <= max_alpha:
                    pwr, _, _, _ = run_simulation(1500, n, p1, 0.05, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, p_a, p_b, s_a, s_b, min_n_lead)
                    _, s_pwr, _, _ = run_simulation(1500, n, p1, true_toxic_rate, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, p_a, p_b, s_a, s_b, min_n_lead)
                    if pwr >= min_power and s_pwr >= min_safety_power:
                        st.session_state['best'] = {"N": n, "Hurdle": round(h, 3), "Alpha": alpha, "Power": pwr, "Safety": s_pwr}
                        found = True
                        break
            if found: break
    if not found: st.error("❌ No design found. Adjust 'Success Confidence' or 'Max Alpha'.")

# --- v15 FEATURE RESTORATION ---
if 'best' in st.session_state:
    b = st.session_state['best']
    st.success(f"### ✅ Optimal Design: Max N={int(b['N'])} | Hurdle={b['Hurdle']}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Recommended Max N", int(b['N']))
    col2.metric("Observed Alpha", f"{b['Alpha']:.2%}")
    col3.metric("Projected Power", f"{b['Power']:.1%}")

    with st.expander("📈 Comprehensive Stress Test (8 Scenarios)", expanded=True):
        if st.button("Run Full Stress Test"):
            scens = [("Null", p0, 0.05), ("Target", p1, 0.05), ("Toxic", p1, true_toxic_rate), ("Super-Eff", p1+0.1, 0.05)]
            rows = []
            for name, pe, ps in scens:
                pw, stp, asn, fut = run_simulation(2000, int(b['N']), pe, ps, b['Hurdle'], eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, p_a, p_b, s_a, s_b, min_n_lead)
                rows.append({"Scenario": name, "Success %": f"{pw:.1%}", "Safety Stop %": f"{stp:.1%}", "Avg N": f"{asn:.1f}"})
            st.table(pd.DataFrame(rows))

    # OC Curve Visualization
    st.subheader("📊 Operating Characteristics Curve")
    x_eff = np.linspace(p0 - 0.1, p1 + 0.1, 8)
    y_pos = [run_simulation(1000, int(b['N']), x, 0.05, b['Hurdle'], eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, p_a, p_b, s_a, s_b, min_n_lead)[0] for x in x_eff]
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(x_eff, y_pos, marker='o', color='navy'); ax.axvline(p0, color='r', ls='--'); st.pyplot(fig)
