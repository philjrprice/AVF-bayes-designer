import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd

st.set_page_config(page_title="AVF Master Designer: Optimized", layout="wide")

st.title("🧬 Master Designer: Optimized Adaptive Suite")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Efficacy & Safety")
    p0, p1 = st.slider("Null (p0)", 0.3, 0.7, 0.5), st.slider("Target (p1)", 0.5, 0.9, 0.7)
    safe_limit = st.slider("SAE Limit", 0.05, 0.30, 0.15)
    true_tox = st.slider("Toxic Rate", 0.10, 0.50, 0.30)
    
    st.header("📐 Risk Standards")
    max_alpha = st.slider("Max Alpha", 0.005, 0.10, 0.01, step=0.005)
    min_power = st.slider("Min Power", 0.70, 0.99, 0.90)
    min_safe_p = st.slider("Min Safety Power", 0.70, 0.99, 0.95)
    
    st.header("⏱️ Adaptive")
    cohort_sz = st.slider("Cohort Size", 1, 20, 5)
    n_range = st.slider("N Range", 40, 120, (60, 90))

# --- FAST SIMULATION ENGINE ---
def run_fast_batch(sims, n, p_eff, p_sae, hurdle, conf, limit, cohort):
    # Vectorized generation of outcomes for speed
    outcomes = np.random.binomial(1, p_eff, (sims, n))
    saes = np.random.binomial(1, p_sae, (sims, n))
    
    stops_n = np.full(sims, n)
    success = np.zeros(sims, dtype=bool)
    safety_stop = np.zeros(sims, dtype=bool)
    finished = np.zeros(sims, dtype=bool)

    # Only check at interim points
    for look in range(cohort, n + 1, cohort):
        if np.all(finished): break
        
        idx = ~finished
        curr_n = look
        s_count = np.sum(outcomes[idx, :curr_n], axis=1)
        e_count = np.sum(saes[idx, :curr_n], axis=1)
        
        # Bayesian math
        p_eff_val = 1 - beta.cdf(hurdle, 1 + s_count, 1 + (curr_n - s_count))
        p_tox_val = 1 - beta.cdf(limit, 1 + e_count, 1 + (curr_n - e_count))
        
        # Determine stops
        is_tox = p_tox_val > 0.90
        is_suc = p_eff_val > conf
        
        # Update trackers
        # Safety stops take priority
        new_tox_stops = idx.copy()
        new_tox_stops[idx] = is_tox
        
        new_suc_stops = idx.copy()
        new_suc_stops[idx] = (is_suc & ~is_tox)
        
        # Apply updates
        safety_stop[new_tox_stops] = True
        success[new_suc_stops] = True
        stops_n[idx & (new_tox_stops | new_suc_stops)] = curr_n
        finished[new_tox_stops | new_suc_stops] = True

    return np.mean(success), np.mean(safety_stop), np.mean(stops_n)

# --- EXECUTION ---
if st.button("🚀 Run Optimized Designer"):
    results = []
    prog = st.progress(0)
    
    # Pre-calculate search space
    n_list = list(range(n_range[0], n_range[1] + 1, 2))
    
    for i, n in enumerate(n_list):
        prog.progress(i / len(n_list))
        for hurdle in [0.55, 0.60, 0.65]:
            for conf in [0.74, 0.80, 0.85]:
                # 1. Quick Final-N Check (No interims) to see if design is even possible
                final_success = 1 - beta.cdf(hurdle, 1 + np.random.binomial(n, p0, 200), 1 + (n - np.random.binomial(n, p0, 200)))
                if np.mean(final_success > conf) > max_alpha: continue
                
                # 2. Run actual Adaptive Sim for promising designs
                alpha, _, _ = run_fast_batch(200, n, p0, 0.05, hurdle, conf, safe_limit, n)
                if alpha <= max_alpha:
                    pwr, _, _ = run_fast_batch(200, n, p1, 0.05, hurdle, conf, safe_limit, n)
                    _, tox, _ = run_fast_batch(200, n, p1, true_tox, hurdle, conf, safe_limit, n)
                    
                    if pwr >= min_power and tox >= min_safety_power:
                        results.append({"N": n, "Hurdle": hurdle, "Conf": conf, "Alpha": alpha, "Power": pwr, "Safety": tox})

    if results:
        best = pd.DataFrame(results).sort_values("N").iloc[0]
        st.success(f"### Found Optimal N: {int(best['N'])}")
        
        # Stress Test with High Precision
        st.subheader("📋 Adaptive Stress Test (ASN Table)")
        scenarios = [("On-Target / Safe", p1, 0.05), ("Null / Safe", p0, 0.05), ("Target / Toxic", p1, true_tox)]
        
        data = []
        for name, pe, ps in scenarios:
            s, t, a = run_fast_batch(1000, int(best['N']), pe, ps, best['Hurdle'], best['Conf'], safe_limit, cohort_sz)
            data.append({"Scenario": name, "Success %": f"{s*100:.1f}%", "Safety Stop %": f"{t*100:.1f}%", "Avg N": f"{a:.1f}"})
        st.table(pd.DataFrame(data))
    else:
        st.error("No design met requirements. Try widening N range.")
