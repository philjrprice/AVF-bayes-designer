import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Stress-Tester")
st.markdown("This version is optimized for speed using vectorized matrix simulations.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5)
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7)
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30)

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.005, 0.20, 0.01, step=0.005)
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.90)
min_safety_power = st.sidebar.slider("Min Safety Power (Detection)", 0.70, 0.99, 0.95)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Settings")
cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5)
n_range = st.sidebar.slider("N Search Range", 40, 150, (60, 100))

# --- OPTIMIZED VECTORIZED SIMULATION ENGINE ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, conf, limit, cohort_sz, safe_conf=0.90):
    # Generate all patient data at once (Sims x Max_N)
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    
    # Trackers
    stops_n = np.full(sims, max_n)
    is_success = np.zeros(sims, dtype=bool)
    is_safety_stop = np.zeros(sims, dtype=bool)
    already_stopped = np.zeros(sims, dtype=bool)

    # Interim checks
    for n in range(cohort_sz, max_n + 1, cohort_sz):
        if np.all(already_stopped): break
        
        # Calculate sums for trials not yet stopped
        active = ~already_stopped
        curr_succ = np.sum(outcomes[active, :n], axis=1)
        curr_saes = np.sum(saes[active, :n], axis=1)
        
        # Bayesian math (vectorized)
        prob_eff = 1 - beta.cdf(hurdle, 1 + curr_succ, 1 + (n - curr_succ))
        prob_tox = 1 - beta.cdf(limit, 1 + curr_saes, 1 + (n - curr_saes))
        
        # Determine Stop Conditions
        tox_stop = prob_tox > safe_conf
        eff_stop = prob_eff > conf
        
        # Logic: Safety stops take priority over efficacy
        newly_stopped_safety = active.copy()
        newly_stopped_safety[active] = tox_stop
        
        newly_stopped_eff = active.copy()
        newly_stopped_eff[active] = (eff_stop & ~tox_stop)
        
        # Update trackers
        is_safety_stop[newly_stopped_safety] = True
        is_success[newly_stopped_eff] = True
        stops_n[active & (tox_stop | eff_stop)] = n
        already_stopped[newly_stopped_safety | newly_stopped_eff] = True

    # Final result for trials that reached the end
    if np.any(~already_stopped):
        active = ~already_stopped
        final_succ = np.sum(outcomes[active, :max_n], axis=1)
        prob_eff_final = 1 - beta.cdf(hurdle, 1 + final_succ, 1 + (max_n - final_succ))
        is_success[active] = prob_eff_final > conf

    return np.mean(is_success), np.mean(is_safety_stop), np.mean(stops_n)

# --- EXECUTION ---
if st.button("🚀 Run Optimized Adaptive Designer"):
    results = []
    prog_bar = st.progress(0)
    n_list = list(range(n_range[0], n_range[1] + 1, 2))
    
    with st.spinner("Searching search space..."):
        for i, n in enumerate(n_list):
            prog_bar.progress(i / len(n_list))
            for hurdle in [0.55, 0.60, 0.65]:
                for conf in [0.74, 0.80, 0.85]:
                    # Quick Alpha Check (Fixed N look for speed)
                    alpha, _, _ = run_fast_batch(300, n, p0, 0.05, hurdle, conf, safe_limit, n)
                    
                    if alpha <= max_alpha:
                        # Full Adaptive Check for Power and Safety
                        power, _, _ = run_fast_batch(300, n, p1, 0.05, hurdle, conf, safe_limit, n)
                        _, tox_stop, _ = run_fast_batch(300, n, p1, true_toxic_rate, hurdle, conf, safe_limit, n)
                        
                        if power >= min_power and tox_stop >= min_safety_power:
                            results.append({"N": n, "Hurdle": hurdle, "Conf": conf, "Alpha": alpha, "Power": power, "Safety": tox_stop})

    if results:
        df = pd.DataFrame(results)
        best = df.sort_values("N").iloc[0]
        
        st.success(f"### ✅ Optimal Adaptive Design Found: Max N = {int(best['N'])}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max Enrollment", int(best['N']))
        c2.metric("Efficacy Power", f"{best['Power']:.1%}")
        c3.metric("Safety Stop Prob.", f"{best['Safety']:.1%}")
        c4.metric("Risk (Alpha)", f"{best['Alpha']:.2%}")

        st.markdown("---")
        st.subheader("📋 Adaptive Operational Stress Test (OC Table)")
        
        scenarios = [
            ("1. High Eff / Safe", p1 + 0.1, 0.05),
            ("2. On-Target / Safe", p1, 0.05),
            ("10. Futile (Null)", p0, 0.05),
            ("11. High Eff / Toxic", p1 + 0.1, true_toxic_rate),
            ("12. Target Eff / Toxic", p1, true_toxic_rate),
        ]
        
        stress_data = []
        for name, pe, ps in scenarios:
            pow_val, stop_val, asn_val = run_fast_batch(1000, int(best['N']), pe, ps, best['Hurdle'], best['Conf'], safe_limit, cohort_size)
            stress_data.append({
                "Scenario Name": name,
                "Success %": f"{pow_val*100:.1f}%",
                "Safety Stop %": f"{stop_val*100:.1f}%",
                "Avg N (ASN)": f"{asn_val:.1f}",
                "True Eff": f"{pe*100:.0f}%",
                "True SAE": f"{ps*100:.0f}%"
            })
        st.table(pd.DataFrame(stress_data))
    else:
        st.error("No design found. Try widening the N Search Range or relaxing Risk Standards.")
