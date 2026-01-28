import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Stress-Tester")
st.markdown("Optimized for speed and clinical legitimacy using vectorized Bayesian simulations.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.50)
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.70)
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30)

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.005, 0.20, 0.05)
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.80)
min_safety_power = st.sidebar.slider("Min Safety Power (Detection)", 0.70, 0.99, 0.90)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Settings")
min_n_lead_in = st.sidebar.number_input("Minimum N before first check", 1, 100, 15)
cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 1)
n_range = st.sidebar.slider("N Search Range", 20, 200, (40, 120))

# --- VECTORIZED ENGINE ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, conf, limit, cohort_sz, lead_in, safe_conf=0.90):
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    
    stops_n = np.full(sims, max_n)
    is_success = np.zeros(sims, dtype=bool)
    is_safety_stop = np.zeros(sims, dtype=bool)
    is_futility_stop = np.zeros(sims, dtype=bool)
    already_stopped = np.zeros(sims, dtype=bool)

    # Generate checkpoints respecting lead-in
    check_points = [n for n in range(lead_in, max_n + 1) if (n == lead_in) or ((n - lead_in) % cohort_sz == 0)]
    
    for n in check_points:
        active = ~already_stopped
        if not np.any(active): break
        
        curr_succ = np.sum(outcomes[active, :n], axis=1)
        curr_saes = np.sum(saes[active, :n], axis=1)
        
        prob_eff = 1 - beta.cdf(hurdle, 1 + curr_succ, 1 + (n - curr_succ))
        prob_tox = 1 - beta.cdf(limit, 1 + curr_saes, 1 + (n - curr_saes))
        
        tox_trigger = prob_tox > safe_conf
        eff_trigger = prob_eff > conf
        fut_trigger = (n >= max_n/2) & (prob_eff < 0.05)
        
        # Determine global indices for stopping
        batch_stopped = (tox_trigger | eff_trigger | fut_trigger)
        new_stops = np.zeros(sims, dtype=bool)
        new_stops[active] = batch_stopped
        
        # Track reasons for stopping
        m_tox = np.zeros(sims, dtype=bool); m_tox[active] = tox_trigger
        m_eff = np.zeros(sims, dtype=bool); m_eff[active] = (eff_trigger & ~tox_trigger)
        m_fut = np.zeros(sims, dtype=bool); m_fut[active] = (fut_trigger & ~tox_trigger & ~eff_trigger)
        
        is_safety_stop[active & m_tox] = True
        is_success[active & m_eff] = True
        is_futility_stop[active & m_fut] = True
        
        stops_n[new_stops & ~already_stopped] = n
        already_stopped[new_stops] = True

    # Final check for those who reached max_n without stopping
    remaining = ~already_stopped
    if np.any(remaining):
        final_succ = np.sum(outcomes[remaining, :max_n], axis=1)
        is_success[remaining] = (1 - beta.cdf(hurdle, 1 + final_succ, 1 + (max_n - final_succ))) > conf

    return np.mean(is_success), np.mean(is_safety_stop), np.mean(stops_n), np.mean(is_futility_stop)

# --- PHASE 1: SEARCH ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    n_list = list(range(n_range[0], n_range[1] + 1, 5)) # Step by 5 for speed
    
    # EXPANDED SEARCH SPACE for Hurdles and Confidence
    hurdle_space = [p0, (p0 + p1)/2, p1 - 0.05]
    conf_space = [0.70, 0.80, 0.90, 0.95]

    with st.spinner("Searching wide parameter space..."):
        for n in n_list:
            if n < min_n_lead_in: continue
            for hurdle in hurdle_space:
                for conf in conf_space:
                    alpha, _, _, _ = run_fast_batch(1000, n, p0, 0.05, hurdle, conf, safe_limit, cohort_size, min_n_lead_in)
                    if alpha <= max_alpha:
                        power, _, _, _ = run_fast_batch(1000, n, p1, 0.05, hurdle, conf, safe_limit, cohort_size, min_n_lead_in)
                        _, tox_stop, _, _ = run_fast_batch(1000, n, p1, true_toxic_rate, hurdle, conf, safe_limit, cohort_size, min_n_lead_in)
                        
                        if power >= min_power and tox_stop >= min_safety_power:
                            results.append({"N": n, "Hurdle": hurdle, "Conf": conf, "Alpha": alpha, "Power": power, "Safety": tox_stop})

    if results:
        best = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['best_design'] = best
        st.success(f"### Optimal Design Found: Max N = {int(best['N'])}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max Enrollment", int(best['N']))
        c2.metric("Efficacy Power", f"{best['Power']:.1%}")
        c3.metric("Safety Detection", f"{best['Safety']:.1%}")
        c4.metric("Risk (Alpha)", f"{best['Alpha']:.2%}")
    else:
        st.error("No design found. To fix this, try:")
        st.write("1. **Increase N Search Range** (Try up to 150-200).")
        st.write(f"2. **Lower Lead-in N** (Currently {min_n_lead_in}).")
        st.write(f"3. **Widen Delta** (Current p0={p0} vs p1={p1} is a tight margin).")

# --- PHASE 2: STRESS TEST ---
st.markdown("---")
st.subheader("📋 Operational Stress-Tester")
if 'best_design' not in st.session_state:
    st.info("Run search above first.")
else:
    if st.button("📊 Run Multi-Scenario OC Simulations"):
        best = st.session_state['best_design']
        scenarios = [
            ("Super-Effective (Target + 10%)", p1 + 0.10, 0.05),
            ("On-Target (Goal Met)", p1, 0.05),
            ("Marginal (Mid-point)", (p0+p1)/2, 0.05),
            ("Null (Standard Care)", p0, 0.05),
            ("Futile (Below Null)", p0 - 0.10, 0.05),
            ("Toxic / High Efficacy", p1, true_toxic_rate),
            ("Toxic / Low Efficacy", p0, true_toxic_rate),
        ]
        stress_results = []
        for name, pe, ps in scenarios:
            pe = np.clip(pe, 0.01, 0.99)
            pow_v, stop_v, asn_v, fut_v = run_fast_batch(2000, int(best['N']), pe, ps, best['Hurdle'], best['Conf'], safe_limit, cohort_size, min_n_lead_in)
            stress_results.append({"Scenario": name, "Success %": f"{pow_v:.1%}", "Safety Stop %": f"{stop_v:.1%}", "Futility Stop %": f"{fut_v:.1%}", "Avg Patients (ASN)": f"{asn_v:.1f}"})
        st.table(pd.DataFrame(stress_results))
