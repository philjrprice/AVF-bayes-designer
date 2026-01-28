import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Stress-Tester")
st.markdown("Reverted to stabilized version with expanded, independent Stress-Testing suite.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, 
    help="The success rate of the current standard of care.")

p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, 
    help="The 'Goal' success rate you hope the drug achieves.")

safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15, 
    help="Maximum allowable SAE rate.")

true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30, 
    help="Hypothetical rate to test safety detection.")

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.005, 0.20, 0.01, step=0.005, 
    help="Risk of a 'False Win.'")

min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.90, 
    help="Probability of correctly identifying success.")

min_safety_power = st.sidebar.slider("Min Safety Power (Detection)", 0.70, 0.99, 0.95, 
    help="Probability of triggering a safety stop if toxic.")

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Settings")
cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5, 
    help="Frequency of Bayesian monitoring checks.")

n_range = st.sidebar.slider("N Search Range", 40, 150, (60, 100), 
    help="The range of patients to test.")

# --- STABLE VECTORIZED ENGINE ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, conf, limit, cohort_sz, safe_conf=0.90):
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    
    stops_n = np.full(sims, max_n)
    is_success, is_safety_stop, is_futility_stop, already_stopped = [np.zeros(sims, dtype=bool) for _ in range(4)]

    for n in range(cohort_sz, max_n + 1, cohort_sz):
        active = ~already_stopped
        if not np.any(active): break
        
        curr_succ, curr_saes = np.sum(outcomes[active, :n], axis=1), np.sum(saes[active, :n], axis=1)
        
        prob_eff = 1 - beta.cdf(hurdle, 1 + curr_succ, 1 + (n - curr_succ))
        prob_tox = 1 - beta.cdf(limit, 1 + curr_saes, 1 + (n - curr_saes))
        
        tox_trigger, eff_trigger = prob_tox > safe_conf, prob_eff > conf
        fut_trigger = (n >= max_n/2) & (prob_eff < 0.05)
        
        new_stops = active.copy()
        new_stops[active] = (tox_trigger | eff_trigger | fut_trigger)
        
        is_safety_stop[active & newly_mapped_tox(active, tox_trigger)] = True
        is_success[active & newly_mapped_eff(active, eff_trigger, tox_trigger)] = True
        is_futility_stop[active & newly_mapped_fut(active, fut_trigger, tox_trigger, eff_trigger)] = True
        
        stops_n[new_stops & ~already_stopped] = n
        already_stopped[new_stops] = True

    remaining = ~already_stopped
    if np.any(remaining):
        final_succ = np.sum(outcomes[remaining, :max_n], axis=1)
        is_success[remaining] = (1 - beta.cdf(hurdle, 1 + final_succ, 1 + (max_n - final_succ))) > conf

    return np.mean(is_success), np.mean(is_safety_stop), np.mean(stops_n), np.mean(is_futility_stop)

def newly_mapped_tox(active, tox_trig):
    m = np.zeros(len(active), dtype=bool); m[active] = tox_trig; return m
def newly_mapped_eff(active, eff_trig, tox_trig):
    m = np.zeros(len(active), dtype=bool); m[active] = (eff_trig & ~tox_trig); return m
def newly_mapped_fut(active, fut_trig, tox_trig, eff_trig):
    m = np.zeros(len(active), dtype=bool); m[active] = (fut_trig & ~tox_trig & ~eff_trig); return m

# --- PHASE 1: SEARCH ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    n_list = list(range(n_range[0], n_range[1] + 1, 2))
    STABILITY_SIMS = 2000 
    
    with st.spinner(f"Running {STABILITY_SIMS} simulations per design point..."):
        for n in n_list:
            for hurdle in [0.55, 0.60, 0.65]:
                for conf in [0.74, 0.80, 0.85, 0.90]:
                    alpha, _, _, _ = run_fast_batch(STABILITY_SIMS, n, p0, 0.05, hurdle, conf, safe_limit, n)
                    if alpha <= max_alpha:
                        power, _, _, _ = run_fast_batch(STABILITY_SIMS, n, p1, 0.05, hurdle, conf, safe_limit, n)
                        _, tox_stop, _, _ = run_fast_batch(STABILITY_SIMS, n, p1, true_toxic_rate, hurdle, conf, safe_limit, n)
                        if power >= min_power and tox_stop >= min_safety_power:
                            results.append({"N": n, "Hurdle": hurdle, "Conf": conf, "Alpha": alpha, "Power": power, "Safety": tox_stop})

    if results:
        best = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['best_design'] = best
        st.success(f"### ✅ Stabilized Design Found: Max N = {int(best['N'])}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max Enrollment", int(best['N'])); c2.metric("Efficacy Power", f"{best['Power']:.1%}"); c3.metric("Safety Stop Prob.", f"{best['Safety']:.1%}"); c4.metric("Risk (Alpha)", f"{best['Alpha']:.2%}")
    else:
        st.error("No design found. Try relaxing Risk Standards.")

# --- PHASE 2: INDEPENDENT STRESS TESTER ---
st.markdown("---")
st.subheader("📋 Operational Characteristics (OC) Stress-Tester")
if 'best_design' not in st.session_state:
    st.info("Run the 'Find Optimal Sample Size' search above first.")
else:
    if st.button("📊 Run Multi-Scenario OC Stress Test"):
        best = st.session_state['best_design']
        # Expanded scenario list adapting to user inputs
        scenarios = [
            ("1. Super-Effective (Target + 10%)", p1 + 0.1, 0.05),
            ("2. On-Target (Goal Met)", p1, 0.05),
            ("3. Marginal (Midpoint)", (p0 + p1)/2, 0.05),
            ("4. Null (Standard Care)", p0, 0.05),
            ("5. Futile (Below Null)", p0 - 0.1, 0.05),
            ("6. High Eff / Toxic", p1 + 0.1, true_toxic_rate),
            ("7. Target Eff / Toxic", p1, true_toxic_rate),
            ("8. Null / Toxic", p0, true_toxic_rate),
        ]
        stress_data = []
        with st.spinner("Simulating thousands of trials per scenario..."):
            for name, pe, ps in scenarios:
                pe = np.clip(pe, 0.01, 0.99)
                pow_v, stop_v, asn_v, fut_v = run_fast_batch(2000, int(best['N']), pe, ps, best['Hurdle'], best['Conf'], safe_limit, cohort_size)
                stress_data.append({"Scenario": name, "Success %": f"{pow_v*100:.1f}%", "Safety Stop %": f"{stop_v*100:.1f}%", "Futility Stop %": f"{fut_v*100:.1f}%", "Avg N (ASN)": f"{asn_v:.1f}"})
        st.table(pd.DataFrame(stress_data))
        st.caption(f"Design: Max N={int(best['N'])}, Hurdle={best['Hurdle']}, Conf={best['Conf']}, Cohort={cohort_size}")
