import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Stress-Tester")
st.markdown("Full Code: Independent Efficacy/Safety Confidence with persistent OC Stress-Testing.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, 
    help="The success rate of the current standard of care. If the drug performs at or below this level, it is considered a failure.")

p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, 
    help="The 'Goal' success rate you hope the drug achieves.")

safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15, 
    help="The maximum allowable rate of Serious Adverse Events (SAEs).")

true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30, 
    help="Hypothetical toxicity rate used to test if the monitor is strong enough to catch a dangerous drug.")

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.005, 0.20, 0.01, step=0.005, 
    help="Risk of a 'False Win.' 0.01 means a 1% chance of calling a failing drug a success.")

min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.90, 
    help="The probability of correctly identifying a successful drug.")

min_safety_power = st.sidebar.slider("Min Safety Power (Detection)", 0.70, 0.99, 0.95, 
    help="The probability that the trial will stop if the drug is actually toxic.")

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Thresholds")
eff_conf = st.sidebar.slider("Efficacy Success Confidence", 0.70, 0.99, 0.85, 
    help="The probability threshold required to declare efficacy success. Higher = harder to stop early.")

safety_conf = st.sidebar.slider("Safety Stop Confidence", 0.50, 0.99, 0.90, 
    help="The probability threshold required to trigger a safety stop. Lower = more sensitive/precautionary.")

cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5, 
    help="How often the monitor checks the data.")

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
        
        curr_succ, curr_saes = np.sum(outcomes[active, :n], axis=1), np.sum(saes[active, :n], axis=1)
        
        prob_eff = 1 - beta.cdf(hurdle, 1 + curr_succ, 1 + (n - curr_succ))
        prob_tox = 1 - beta.cdf(limit, 1 + curr_saes, 1 + (n - curr_saes))
        
        tox_trig = prob_tox > s_conf
        eff_trig = prob_eff > e_conf
        fut_trig = (n >= max_n/2) & (prob_eff < 0.05)
        
        new_stops = active.copy()
        new_stops[active] = (tox_trig | eff_trig | fut_trig)
        
        is_safety_stop[active & newly_mapped(active, tox_trig)] = True
        is_success[active & newly_mapped(active, eff_trig & ~tox_trig)] = True
        is_futility_stop[active & newly_mapped(active, fut_trig & ~tox_trig & ~eff_trig)] = True
        
        stops_n[new_stops & ~already_stopped] = n
        already_stopped[new_stops] = True

    remaining = ~already_stopped
    if np.any(remaining):
        final_succ = np.sum(outcomes[remaining, :max_n], axis=1)
        is_success[remaining] = (1 - beta.cdf(hurdle, 1 + final_succ, 1 + (max_n - final_succ))) > e_conf

    return np.mean(is_success), np.mean(is_safety_stop), np.mean(stops_n), np.mean(is_futility_stop)

def newly_mapped(active, trig):
    m = np.zeros(len(active), dtype=bool)
    m[active] = trig
    return m

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
    used_eff_conf = st.session_state['best_design_eff_conf']
    
    st.success(f"### ✅ Optimal Design Parameters (Max N = {int(best['N'])})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Enrollment", int(best['N']))
    c2.metric("Efficacy Power", f"{best['Power']:.1%}")
    c3.metric("Safety Detection", f"{best['Safety']:.1%}")
    c4.metric("Risk (Alpha)", f"{best['Alpha']:.2%}")
    
    with st.expander("📝 Protocol Summary: Bayesian Monitoring Rules", expanded=True):
        st.markdown(f"""
        1. **Interim Analysis Schedule**: Data monitored in cohorts of **{cohort_size}** patients.
        2. **Efficacy Success Rule**: Declare success if $P(\\text{{Response Rate}} > {best['Hurdle']}) > {used_eff_conf}$.
        3. **Safety Stopping Rule**: Stop for toxicity if $P(\\text{{SAE Rate}} > {safe_limit}) > {safety_conf}$.
        4. **Futility Rule**: At patient **{int(best['N']//2)}**, stop if the probability of success is **< 5%**.
        """)

    st.markdown("---")
    st.subheader("📊 Operational Characteristics (OC) Stress-Tester")
    if st.button("📈 Run Multi-Scenario Stress Test"):
        scenarios = [
            ("1. Super-Effective", p1 + 0.1, 0.05),
            ("2. On-Target (Goal)", p1, 0.05),
            ("3. Marginal", (p0 + p1)/2, 0.05),
            ("4. Null (Standard Care)", p0, 0.05),
            ("5. Futile (Below Null)", p0 - 0.1, 0.05),
            ("6. High Eff / Toxic", p1 + 0.1, true_toxic_rate),
            ("7. Target Eff / Toxic", p1, true_toxic_rate),
            ("8. Null / Toxic", p0, true_toxic_rate),
        ]
        stress_data = []
        with st.spinner("Running simulations..."):
            for name, pe, ps in scenarios:
                pe = np.clip(pe, 0.01, 0.99)
                pow_v, stop_v, asn_v, fut_v = run_fast_batch(2000, int(best['N']), pe, ps, best['Hurdle'], used_eff_conf, safe_limit, cohort_size, safety_conf)
                stress_data.append({
                    "Scenario": name, 
                    "Success %": f"{pow_v*100:.1f}%", 
                    "Safety Stop %": f"{stop_v*100:.1f}%", 
                    "Futility Stop %": f"{fut_v*100:.1f}%",
                    "Avg N (ASN)": f"{asn_v:.1f}"
                })
        st.table(pd.DataFrame(stress_data))
