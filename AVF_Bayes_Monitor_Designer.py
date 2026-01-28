import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Prior Sensitivity")
st.markdown("Updated: Prior Strength controls and Dynamic OC Interpretation added.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, help="Success rate of standard care.")
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, help="Goal success rate.")
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15, help="Max allowable SAE rate.")
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30, help="Hypothetical toxicity for testing.")

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Prior Strength (Bayesian Initial State)")
# NEW: Prior Strength Sliders
prior_alpha = st.sidebar.slider("Prior 'Successes' (α)", 1.0, 10.0, 1.0, step=0.5, 
    help="Default is 1.0 (Flat Prior). Increasing this assumes the drug already has some evidence of working before the trial starts.")
prior_beta = st.sidebar.slider("Prior 'Failures' (β)", 1.0, 10.0, 1.0, step=0.5, 
    help="Default is 1.0 (Flat Prior). Increasing this assumes a 'Skeptical Prior', requiring the drug to work harder to prove efficacy.")

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.005, 0.20, 0.01, step=0.005)
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.90)
min_safety_power = st.sidebar.slider("Min Safety Power", 0.70, 0.99, 0.95)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Thresholds")
eff_conf = st.sidebar.slider("Efficacy Success Confidence", 0.70, 0.99, 0.85)
safety_conf = st.sidebar.slider("Safety Stop Confidence", 0.50, 0.99, 0.90)
cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5)
n_range = st.sidebar.slider("N Search Range", 40, 150, (60, 100))

# --- STABLE VECTORIZED ENGINE (Updated for Priors) ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, e_conf, limit, cohort_sz, s_conf, p_a, p_b):
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    stops_n = np.full(sims, max_n)
    is_success, is_safety_stop, is_futility_stop, already_stopped = [np.zeros(sims, dtype=bool) for _ in range(4)]

    for n in range(cohort_sz, max_n + 1, cohort_sz):
        active = ~already_stopped
        if not np.any(active): break
        c_s, c_tox = np.sum(outcomes[active, :n], axis=1), np.sum(saes[active, :n], axis=1)
        
        # Bayesian math now uses the dynamic prior parameters
        prob_eff = 1 - beta.cdf(hurdle, p_a + c_s, p_b + (n - c_s))
        prob_tox = 1 - beta.cdf(limit, p_a + c_tox, p_b + (n - c_tox))
        
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
        is_success[remaining] = (1 - beta.cdf(hurdle, p_a + f_s, p_b + (max_n - f_s))) > e_conf
    return np.mean(is_success), np.mean(is_safety_stop), np.mean(stops_n), np.mean(is_futility_stop)

def newly_mapped(active, trig):
    m = np.zeros(len(active), dtype=bool); m[active] = trig; return m

# --- PHASE 1: SEARCH ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    n_list = list(range(n_range[0], n_range[1] + 1, 2))
    with st.spinner("Searching with chosen priors..."):
        for n in n_list:
            for hurdle in [0.55, 0.60, 0.65]:
                alpha, _, _, _ = run_fast_batch(2000, n, p0, 0.05, hurdle, eff_conf, safe_limit, n, safety_conf, prior_alpha, prior_beta)
                if alpha <= max_alpha:
                    pwr, _, _, _ = run_fast_batch(2000, n, p1, 0.05, hurdle, eff_conf, safe_limit, n, safety_conf, prior_alpha, prior_beta)
                    _, tox_p, _, _ = run_fast_batch(2000, n, p1, true_toxic_rate, hurdle, eff_conf, safe_limit, n, safety_conf, prior_alpha, prior_beta)
                    if pwr >= min_power and tox_p >= min_safety_power:
                        results.append({"N": n, "Hurdle": hurdle, "Alpha": alpha, "Power": pwr, "Safety": tox_p})
    
    if results:
        st.session_state['best_design'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['best_design_eff_conf'] = eff_conf
    else:
        st.error("No design found. Relax Risk Standards or Prior constraints.")

# --- PERSISTENT DISPLAY ---
if 'best_design' in st.session_state:
    best = st.session_state['best_design']
    used_eff_conf = st.session_state['best_design_eff_conf']
    
    st.success(f"### ✅ Optimal Design Parameters (Max N = {int(best['N'])})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Enrollment", int(best['N'])); c2.metric("Efficacy Power", f"{best['Power']:.1%}")
    c3.metric("Safety Detection", f"{best['Safety']:.1%}"); c4.metric("Risk (Alpha)", f"{best['Alpha']:.2%}")
    
    with st.expander("📝 Protocol Summary: Bayesian Monitoring Rules", expanded=True):
        st.markdown(f"""
        1. **Prior Assumption**: Trial starts with a $Beta({prior_alpha}, {prior_beta})$ prior (Strength: {prior_alpha + prior_beta} virtual pts).
        2. **Analysis Schedule**: Interim cohorts of **{cohort_size}**.
        3. **Efficacy Success**: Declare if $P(RR > {best['Hurdle']}) > {used_eff_conf}$.
        4. **Safety Stop**: Stop if $P(SAE > {safe_limit}) > {safety_conf}$.
        """)

    st.markdown("---")
    st.subheader("📊 Operational Characteristics (OC) Stress-Tester")
    if st.button("📈 Run Multi-Scenario Stress Test"):
        scenarios = [("Super-Effective", p1+0.1, 0.05), ("On-Target", p1, 0.05), ("Null (Fail)", p0, 0.05), ("High Tox", p1, true_toxic_rate)]
        stress_data = []
        for name, pe, ps in scenarios:
            pow_v, stop_v, asn_v, fut_v = run_fast_batch(2000, int(best['N']), pe, ps, best['Hurdle'], used_eff_conf, safe_limit, cohort_size, safety_conf, prior_alpha, prior_beta)
            stress_data.append({"Scenario": name, "Succ%": pow_v, "Saf%": stop_v, "ASN": asn_v})
        
        df_stress = pd.DataFrame(stress_data)
        st.table(df_stress)
        
        # NEW: Interpretation Logic
        st.info("### 🧐 Summary Interpretation")
        interp = []
        # Check Toxicity Sensitivity
        tox_row = df_stress[df_stress['Scenario'] == "High Tox"].iloc[0]
        if tox_row['Saf%'] > 0.95:
            interp.append(f"✅ **Safety Monitor is Robust**: Successfully catches toxic drugs in {tox_row['Saf%']:.1%} of cases with an average of only {tox_row['ASN']:.1} patients.")
        
        # Check Early Winning
        eff_row = df_stress[df_stress['Scenario'] == "Super-Effective"].iloc[0]
        if eff_row['ASN'] < (best['N'] * 0.6):
            interp.append(f"⚡ **High Efficiency**: For a strong drug, the trial 'graduates' early, saving ~{best['N'] - eff_row['ASN']:.0f} patients on average.")
            
        # Check Prior Impact
        if prior_beta > prior_alpha:
            interp.append("🛡️ **Skeptical Guardrail**: Your skeptical prior is actively suppressing false positives, but verify it isn't 'choking' early efficacy signals.")

        for line in interp:
            st.write(line)
