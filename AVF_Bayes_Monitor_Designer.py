import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Specialized Priors")
st.markdown("Updated v16: Synchronized Search Engine & Precision Adaptive Logic.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, 
    help="The success rate of the current standard of care.")
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, 
    help="The 'Goal' success rate you hope the drug achieves.")
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15, 
    help="The maximum allowable rate of Serious Adverse Events (SAEs).")
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30, 
    help="For testing purposes: If the drug were actually this dangerous, how well does the trial stop?")

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Efficacy Prior Strength")
prior_alpha = st.sidebar.slider("Eff Prior α", 1.0, 10.0, 1.0, step=0.5)
prior_beta = st.sidebar.slider("Eff Prior β", 1.0, 10.0, 1.0, step=0.5)

st.sidebar.header("🛡️ Safety Prior Strength")
s_prior_alpha = st.sidebar.slider("Saf Prior α", 1.0, 10.0, 1.0, step=0.5)
s_prior_beta = st.sidebar.slider("Saf Prior β", 1.0, 10.0, 1.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.005, 0.20, 0.01, step=0.005)
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.90)
min_safety_power = st.sidebar.slider("Min Safety Power (Detection)", 0.70, 0.99, 0.95)

st.sidebar.markdown("---")
st.sidebar.header("🔬 Simulation Rigor")
n_sims = st.sidebar.select_slider("Simulations", options=[2000, 5000, 10000], value=2000)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Thresholds")
eff_conf = st.sidebar.slider("Efficacy Success Confidence", 0.70, 0.99, 0.85)
safety_conf = st.sidebar.slider("Safety Stop Confidence", 0.50, 0.99, 0.90)
fut_conf = st.sidebar.slider("Futility Stop Threshold", 0.01, 0.20, 0.05)
cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5)
n_range = st.sidebar.slider("N Search Range", 40, 150, (60, 100))

# --- FIXED VECTORIZED ENGINE ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, e_conf, limit, cohort_sz, s_conf, f_conf, p_a, p_b, s_a, s_b):
    np.random.seed(42)
    p_eff, p_sae = np.clip(p_eff, 0.001, 0.999), np.clip(p_sae, 0.001, 0.999)
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    stops_n = np.full(sims, max_n)
    is_success, is_safety_stop, is_futility_stop, already_stopped = [np.zeros(sims, dtype=bool) for _ in range(4)]

    # Synchronized looking schedule
    for n in range(cohort_sz, max_n + 1, cohort_sz):
        active = ~already_stopped
        if not np.any(active): break
        c_s, c_tox = np.sum(outcomes[active, :n], axis=1), np.sum(saes[active, :n], axis=1)
        prob_eff = 1 - beta.cdf(hurdle, p_a + c_s, p_b + (n - c_s))
        prob_tox = 1 - beta.cdf(limit, s_a + c_tox, s_b + (n - c_tox))
        
        tox_trig, eff_trig = prob_tox > s_conf, prob_eff > e_conf
        fut_trig = (n >= max_n/2) & (prob_eff < f_conf)
        
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

# --- SYNCHRONIZED SEARCH ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    n_list = range(n_range[0], n_range[1] + 1, 2)
    hurdle_options = np.linspace(p0, (p0 + p1)/2, 5)
    
    with st.spinner(f"Analyzing {n_sims:,} simulations..."):
        for n in n_list:
            for h in hurdle_options:
                h = round(float(h), 3)
                # FIX: Search now uses the actual cohort_size instead of 'n'
                a, _, _, _ = run_fast_batch(n_sims, n, p0, 0.05, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta)
                if a <= max_alpha:
                    p, _, _, _ = run_fast_batch(n_sims, n, p1, 0.05, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta)
                    _, sp, _, _ = run_fast_batch(n_sims, n, p1, true_toxic_rate, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta)
                    if p >= min_power and sp >= min_safety_power:
                        results.append({"N": n, "Hurdle": h, "Alpha": a, "Power": p, "Safety": sp})
                        break
    if results:
        st.session_state['best'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['params'] = {"p0": p0, "p1": p1, "limit": safe_limit, "tox": true_toxic_rate, "e_c": eff_conf, "s_c": safety_conf, "f_c": fut_conf, "p_a": prior_alpha, "p_b": prior_beta, "s_a": s_prior_alpha, "s_b": s_prior_beta, "cohort": cohort_size, "sims": n_sims}
    else:
        st.error("❌ No design found. Suggestions: 1. Increase Max Alpha. 2. Decrease Success Confidence. 3. Increase N Range.")

# --- PERSISTENT DISPLAY (RESTORED FROM V15) ---
if 'best' in st.session_state:
    best, up = st.session_state['best'], st.session_state['params']
    st.success(f"### ✅ Optimal Design Found (Max N = {int(best['N'])})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Enrollment", int(best['N']))
    c2.metric("Efficacy Power", f"{best['Power']:.1%}")
    c3.metric("Safety Detection", f"{best['Safety']:.1%}")
    c4.metric("Risk (Alpha)", f"{best['Alpha']:.2%}")

    with st.expander("📝 Protocol Rules", expanded=True):
        st.write(f"1. **Monitoring**: Analysis every {up['cohort']} patients.")
        st.write(f"2. **Success**: Stop if $P(Rate > {best['Hurdle']}) > {up['e_c']}$.")
        st.write(f"3. **Safety**: Stop if $P(SAE Rate > {up['limit']}) > {up['s_c']}$.")
        st.write(f"4. **Futility**: From patient {int(best['N']/2)} onwards, stop if $P(Success) < {up['f_c']}$.")

    if st.button("📈 Run Multi-Scenario Stress Test"):
        scenarios = [("Target", up['p1'], 0.05), ("Null", up['p0'], 0.05), ("Toxic", up['p1'], up['tox'])]
        stress_data = []
        for name, pe, ps in scenarios:
            pw, stp, asn, fut = run_fast_batch(up['sims'], int(best['N']), pe, ps, best['Hurdle'], up['e_c'], up['limit'], up['cohort'], up['s_c'], up['f_c'], up['p_a'], up['p_b'], up['s_a'], up['s_b'])
            stress_data.append({"Scenario": name, "Success %": f"{pw:.1%}", "Safety Stop %": f"{stp:.1%}", "ASN": f"{asn:.1f}"})
        st.table(pd.DataFrame(stress_data))
