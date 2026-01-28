import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Specialized Priors")
st.markdown("Updated v17: Restored full scenario suite, detailed priors, and lead-in logic.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, 
    help="The success rate of the current standard of care.")
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, 
    help="The 'Goal' success rate you hope the drug achieves.")
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15, 
    help="The maximum allowable rate of Serious Adverse Events (SAEs).")
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30, 
    help="Testing: If the drug were this dangerous, how well does the trial stop?")

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Efficacy Prior Strength")
prior_alpha = st.sidebar.slider("Eff Prior 'Successes' (α_eff)", 1.0, 10.0, 1.0, step=0.5)
prior_beta = st.sidebar.slider("Eff Prior 'Failures' (β_eff)", 1.0, 10.0, 1.0, step=0.5)

st.sidebar.header("🛡️ Safety Prior Strength")
s_prior_alpha = st.sidebar.slider("Saf Prior 'Events' (α_saf)", 1.0, 10.0, 1.0, step=0.5)
s_prior_beta = st.sidebar.slider("Saf Prior 'Non-Events' (β_saf)", 1.0, 10.0, 1.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.005, 0.20, 0.01, step=0.005)
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.90)
min_safety_power = st.sidebar.slider("Min Safety Power (Detection)", 0.70, 0.99, 0.95)

st.sidebar.markdown("---")
st.sidebar.header("🔬 Simulation Rigor")
n_sims = st.sidebar.select_slider(
    "Number of Simulations",
    options=[2000, 5000, 7500, 10000, 12500, 15000],
    value=2000
)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Thresholds")
min_n_lead = st.sidebar.slider("Min N Before First Check", 5, 50, 20)
eff_conf = st.sidebar.slider("Efficacy Success Confidence", 0.70, 0.99, 0.85)
safety_conf = st.sidebar.slider("Safety Stop Confidence", 0.50, 0.99, 0.90)
fut_conf = st.sidebar.slider("Futility Stop Threshold", 0.01, 0.20, 0.05)

cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5)
n_range = st.sidebar.slider("N Search Range", 40, 150, (60, 100))

# --- STABLE VECTORIZED ENGINE ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, e_conf, limit, cohort_sz, s_conf, f_conf, p_a, p_b, s_a, s_b, min_n):
    np.random.seed(42)
    p_eff = np.clip(p_eff, 0.001, 0.999)
    p_sae = np.clip(p_sae, 0.001, 0.999)
    
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    stops_n = np.full(sims, max_n)
    is_success, is_safety_stop, is_futility_stop, already_stopped = [np.zeros(sims, dtype=bool) for _ in range(4)]

    # Analysis points start at min_n
    look_points = sorted(list(set([min_n] + [n for n in range(min_n, max_n + 1, cohort_sz)])))

    for n in look_points:
        if n > max_n: break
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

# --- SEARCH ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    n_list = list(range(n_range[0], n_range[1] + 1, 2))
    hurdle_options = np.linspace(p0, (p0 + p1)/2, 5)
    
    with st.spinner(f"Searching for optimal design..."):
        for n in n_list:
            if n < min_n_lead: continue 
            for hurdle in hurdle_options:
                hurdle = round(float(hurdle), 3)
                alpha, _, _, _ = run_fast_batch(n_sims, n, p0, 0.05, hurdle, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta, min_n_lead)
                if alpha <= max_alpha:
                    pwr, _, _, _ = run_fast_batch(n_sims, n, p1, 0.05, hurdle, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta, min_n_lead)
                    _, tox_p, _, _ = run_fast_batch(n_sims, n, p1, true_toxic_rate, hurdle, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta, min_n_lead)
                    if pwr >= min_power and tox_p >= min_safety_power:
                        results.append({"N": n, "Hurdle": hurdle, "Alpha": alpha, "Power": pwr, "Safety": tox_p})
    
    if results:
        st.session_state['best_design'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['used_params'] = {
            "p0": p0, "p1": p1, "safe_limit": safe_limit, "toxic_rate": true_toxic_rate,
            "eff_conf": eff_conf, "saf_conf": safety_conf, "fut_conf": fut_conf,
            "p_a": prior_alpha, "p_b": prior_beta, "s_a": s_prior_alpha, "s_b": s_prior_beta,
            "alpha_req": max_alpha, "pwr_req": min_power, "saf_pwr_req": min_safety_power,
            "cohort": cohort_size, "sim_rigor": n_sims, "min_n": min_n_lead
        }

# --- PERSISTENT DISPLAY ---
if 'best_design' in st.session_state:
    best, up = st.session_state['best_design'], st.session_state['used_params']
    
    st.success(f"### ✅ Optimal Design Parameters (Max N = {int(best['N'])})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Enrollment", int(best['N']))
    c2.metric("Efficacy Power", f"{best['Power']:.1%}")
    c3.metric("Safety Detection", f"{best['Safety']:.1%}")
    c4.metric("Risk (Alpha)", f"{best['Alpha']:.2%}")
    
    with st.expander("📝 Protocol Summary & Final Analysis Rules", expanded=True):
        st.write(f"1. **Analysis Schedule**: Lead-in of **{up['min_n']}** patients. Thereafter, cohorts of **{up['cohort']}**.")
        st.write(f"2. **Interim Success**: After N={up['min_n']}, stop if $P(Rate > {best['Hurdle']}) > {up['eff_conf']}$.")
        st.write(f"3. **Safety Stop**: After N={up['min_n']}, stop if $P(SAE Rate > {up['safe_limit']}) > {up['saf_conf']}$.")
        st.write(f"4. **Futility Rule**: From patient {int(best['N']/2)}, stop if $P(Success) < {up['fut_conf']}$.")
        st.write(f"5. **Final Analysis**: Successful if $P(Rate > {best['Hurdle']})$ exceeds **{up['eff_conf']}**.")

    # --- OC CURVE ---
    st.markdown("---")
    st.subheader("📈 Operating Characteristic (OC) Curve")
    eff_range = np.linspace(max(0, up['p0'] - 0.15), min(1, up['p1'] + 0.15), 15)
    oc_probs, saf_probs = [], []
    with st.spinner("Generating OC Data..."):
        for pe in eff_range:
            p_succ, _, _, _ = run_fast_batch(up['sim_rigor'], int(best['N']), pe, 0.05, best['Hurdle'], up['eff_conf'], up['safe_limit'], up['cohort'], up['saf_conf'], up['fut_conf'], up['p_a'], up['p_b'], up['s_a'], up['s_b'], up['min_n'])
            oc_probs.append(p_succ)
            _, p_saf_stop, _, _ = run_fast_batch(up['sim_rigor'], int(best['N']), pe, pe/2, best['Hurdle'], up['eff_conf'], up['safe_limit'], up['cohort'], up['saf_conf'], up['fut_conf'], up['p_a'], up['p_b'], up['s_a'], up['s_b'], up['min_n'])
            saf_probs.append(p_saf_stop)
    
    st.session_state['oc_chart_data'] = pd.DataFrame({"True_Rate": eff_range, "PoS": oc_probs, "SafetyStop": saf_probs})
    fig_oc, ax_oc = plt.subplots(figsize=(10, 4))
    ax_oc.plot(eff_range, oc_probs, marker='o', color='teal', label='Prob. Success')
    ax_oc.plot(eff_range, saf_probs, marker='x', linestyle=':', color='orange', label='Prob. Safety Stop')
    ax_oc.axvline(up['p0'], color='red', linestyle='--', label='Null')
    ax_oc.axvline(up['p1'], color='green', linestyle='--', label='Target')
    ax_oc.legend(); ax_oc.grid(alpha=0.3)
    st.pyplot(fig_oc)

    # --- RESTORED: FULL 8-SCENARIO STRESS TEST ---
    st.markdown("---")
    st.subheader("📊 Operational Stress-Tester (Full Suite)")
    if st.button("📈 Run Multi-Scenario Stress Test"):
        scenarios = [
            (f"1. Super-Effective (Eff: {min(1.0, up['p1']+0.1):.0%}, Saf: 5%)", up['p1'] + 0.1, 0.05),
            (f"2. On-Target (Eff: {up['p1']:.0%}, Saf: 5%)", up['p1'], 0.05),
            (f"3. Marginal (Eff: {(up['p0']+up['p1'])/2:.0%}, Saf: 5%)", (up['p0'] + up['p1'])/2, 0.05),
            (f"4. Null (Eff: {up['p0']:.0%}, Saf: 5%)", up['p0'], 0.05),
            (f"5. Futile (Eff: {max(0.0, up['p0']-0.1):.0%}, Saf: 5%)", up['p0'] - 0.1, 0.05),
            (f"6. High Eff / Toxic (Eff: {min(1.0, up['p1']+0.1):.0%}, Saf: {up['toxic_rate']:.0%})", up['p1'] + 0.1, up['toxic_rate']),
            (f"7. Target Eff / Toxic (Eff: {up['p1']:.0%}, Saf: {up['toxic_rate']:.0%})", up['p1'], up['toxic_rate']),
            (f"8. Null / Toxic (Eff: {up['p0']:.0%}, Saf: {up['toxic_rate']:.0%})", up['p0'], up['toxic_rate']),
        ]
        stress_data = []
        for name, pe, ps in scenarios:
            pe = np.clip(pe, 0.001, 0.999)
            pw, stp, asn, fut = run_fast_batch(up['sim_rigor'], int(best['N']), pe, ps, best['Hurdle'], up['eff_conf'], up['safe_limit'], up['cohort'], up['saf_conf'], up['fut_conf'], up['p_a'], up['p_b'], up['s_a'], up['s_b'], up['min_n'])
            stress_data.append({"Scenario": name, "Success %": pw, "Safety Stop %": stp, "Futility %": fut, "ASN": asn})
        
        st.session_state['stress_results'] = pd.DataFrame(stress_data)
        st.table(st.session_state['stress_results'].assign(**{
            "Success %": lambda x: x["Success %"].apply(lambda y: f"{y:.1%}"),
            "Safety Stop %": lambda x: x["Safety Stop %"].apply(lambda y: f"{y:.1%}"),
            "Futility %": lambda x: x["Futility %"].apply(lambda y: f"{y:.1%}"),
            "Avg N (ASN)": lambda x: x["ASN"].apply(lambda y: f"{y:.1f}")
        }))

    # --- RESTORED: BETA PRIOR DENSITY PLOTS ---
    st.markdown("---")
    st.subheader("📈 Bayesian Prior Densities")
    x = np.linspace(0, 1, 100)
    c_p1, c_p2 = st.columns(2)
    y_eff = beta.pdf(x, up['p_a'], up['p_b'])
    y_saf = beta.pdf(x, up['s_a'], up['s_b'])
    st.session_state['eff_prior_data'] = pd.DataFrame({"Rate": x, "Density": y_eff})
    st.session_state['saf_prior_data'] = pd.DataFrame({"Rate": x, "Density": y_saf})
    
    with c_p1:
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        ax1.plot(x, y_eff, color='blue', label='Eff Prior'); ax1.axvline(up['p0'], color='red', linestyle='--'); ax1.set_title("Efficacy Prior"); st.pyplot(fig1)
    with c_p2:
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        ax2.plot(x, y_saf, color='orange', label='Saf Prior'); ax2.axvline(up['safe_limit'], color='red', linestyle='--'); ax2.set_title("Safety Prior"); st.pyplot(fig2)

    # --- RESTORED: COMPREHENSIVE EXPORT ---
    if 'stress_results' in st.session_state:
        st.markdown("---")
        report_params = pd.DataFrame([up]).T.reset_index().rename(columns={"index": "Metric", 0: "Value"})
        report_results = pd.DataFrame([best]).T.reset_index().rename(columns={"index": "Metric", 0: "Value"})
        combined = pd.concat([
            pd.DataFrame([{"Metric": "--- DESIGN ---", "Value": ""}]), report_params,
            pd.DataFrame([{"Metric": "--- RESULTS ---", "Value": ""}]), report_results,
            pd.DataFrame([{"Metric": "--- STRESS TEST ---", "Value": ""}]), st.session_state['stress_results'].rename(columns={"Scenario": "Metric", "Success %": "Value"}),
            pd.DataFrame([{"Metric": "--- OC DATA ---", "Value": ""}]), st.session_state['oc_chart_data'].rename(columns={"True_Rate": "Metric", "PoS": "Value"})
        ], axis=0, ignore_index=True)
        csv = combined.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Comprehensive Design Report", data=csv, file_name="AVF_Full_Design_v17.csv")
