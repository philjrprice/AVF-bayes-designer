import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Specialized Priors")
st.markdown("Updated: Added Export Report functionality.")

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
st.sidebar.header("⏱️ Adaptive Thresholds")
eff_conf = st.sidebar.slider("Efficacy Success Confidence", 0.70, 0.99, 0.85)
safety_conf = st.sidebar.slider("Safety Stop Confidence", 0.50, 0.99, 0.90)
fut_conf = st.sidebar.slider("Futility Stop Threshold", 0.01, 0.20, 0.05)

cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5)
n_range = st.sidebar.slider("N Search Range", 40, 150, (60, 100))

# --- STABLE VECTORIZED ENGINE ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, e_conf, limit, cohort_sz, s_conf, f_conf, p_a, p_b, s_a, s_b):
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    stops_n = np.full(sims, max_n)
    is_success, is_safety_stop, is_futility_stop, already_stopped = [np.zeros(sims, dtype=bool) for _ in range(4)]

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

# --- SEARCH ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    n_list = list(range(n_range[0], n_range[1] + 1, 2))
    with st.spinner("Searching for optimal design..."):
        for n in n_list:
            for hurdle in [0.55, 0.60, 0.65]:
                alpha, _, _, _ = run_fast_batch(2000, n, p0, 0.05, hurdle, eff_conf, safe_limit, n, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta)
                if alpha <= max_alpha:
                    pwr, _, _, _ = run_fast_batch(2000, n, p1, 0.05, hurdle, eff_conf, safe_limit, n, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta)
                    _, tox_p, _, _ = run_fast_batch(2000, n, p1, true_toxic_rate, hurdle, eff_conf, safe_limit, n, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta)
                    if pwr >= min_power and tox_p >= min_safety_power:
                        results.append({"N": n, "Hurdle": hurdle, "Alpha": alpha, "Power": pwr, "Safety": tox_p})
    
    if results:
        st.session_state['best_design'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['used_params'] = {
            "p0": p0, "p1": p1, "safe_limit": safe_limit, "toxic_rate": true_toxic_rate,
            "eff_conf": eff_conf, "saf_conf": safety_conf, "fut_conf": fut_conf,
            "p_a": prior_alpha, "p_b": prior_beta, "s_a": s_prior_alpha, "s_b": s_prior_beta,
            "alpha_req": max_alpha, "pwr_req": min_power, "saf_pwr_req": min_safety_power,
            "cohort": cohort_size
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
    
    with st.expander("📝 Protocol Summary: Bayesian Monitoring Rules", expanded=True):
        st.write(f"1. **Analysis Schedule**: Data monitored in cohorts of {up['cohort']} patients.")
        st.write(f"2. **Success Rule**: Declare success if $P(Response Rate > {best['Hurdle']}) > {up['eff_conf']}$.")
        st.write(f"3. **Safety Stop**: Terminate if $P(SAE Rate > {up['safe_limit']}) > {up['saf_conf']}$.")
        st.write(f"4. **Futility Rule**: From patient {int(best['N']/2)} onwards, stop if $P(Success) < {up['fut_conf']}$.")

    st.markdown("---")
    st.subheader("📊 Operational Characteristics (OC) Stress-Tester")
    if st.button("📈 Run Multi-Scenario Stress Test"):
        scenarios = [
            ("1. Super-Effective", up['p1'] + 0.1, 0.05), ("2. On-Target", up['p1'], 0.05),
            ("3. Marginal", (up['p0'] + up['p1'])/2, 0.05), ("4. Null", up['p0'], 0.05),
            ("5. Futile", up['p0'] - 0.1, 0.05), ("6. High Eff / Toxic", up['p1'] + 0.1, up['toxic_rate']),
            ("7. Target Eff / Toxic", up['p1'], up['toxic_rate']), ("8. Null / Toxic", up['p0'], up['toxic_rate']),
        ]
        stress_data = []
        for name, pe, ps in scenarios:
            pe = np.clip(pe, 0.01, 0.99)
            pow_v, stop_v, asn_v, fut_v = run_fast_batch(2000, int(best['N']), pe, ps, best['Hurdle'], up['eff_conf'], up['safe_limit'], up['cohort'], up['saf_conf'], up['fut_conf'], up['p_a'], up['p_b'], up['s_a'], up['s_b'])
            stress_data.append({"Scenario": name, "Success %": pow_v, "Safety Stop %": stop_v, "Futility Stop %": fut_v, "ASN": asn_v})
        
        df_oc = pd.DataFrame(stress_data)
        st.session_state['stress_results'] = df_oc
        st.table(df_oc.assign(**{
            "Success %": df_oc["Success %"].apply(lambda x: f"{x:.1%}"),
            "Safety Stop %": df_oc["Safety Stop %"].apply(lambda x: f"{x:.1%}"),
            "Futility Stop %": df_oc["Futility Stop %"].apply(lambda x: f"{x:.1%}"),
            "Avg N (ASN)": df_oc["ASN"].apply(lambda x: f"{x:.1f}")
        }).drop(columns="ASN"))

        st.info("### 🧐 OC Summary Interpretation")
        tox_capture = stress_data[6]["Safety Stop %"]
        tox_asn = stress_data[6]["ASN"]
        grad_asn = stress_data[0]["ASN"]
        savings = (1 - (grad_asn / best['N'])) * 100

        st.markdown(f"""
        * **Safety Guardrail**: Identifies toxic drugs with **{tox_capture:.1%} accuracy**, stopping at an average of **{tox_asn:.1f}** patients.
        * **Early Graduation**: Effective drugs save **{savings:.1f}%** of enrollment.
        * **Futility Efficiency**: The **{up['fut_conf']:.0%}** threshold protects resources when success is unlikely.
        """)

    # --- NEW: EXPORT REPORT BUTTON ---
    if 'stress_results' in st.session_state:
        st.markdown("---")
        # Compile Report Data
        report_params = pd.DataFrame([up]).T.reset_index().rename(columns={"index": "Parameter", 0: "Value"})
        report_results = pd.DataFrame([best]).T.reset_index().rename(columns={"index": "Metric", 0: "Result"})
        
        combined_report = pd.concat([
            pd.DataFrame([{"Parameter": "--- DESIGN SETTINGS ---", "Value": ""}]),
            report_params,
            pd.DataFrame([{"Parameter": "--- OPTIMAL RESULTS ---", "Value": ""}]),
            report_results,
            pd.DataFrame([{"Parameter": "--- STRESS TEST DATA ---", "Value": ""}]),
            st.session_state['stress_results']
        ], axis=0, ignore_index=True)

        csv = combined_report.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Full Design Report (CSV)",
            data=csv,
            file_name="AVF_Bayesian_Design_Report.csv",
            mime="text/csv",
            help="Downloads all design parameters, constraints, and stress-test performance data."
        )

    # --- BETA PLOTS ---
    st.markdown("---")
    st.subheader("📈 Bayesian Prior Probability Densities")
    x = np.linspace(0, 1, 100)
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        y_eff = beta.pdf(x, up['p_a'], up['p_b'])
        fig_eff, ax_eff = plt.subplots(figsize=(6, 3.5))
        ax_eff.plot(x, y_eff, color='blue', lw=2, label=f'Eff Prior: Beta({up["p_a"]}, {up["p_b"]})')
        ax_eff.fill_between(x, 0, y_eff, color='blue', alpha=0.1)
        ax_eff.axvline(up['p0'], color='red', linestyle='--', label=f'Null Hurdle ({up["p0"]})')
        ax_eff.set_title("Efficacy Prior Distribution", fontweight='bold'); ax_eff.set_xlabel("True Response Rate"); ax_eff.set_ylabel("Density"); ax_eff.legend(fontsize='small')
        st.pyplot(fig_eff)
    with col_plot2:
        y_saf = beta.pdf(x, up['s_a'], up['s_b'])
        fig_saf, ax_saf = plt.subplots(figsize=(6, 3.5))
        ax_saf.plot(x, y_saf, color='orange', lw=2, label=f'Saf Prior: Beta({up["s_a"]}, {up["s_b"]})')
        ax_saf.fill_between(x, 0, y_saf, color='orange', alpha=0.1)
        ax_saf.axvline(up['safe_limit'], color='red', linestyle='--', label=f'Safety Limit ({up["safe_limit"]})')
        ax_saf.set_title("Safety Prior Distribution", fontweight='bold'); ax_saf.set_xlabel("True SAE Rate"); ax_saf.set_ylabel("Density"); ax_saf.legend(fontsize='small')
        st.pyplot(fig_saf)

    # --- TREND EXPLANATION ---
    st.info("### 🧐 Prior Trend Analysis")
    eff_mode = (up['p_a'] - 1) / (up['p_a'] + up['p_b'] - 2) if (up['p_a'] + up['p_b']) > 2 else 0.5
    saf_mode = (up['s_a'] - 1) / (up['s_a'] + up['s_b'] - 2) if (up['s_a'] + up['s_b']) > 2 else 0.5
    st.markdown(f"""
    * **Efficacy Trend**: Prior is currently **{"Optimistic" if eff_mode > up['p0'] else "Skeptical" if eff_mode < up['p0'] else "Neutral"}**.
    * **Safety Trend**: Prior is **{"Cautious" if saf_mode > up['safe_limit']/2 else "Confident"}**.
    * **Prior Strength**: Efficacy weight = **{up['p_a'] + up['p_b']:.1f}** patients | Safety weight = **{up['s_a'] + up['s_b']:.1f}** patients.
    """)
