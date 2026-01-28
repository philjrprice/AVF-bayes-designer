import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Specialized Priors")
st.markdown("Updated: Restored Power sliders and added comprehensive tooltips to all controls.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, 
    help="The success rate of the current standard of care. If the new drug performs at or below this level, it is considered a failure.")
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, 
    help="The 'Goal' success rate you hope the drug achieves to justify further development.")
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15, 
    help="The maximum allowable rate of Serious Adverse Events (SAEs). If the drug is likely above this limit, the trial will stop for safety.")
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30, 
    help="For testing purposes: If the drug were actually this dangerous, how well does the trial detect it?")

st.sidebar.markdown("---")
# SECTION: Efficacy Priors
st.sidebar.header("⚖️ Efficacy Prior Strength")
prior_alpha = st.sidebar.slider("Eff Prior 'Successes' (α_eff)", 1.0, 10.0, 1.0, step=0.5, 
    help="Initial successes assumed for efficacy. Default 1.0 is a Flat/Non-informative Prior.")
prior_beta = st.sidebar.slider("Eff Prior 'Failures' (β_eff)", 1.0, 10.0, 1.0, step=0.5, 
    help="Initial failures assumed for efficacy. Increase this to create a 'Skeptical' prior regarding efficacy.")

# SECTION: Safety Priors
st.sidebar.header("🛡️ Safety Prior Strength")
s_prior_alpha = st.sidebar.slider("Saf Prior 'Events' (α_saf)", 1.0, 10.0, 1.0, step=0.5, 
    help="Initial SAEs assumed. Increasing this makes the safety monitor more 'Precautionary'.")
s_prior_beta = st.sidebar.slider("Saf Prior 'Non-Events' (β_saf)", 1.0, 10.0, 1.0, step=0.5, 
    help="Initial non-SAEs assumed. Increase to assume the drug is likely safe (Skeptical Safety Prior).")

st.sidebar.markdown("---")
# RESTORED POWER SLIDERS
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.005, 0.20, 0.01, step=0.005, 
    help="The maximum allowed probability of declaring a drug successful when its true rate is actually p0.")
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.90, 
    help="The minimum probability required to correctly detect a drug that hits the Target Efficacy (p1).")
min_safety_power = st.sidebar.slider("Min Safety Power (Detection)", 0.70, 0.99, 0.95, 
    help="The minimum probability required to stop the trial for a drug with the 'Toxic' SAE rate.")

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Thresholds")
eff_conf = st.sidebar.slider("Efficacy Success Confidence", 0.70, 0.99, 0.85, 
    help="The Bayesian posterior probability required to declare the drug a success at any interim or final analysis.")
safety_conf = st.sidebar.slider("Safety Stop Confidence", 0.50, 0.99, 0.90, 
    help="The Bayesian posterior probability required to stop the trial early for toxicity.")
fut_conf = st.sidebar.slider("Futility Stop Threshold", 0.01, 0.20, 0.05, 
    help="If the probability of success drops below this value, the trial stops early for futility.")

cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5, 
    help="How many patients are enrolled between each interim look.")
n_range = st.sidebar.slider("N Search Range", 40, 150, (60, 100), 
    help="The range of maximum sample sizes to search for the optimal design.")

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

# --- PHASE 1: SEARCH ---
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
                    # Applying power constraints in search
                    if pwr >= min_power and tox_p >= min_safety_power:
                        results.append({"N": n, "Hurdle": hurdle, "Alpha": alpha, "Power": pwr, "Safety": tox_p})
    
    if results:
        st.session_state['best_design'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['used_params'] = {
            "eff_conf": eff_conf, "saf_conf": safety_conf, "fut_conf": fut_conf,
            "p_a": prior_alpha, "p_b": prior_beta, "s_a": s_prior_alpha, "s_b": s_prior_beta
        }
    else:
        st.error("No design found. Try relaxing Risk Standards.")

# --- PERSISTENT DISPLAY ---
if 'best_design' in st.session_state:
    best = st.session_state['best_design']
    up = st.session_state['used_params']
    
    st.success(f"### ✅ Optimal Design Parameters (Max N = {int(best['N'])})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Enrollment", int(best['N']))
    c2.metric("Efficacy Power", f"{best['Power']:.1%}")
    c3.metric("Safety Detection", f"{best['Safety']:.1%}")
    c4.metric("Risk (Alpha)", f"{best['Alpha']:.2%}")
    
    st.info(f"**Decision Rules:** Success if $P(Eff > {best['Hurdle']}) > {up['eff_conf']}$ | Stop for Safety if $P(SAE > {safe_limit}) > {up['saf_conf']}$")

    st.markdown("---")
    st.subheader("📊 Operational Characteristics (OC) Stress-Tester")
    if st.button("📈 Run Multi-Scenario Stress Test"):
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
        with st.spinner("Running stress simulations..."):
            for name, pe, ps in scenarios:
                pe = np.clip(pe, 0.01, 0.99)
                pow_v, stop_v, asn_v, fut_v = run_fast_batch(2000, int(best['N']), pe, ps, best['Hurdle'], up['eff_conf'], safe_limit, cohort_size, up['saf_conf'], up['fut_conf'], up['p_a'], up['p_b'], up['s_a'], up['s_b'])
                stress_data.append({"Scenario": name, "Success %": pow_v, "Safety Stop %": stop_v, "Futility Stop %": fut_v, "ASN": asn_v})
        
        df_oc = pd.DataFrame(stress_data)
        st.table(df_oc.assign(**{
            "Success %": df_oc["Success %"].apply(lambda x: f"{x:.1%}"),
            "Safety Stop %": df_oc["Safety Stop %"].apply(lambda x: f"{x:.1%}"),
            "Futility Stop %": df_oc["Futility Stop %"].apply(lambda x: f"{x:.1%}"),
            "Avg N (ASN)": df_oc["ASN"].apply(lambda x: f"{x:.1f}")
        }).drop(columns="ASN"))

        # --- DYNAMIC INTERPRETATION ---
        st.info("### 🧐 Summary Interpretation")
        tox_capture = stress_data[6]["Safety Stop %"]
        tox_asn = stress_data[6]["ASN"]
        grad_asn = stress_data[0]["ASN"]
        savings = (1 - (grad_asn / best['N'])) * 100

        st.markdown(f"""
        * **Safety Guardrail**: The monitor identifies toxic drugs with **{tox_capture:.1%} accuracy**, stopping the trial at an average of **{tox_asn:.1f}** patients in the Target Eff / Toxic scenario.
        * **Ethical Efficiency**: For a highly effective drug, the design saves **{savings:.1f}%** of enrollment through early 'graduation'.
        * **Prior Analysis**: 
            * Efficacy Prior weight: **{up['p_a'] + up['p_b']:.1f}** virtual patients.
            * Safety Prior weight: **{up['s_a'] + up['s_b']:.1f}** virtual patients.
        """)
