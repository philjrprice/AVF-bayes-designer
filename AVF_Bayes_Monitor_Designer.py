import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Bayesian Visualization")
st.markdown("Updated: Added Dynamic Futility, Prior Curve Visualization, and Unified CSV Export.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5)
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7)
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30)

st.sidebar.markdown("---")
# SECTION: Efficacy Priors
st.sidebar.header("⚖️ Efficacy Prior Strength")
prior_alpha = st.sidebar.slider("Eff Prior Successes (α_eff)", 1.0, 10.0, 1.0, step=0.5)
prior_beta = st.sidebar.slider("Eff Prior Failures (β_eff)", 1.0, 10.0, 1.0, step=0.5)

# SECTION: Safety Priors
st.sidebar.header("🛡️ Safety Prior Strength")
s_prior_alpha = st.sidebar.slider("Saf Prior Events (α_saf)", 1.0, 10.0, 1.0, step=0.5)
s_prior_beta = st.sidebar.slider("Saf Prior Non-Events (β_saf)", 1.0, 10.0, 1.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Thresholds")
eff_conf = st.sidebar.slider("Efficacy Success Confidence", 0.70, 0.99, 0.85)
safety_conf = st.sidebar.slider("Safety Stop Confidence", 0.50, 0.99, 0.90)
# NEW FEATURE: Futility Confidence Slider
fut_conf = st.sidebar.slider("Futility Stop Threshold", 0.01, 0.20, 0.05, 
    help="Stop if Prob(Efficacy > Hurdle) falls below this level (e.g., 0.05 for 5%).")

cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5)
n_range = st.sidebar.slider("N Search Range", 40, 150, (60, 100))

# --- NEW: PRIOR VISUALIZATION ---
st.subheader("📊 Bayesian Prior Visualization")
c_plot1, c_plot2 = st.columns(2)

def plot_prior(a, b, title, color, target):
    x = np.linspace(0, 1, 100)
    y = beta.pdf(x, a, b)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y, color=color, lw=2, label='Prior Distribution')
    ax.fill_between(x, 0, y, color=color, alpha=0.2)
    ax.axvline(target, color='red', linestyle='--', label=f'Threshold ({target})')
    ax.set_title(title)
    ax.set_xlabel("Probability")
    ax.legend()
    return fig

with c_plot1:
    st.pyplot(plot_prior(prior_alpha, prior_beta, "Efficacy Prior Profile", "blue", p0))
with c_plot2:
    st.pyplot(plot_prior(s_prior_alpha, s_prior_beta, "Safety Prior Profile", "orange", safe_limit))

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
        # Engine now uses Dynamic Futility Threshold
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

# --- SEARCH & RESULTS ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    n_list = list(range(n_range[0], n_range[1] + 1, 2))
    with st.spinner("Searching for optimal design..."):
        for n in n_list:
            for hurdle in [0.55, 0.60, 0.65]:
                alpha, _, _, _ = run_fast_batch(2000, n, p0, 0.05, hurdle, eff_conf, safe_limit, n, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta)
                if alpha <= 0.05: # Risk standard check
                    pwr, _, _, _ = run_fast_batch(2000, n, p1, 0.05, hurdle, eff_conf, safe_limit, n, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta)
                    _, tox_p, _, _ = run_fast_batch(2000, n, p1, true_toxic_rate, hurdle, eff_conf, safe_limit, n, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta)
                    results.append({"N": n, "Hurdle": hurdle, "Alpha": alpha, "Power": pwr, "Safety": tox_p})
    
    if results:
        st.session_state['best_design'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['used_params'] = {
            "eff_conf": eff_conf, "saf_conf": safety_conf, "fut_conf": fut_conf,
            "p_a": prior_alpha, "p_b": prior_beta, "s_a": s_prior_alpha, "s_b": s_prior_beta
        }

if 'best_design' in st.session_state:
    best, up = st.session_state['best_design'], st.session_state['used_params']
    
    st.success(f"### ✅ Optimal Design Parameters (Max N = {int(best['N'])})")
    
    # OC STRESS TESTER
    st.markdown("---")
    if st.button("📈 Run OC Stress Test & Prepare Export"):
        scenarios = [
            ("1. Super-Effective", p1+0.1, 0.05), ("2. On-Target", p1, 0.05), ("3. Marginal", (p0+p1)/2, 0.05),
            ("4. Null", p0, 0.05), ("5. Futile", p0-0.1, 0.05), ("6. High Eff/Toxic", p1+0.1, true_toxic_rate),
            ("7. Target Eff/Toxic", p1, true_toxic_rate), ("8. Null/Toxic", p0, true_toxic_rate)
        ]
        stress_data = []
        for name, pe, ps in scenarios:
            pe = np.clip(pe, 0.01, 0.99)
            pw, stp, asn, fut = run_fast_batch(2000, int(best['N']), pe, ps, best['Hurdle'], up['eff_conf'], safe_limit, cohort_size, up['saf_conf'], up['fut_conf'], up['p_a'], up['p_b'], up['s_a'], up['s_b'])
            stress_data.append({"Scenario": name, "Success %": pw, "Safety Stop %": stp, "Futility Stop %": fut, "ASN": asn})
        
        df_oc = pd.DataFrame(stress_data)
        st.table(df_oc)
        
        # UNIFIED EXPORT
        export_df = pd.concat([pd.DataFrame([best]).assign(Type="Design"), df_oc.assign(Type="Scenario")], ignore_index=True)
        st.download_button("📥 Download Unified Report (CSV)", data=export_df.to_csv(index=False).encode('utf-8'), file_name="full_trial_report.csv", mime="text/csv")

        # DYNAMIC INTERPRETATION
        st.info(f"**Interpretation:** This design identifies toxic drugs with **{stress_data[6]['Safety Stop %']:.1%} accuracy**. The prior configuration adds **{up['p_a']+up['p_b']:.1f}** virtual patients to efficacy decisions and **{up['s_a']+up['s_b']:.1f}** to safety.")
