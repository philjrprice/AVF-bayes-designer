import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Stress-Tester")
st.markdown("Full Code: Independent Efficacy/Safety Confidence with persistent OC Stress-Testing.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, help="Standard of care success rate.")
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, help="Goal success rate.")
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15, help="Max allowable SAE rate.")
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.25, help="Rate for safety detection testing.")

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.005, 0.20, 0.05, step=0.005)
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.80)
min_safety_power = st.sidebar.slider("Min Safety Power (Detection)", 0.70, 0.99, 0.95)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Prior Strength")
p_alpha = st.sidebar.slider("Prior 'Successes' (α)", 1.0, 10.0, 1.0, step=0.5)
p_beta = st.sidebar.slider("Prior 'Failures' (β)", 1.0, 10.0, 1.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Thresholds")
eff_conf = st.sidebar.slider("Efficacy Success Confidence", 0.70, 0.99, 0.90)
safety_conf = st.sidebar.slider("Safety Stop Confidence", 0.50, 0.99, 0.90)
# NEW: Futility Slider added to sidebar
fut_conf = st.sidebar.slider("Futility Stop Threshold", 0.01, 0.20, 0.05, help="Stop if probability of success < this value.")
cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 6)
n_range = st.sidebar.slider("N Search Range", 40, 150, (60, 100))

# --- STABLE VECTORIZED ENGINE ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, e_conf, limit, cohort_sz, s_conf, f_conf, pa, pb):
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    stops_n = np.full(sims, max_n)
    is_success, is_safety_stop, is_futility_stop, already_stopped = [np.zeros(sims, dtype=bool) for _ in range(4)]

    for n in range(cohort_sz, max_n + 1, cohort_sz):
        active = ~already_stopped
        if not np.any(active): break
        c_s, c_tox = np.sum(outcomes[active, :n], axis=1), np.sum(saes[active, :n], axis=1)
        
        prob_eff = 1 - beta.cdf(hurdle, pa + c_s, pb + (n - c_s))
        prob_tox = 1 - beta.cdf(limit, 1 + c_tox, 1 + (n - c_tox))
        
        tox_trig, eff_trig = prob_tox > s_conf, prob_eff > e_conf
        # ENGINE UPDATE: Uses dynamic futility confidence
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
        is_success[remaining] = (1 - beta.cdf(hurdle, pa + f_s, pb + (max_n - f_s))) > e_conf
    return np.mean(is_success), np.mean(is_safety_stop), np.mean(stops_n), np.mean(is_futility_stop)

def newly_mapped(active, trig):
    m = np.zeros(len(active), dtype=bool); m[active] = trig; return m

# --- CORE APP LOGIC ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    n_list = list(range(n_range[0], n_range[1] + 1, 2))
    with st.spinner("Searching..."):
        for n in n_list:
            for hurdle in [0.55, 0.60, 0.65]:
                alpha, _, _, _ = run_fast_batch(2000, n, p0, 0.05, hurdle, eff_conf, safe_limit, n, safety_conf, fut_conf, p_alpha, p_beta)
                if alpha <= max_alpha:
                    pwr, _, _, _ = run_fast_batch(2000, n, p1, 0.05, hurdle, eff_conf, safe_limit, n, safety_conf, fut_conf, p_alpha, p_beta)
                    _, tox_p, _, _ = run_fast_batch(2000, n, p1, true_toxic_rate, hurdle, eff_conf, safe_limit, n, safety_conf, fut_conf, p_alpha, p_beta)
                    if pwr >= min_power and tox_p >= min_safety_power:
                        results.append({"N": n, "Hurdle": hurdle, "Alpha": alpha, "Power": pwr, "Safety": tox_p})
    
    if results:
        st.session_state['best_design'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['params'] = {"ec": eff_conf, "sc": safety_conf, "fc": fut_conf, "pa": p_alpha, "pb": p_beta}

if 'best_design' in st.session_state:
    best, p = st.session_state['best_design'], st.session_state['params']
    
    st.success(f"### ✅ Optimal Design Parameters (Max N = {int(best['N'])})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Max Enrollment", int(best['N']))
    m2.metric("Efficacy Power", f"{best['Power']:.1%}")
    m3.metric("Safety Detection", f"{best['Safety']:.1%}")
    m4.metric("Risk (Alpha)", f"{best['Alpha']:.2%}")

    # PROTOCOL SUMMARY BOX (Matches Screenshot)
    with st.expander("📝 Protocol Summary: Bayesian Monitoring Rules", expanded=True):
        st.write(f"1. **Interim Analysis Schedule**: Data monitored in cohorts of {cohort_size} patients.")
        st.write(f"2. **Efficacy Success Rule**: Declare success if $P(Response Rate > {best['Hurdle']}) > {p['ec']}$.")
        st.write(f"3. **Safety Stopping Rule**: Stop for toxicity if $P(SAE Rate > {safe_limit}) > {p['sc']}$.")
        st.write(f"4. **Futility Rule**: At patient {int(best['N']/2)}, stop if the probability of success is $< {p['fc']:.0%}$.")

    # OC STRESS TESTER
    st.markdown("---")
    st.subheader("📊 Operational Characteristics (OC) Stress-Tester")
    if st.button("📈 Run Multi-Scenario Stress Test"):
        scenarios = [
            ("1. Super-Effective", p1+0.1, 0.05), ("2. On-Target", p1, 0.05), ("3. Marginal", (p0+p1)/2, 0.05),
            ("4. Null", p0, 0.05), ("5. Futile", p0-0.1, 0.05), ("6. High Eff / Toxic", p1+0.1, true_toxic_rate),
            ("7. Target Eff / Toxic", p1, true_toxic_rate), ("8. Null / Toxic", p0, true_toxic_rate)
        ]
        stress_data = []
        for name, pe, ps in scenarios:
            pe = np.clip(pe, 0.01, 0.99)
            pw, stp, asn, fut = run_fast_batch(3000, int(best['N']), pe, ps, best['Hurdle'], p['ec'], safe_limit, cohort_size, p['sc'], p['fc'], p['pa'], p['pb'])
            stress_data.append({"Scenario": name, "Success %": f"{pw:.1%}", "Safety Stop %": f"{stp:.1%}", "Futility Stop %": f"{fut:.1%}", "Avg N (ASN)": f"{asn:.1f}"})
        
        df_oc = pd.DataFrame(stress_data)
        st.table(df_oc)

        # UNIFIED EXPORT
        export_df = pd.concat([pd.DataFrame([best]).assign(Source="Design"), df_oc.assign(Source="OC_Table")], ignore_index=True)
        st.download_button("📥 Download Full Report (CSV)", data=export_df.to_csv(index=False).encode('utf-8'), file_name="trial_design_report.csv", mime="text/csv")

    # PRIOR VISUALIZATION (Placed at the bottom for clean layout)
    st.markdown("---")
    st.subheader("📈 Bayesian Prior Probability Densities")
    x = np.linspace(0, 1, 100)
    y = beta.pdf(x, p['pa'], p['pb'])
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(x, y, color='blue', lw=2, label=f'Current Prior: Beta({p["pa"]}, {p["pb"]})')
    ax.fill_between(x, 0, y, color='blue', alpha=0.1)
    ax.axvline(p0, color='red', linestyle='--', label=f'Null Hurdle ({p0})')
    ax.set_title("Efficacy Prior Distribution")
    ax.legend()
    st.pyplot(fig)
