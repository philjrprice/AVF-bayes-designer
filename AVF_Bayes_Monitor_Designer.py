import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Prior Sensitivity")
st.markdown("Full 8-Scenario Suite with High-Precision 2-Sig-Fig Formatting.")

# --- SIDEBAR: DESIGN GOALS & PRIORS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, help="Standard of Care rate.")
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, help="Goal rate.")
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Prior Strength")
prior_alpha = st.sidebar.slider("Prior 'Successes' (α)", 1.0, 10.0, 1.0, step=0.5, help="Beta α parameter.")
prior_beta = st.sidebar.slider("Prior 'Failures' (β)", 1.0, 10.0, 1.0, step=0.5, help="Beta β parameter.")

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk & Adaptive Controls")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.005, 0.20, 0.01, step=0.005)
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.90)
eff_conf = st.sidebar.slider("Efficacy Success Confidence", 0.70, 0.99, 0.85)
safety_conf = st.sidebar.slider("Safety Stop Confidence", 0.50, 0.99, 0.90)
cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5)
n_range = st.sidebar.slider("N Search Range", 40, 150, (60, 100))

# --- VECTORIZED ENGINE ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, e_conf, limit, cohort_sz, s_conf, p_a, p_b):
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    stops_n = np.full(sims, max_n)
    is_success, is_safety_stop, is_futility_stop, already_stopped = [np.zeros(sims, dtype=bool) for _ in range(4)]

    for n in range(cohort_sz, max_n + 1, cohort_sz):
        active = ~already_stopped
        if not np.any(active): break
        c_s, c_tox = np.sum(outcomes[active, :n], axis=1), np.sum(saes[active, :n], axis=1)
        
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

# --- SEARCH ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    n_list = list(range(n_range[0], n_range[1] + 1, 2))
    with st.spinner("Optimizing Design..."):
        for n in n_list:
            for hurdle in [0.55, 0.60, 0.65]:
                alpha, _, _, _ = run_fast_batch(2000, n, p0, 0.05, hurdle, eff_conf, safe_limit, n, safety_conf, prior_alpha, prior_beta)
                if alpha <= max_alpha:
                    pwr, _, _, _ = run_fast_batch(2000, n, p1, 0.05, hurdle, eff_conf, safe_limit, n, safety_conf, prior_alpha, prior_beta)
                    _, tox_p, _, _ = run_fast_batch(2000, n, p1, true_toxic_rate, hurdle, eff_conf, safe_limit, n, safety_conf, prior_alpha, prior_beta)
                    if pwr >= min_power:
                        results.append({"N": n, "Hurdle": hurdle, "Alpha": alpha, "Power": pwr, "Safety": tox_p})
    
    if results:
        st.session_state['best_design'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['used_params'] = {"e_conf": eff_conf, "s_conf": safety_conf, "p_a": prior_alpha, "p_b": prior_beta}
    else:
        st.error("No design found. Relax constraints.")

# --- RESULTS & COMPREHENSIVE OC ---
if 'best_design' in st.session_state:
    best = st.session_state['best_design']
    up = st.session_state['used_params']
    
    st.success(f"### ✅ Optimal Design Parameters (Max N = {int(best['N'])})")
    cols = st.columns(4)
    cols[0].metric("Max Enrollment", int(best['N']))
    cols[1].metric("Power", f"{best['Power']:.2g}")
    cols[2].metric("Alpha", f"{best['Alpha']:.2g}")
    cols[3].metric("Hurdle", f"{best['Hurdle']:.2g}")

    st.markdown("---")
    st.subheader("📊 Comprehensive OC Stress-Tester (8 Scenarios)")
    if st.button("📈 Run Full Simulation Suite"):
        scenarios = [
            ("1. Super-Effective (p1 + 10%)", p1 + 0.10, 0.05),
            ("2. Target Reached (p1)", p1, 0.05),
            ("3. Marginal (Midpoint)", (p0+p1)/2, 0.05),
            ("4. Null (Standard Care p0)", p0, 0.05),
            ("5. Futile (p0 - 10%)", p0 - 0.10, 0.05),
            ("6. High Eff / High Tox", p1 + 0.10, true_toxic_rate),
            ("7. Target Eff / High Tox", p1, true_toxic_rate),
            ("8. Null Eff / High Tox", p0, true_toxic_rate),
        ]
        
        stress_results = []
        with st.spinner("Running 16,000 trial simulations..."):
            for name, pe, ps in scenarios:
                pe, ps = np.clip(pe, 0.01, 0.99), np.clip(ps, 0.01, 0.99)
                suc, saf, asn, fut = run_fast_batch(2000, int(best['N']), pe, ps, best['Hurdle'], up['e_conf'], safe_limit, cohort_size, up['s_conf'], up['p_a'], up['p_b'])
                stress_results.append({
                    "Scenario": name,
                    "Success %": f"{suc*100:.1f}%",
                    "Safety Stop %": f"{saf*100:.1f}%",
                    "Futility Stop %": f"{fut*100:.1f}%",
                    "Avg N (ASN)": f"{asn:.2g}"
                })
        
        df_final = pd.DataFrame(stress_results)
        st.table(df_final)

        # --- COMPREHENSIVE INTERPRETATION ---
        st.info("### 🧐 Comprehensive Output Interpretation")
        
        # Ethical Efficiency
        eff_asn = float(stress_results[0]["Avg N (ASN)"])
        savings = (1 - (eff_asn / best['N'])) * 100
        
        # Safety Integrity
        tox_capture = float(stress_results[6]["Safety Stop %"].strip('%'))
        
        st.markdown(f"""
        * **Early Graduation**: In the 'Super-Effective' scenario, the trial stops at **{eff_asn:.2g}** patients on average, a **{savings:.1f}% reduction** in enrollment compared to a fixed design.
        * **Safety Guardrail**: The monitor identifies toxic drugs (Scenario 7) with **{tox_capture:.1f}% accuracy**.
        * **Futility Performance**: In the 'Futile' scenario, the trial successfully shuts down **{stress_results[4]['Futility Stop %']}** of the time, preventing unnecessary exposure when efficacy is lacking.
        * **Prior Sensitivity**: The $Beta({up['p_a']}, {up['p_b']})$ prior is currently contributing **{up['p_a'] + up['p_b']:.2g} virtual patients** of weight to the initial decision boundary.
        """)
