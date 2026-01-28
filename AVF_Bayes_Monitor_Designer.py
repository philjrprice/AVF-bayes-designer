import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Specialized Priors")
st.markdown("Updated v19: Re-implemented v15 functionality with Robust Lead-in Logic.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, help="Success rate of standard of care.")
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, help="Goal success rate for the new drug.")
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15, help="Max allowable SAE rate.")
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30, help="Rate used for safety power testing.")

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Efficacy Prior Strength")
prior_alpha = st.sidebar.slider("Eff Prior α", 1.0, 10.0, 1.0, step=0.5)
prior_beta = st.sidebar.slider("Eff Prior β", 1.0, 10.0, 1.0, step=0.5)

st.sidebar.header("🛡️ Safety Prior Strength")
s_prior_alpha = st.sidebar.slider("Saf Prior α", 1.0, 10.0, 1.0, step=0.5)
s_prior_beta = st.sidebar.slider("Saf Prior β", 1.0, 10.0, 1.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max Alpha", 0.005, 0.20, 0.01, step=0.005)
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.90)
min_safety_power = st.sidebar.slider("Min Safety Detection", 0.70, 0.99, 0.95)

st.sidebar.markdown("---")
st.sidebar.header("🔬 Simulation Rigor")
n_sims = st.sidebar.select_slider("Simulations", options=[2000, 5000, 10000, 15000], value=2000)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Thresholds")
min_n_lead = st.sidebar.slider("Min N Before First Check", 5, 50, 20)
eff_conf = st.sidebar.slider("Efficacy Success Confidence", 0.70, 0.99, 0.85)
safety_conf = st.sidebar.slider("Safety Stop Confidence", 0.50, 0.99, 0.90)
fut_conf = st.sidebar.slider("Futility Stop Threshold", 0.01, 0.20, 0.05)
cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5)
n_range = st.sidebar.slider("N Search Range", 40, 150, (60, 100))

# --- ROBUST VECTORIZED ENGINE ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, e_conf, limit, cohort_sz, s_conf, f_conf, p_a, p_b, s_a, s_b, min_n):
    np.random.seed(42)
    p_eff, p_sae = np.clip(p_eff, 0.001, 0.999), np.clip(p_sae, 0.001, 0.999)
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    stops_n = np.full(sims, max_n)
    is_success, is_safety_stop, is_futility_stop, already_stopped = [np.zeros(sims, dtype=bool) for _ in range(4)]

    # Dynamic Analysis Schedule
    look_points = sorted(list(set([min_n] + [n for n in range(min_n, max_n + 1, cohort_sz) if n <= max_n])))

    for n in look_points:
        active = ~already_stopped
        if not np.any(active): break
        c_s, c_tox = np.sum(outcomes[active, :n], axis=1), np.sum(saes[active, :n], axis=1)
        prob_eff = 1 - beta.cdf(hurdle, p_a + c_s, p_b + (n - c_s))
        prob_tox = 1 - beta.cdf(limit, s_a + c_tox, s_b + (n - c_tox))
        
        tox_trig, eff_trig = prob_tox > s_conf, prob_eff > e_conf
        fut_trig = (n >= max_n/2) & (prob_eff < f_conf)
        
        idx_tox = active.copy(); idx_tox[active] = tox_trig
        idx_eff = active.copy(); idx_eff[active] = eff_trig & ~tox_trig
        idx_fut = active.copy(); idx_fut[active] = fut_trig & ~tox_trig & ~eff_trig
        
        is_safety_stop[idx_tox & ~already_stopped] = True
        is_success[idx_eff & ~already_stopped] = True
        is_futility_stop[idx_fut & ~already_stopped] = True
        stops_n[(idx_tox | idx_eff | idx_fut) & ~already_stopped] = n
        already_stopped[idx_tox | idx_eff | idx_fut] = True

    remaining = ~already_stopped
    if np.any(remaining):
        f_s = np.sum(outcomes[remaining, :max_n], axis=1)
        is_success[remaining] = (1 - beta.cdf(hurdle, p_a + f_s, p_b + (max_n - f_s))) > e_conf
    return np.mean(is_success), np.mean(is_safety_stop), np.mean(stops_n), np.mean(is_futility_stop)

# --- SEARCH ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    n_list = range(max(n_range[0], min_n_lead), n_range[1] + 1, 2)
    hurdle_options = np.linspace(p0, (p0 + p1)/2, 5)
    
    with st.spinner(f"Searching using {n_sims:,} simulations..."):
        for n in n_list:
            for hurdle in hurdle_options:
                hurdle = round(float(hurdle), 3)
                alpha, _, _, _ = run_fast_batch(n_sims, n, p0, 0.05, hurdle, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta, min_n_lead)
                if alpha <= max_alpha:
                    pwr, _, _, _ = run_fast_batch(n_sims, n, p1, 0.05, hurdle, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta, min_n_lead)
                    _, tox_p, _, _ = run_fast_batch(n_sims, n, p1, true_toxic_rate, hurdle, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta, min_n_lead)
                    if pwr >= min_power and tox_p >= min_safety_power:
                        results.append({"N": n, "Hurdle": hurdle, "Alpha": alpha, "Power": pwr, "Safety": tox_p})
                        break
    if results:
        st.session_state['best'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['params'] = {"p0": p0, "p1": p1, "limit": safe_limit, "toxic": true_toxic_rate, "e_c": eff_conf, "s_c": safety_conf, "f_c": fut_conf, "p_a": prior_alpha, "p_b": prior_beta, "s_a": s_prior_alpha, "s_b": s_prior_beta, "cohort": cohort_size, "min_n": min_n_lead, "sims": n_sims}
    else:
        st.error("❌ No design found. Loosen constraints or increase N Range.")

# --- RESULTS DISPLAY (RE-IMPLEMENTED FROM V15) ---
if 'best' in st.session_state:
    b, u = st.session_state['best'], st.session_state['params']
    st.success(f"### ✅ Optimal Design Found: Max N={int(b['N'])} | Hurdle={b['Hurdle']}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Enrollment", int(b['N']))
    c2.metric("Efficacy Power", f"{b['Power']:.1%}")
    c3.metric("Safety Detection", f"{b['Safety']:.1%}")
    c4.metric("Risk (Alpha)", f"{b['Alpha']:.2%}")

    with st.expander("📊 Multi-Scenario Stress Test", expanded=True):
        if st.button("📈 Run Full 8-Scenario Suite"):
            scenarios = [
                ("1. Super-Effective", u['p1'] + 0.1, 0.05), ("2. On-Target", u['p1'], 0.05),
                ("3. Marginal", (u['p0'] + u['p1'])/2, 0.05), ("4. Null (Alpha)", u['p0'], 0.05),
                ("5. Futile", u['p0'] - 0.1, 0.05), ("6. High Eff / Toxic", u['p1'] + 0.1, u['toxic']),
                ("7. Target Eff / Toxic", u['p1'], u['toxic']), ("8. Null / Toxic", u['p0'], u['toxic'])
            ]
            stress_results = []
            for name, pe, ps in scenarios:
                pw, stp, asn, fut = run_fast_batch(u['sims'], int(b['N']), pe, ps, b['Hurdle'], u['e_c'], u['limit'], u['cohort'], u['s_c'], u['f_c'], u['p_a'], u['p_b'], u['s_a'], u['s_b'], u['min_n'])
                stress_results.append({"Scenario": name, "Success %": pw, "Safety Stop %": stp, "Futility %": fut, "ASN": asn})
            st.session_state['stress'] = pd.DataFrame(stress_results)
            st.table(st.session_state['stress'].style.format({"Success %": "{:.1%}", "Safety Stop %": "{:.1%}", "Futility %": "{:.1%}", "ASN": "{:.1f}"}))

    # OC Curve Restoration
    st.markdown("---")
    st.subheader("📈 Operating Characteristic (OC) Curve")
    eff_range = np.linspace(max(0, u['p0'] - 0.1), min(1, u['p1'] + 0.1), 10)
    oc_pos = [run_fast_batch(u['sims'], int(b['N']), pe, 0.05, b['Hurdle'], u['e_c'], u['limit'], u['cohort'], u['s_c'], u['f_c'], u['p_a'], u['p_b'], u['s_a'], u['s_b'], u['min_n'])[0] for pe in eff_range]
    fig_oc, ax_oc = plt.subplots(figsize=(10, 4))
    ax_oc.plot(eff_range, oc_pos, marker='o', color='teal', label='PoS'); ax_oc.axvline(u['p0'], color='red', linestyle='--'); ax_oc.legend(); st.pyplot(fig_oc)

    # Export Restoration
    if 'stress' in st.session_state:
        csv = pd.concat([pd.DataFrame([u]).T, pd.DataFrame([b]).T, st.session_state['stress']]).to_csv().encode('utf-8')
        st.download_button("📥 Export Full Design Report (CSV)", csv, "AVF_Final_Report.csv", "text/csv")
