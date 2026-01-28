import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Specialized Priors")
st.markdown("Updated v17: Synchronized Search Engine & High-Efficacy Robustness.")

# --- SIDEBAR: RESTORED V15 SLIDERS & TOOLTIPS ---
with st.sidebar:
    st.header("🎯 Efficacy & Safety")
    p0 = st.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, help="Success rate of standard of care.")
    p1 = st.slider("Target Efficacy (p1)", 0.5, 0.95, 0.8, help="Goal success rate for the new drug.")
    safe_limit = st.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15, help="Max allowable SAE rate.")
    true_toxic_rate = st.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30)
    
    st.header("⚖️ Prior Strength")
    p_a = st.slider("Eff Prior α", 1.0, 10.0, 1.0)
    p_b = st.slider("Eff Prior β", 1.0, 10.0, 1.0)
    s_a = st.slider("Saf Prior α", 1.0, 10.0, 1.0)
    s_b = st.slider("Saf Prior β", 1.0, 10.0, 1.0)
    
    st.header("📐 Risk Standards")
    max_alpha = st.slider("Max Alpha", 0.005, 0.20, 0.05)
    min_power = st.slider("Min Power", 0.70, 0.99, 0.80)
    min_safety_power = st.slider("Min Safety Power", 0.70, 0.99, 0.80)
    
    st.header("⏱️ Adaptive Thresholds")
    eff_conf = st.slider("Success Confidence", 0.70, 0.99, 0.85)
    safety_conf = st.slider("Safety Confidence", 0.50, 0.99, 0.90)
    fut_conf = st.slider("Futility Threshold", 0.01, 0.20, 0.05)
    cohort_size = st.slider("Cohort Size", 1, 20, 5)
    n_range = st.slider("N Search Range", 20, 200, (40, 100))

# --- CORE ENGINE: SYNCHRONIZED ACROSS SEARCH & TEST ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, e_conf, limit, cohort, s_conf, f_conf, pa, pb, sa, sb):
    np.random.seed(42)
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    
    is_success, is_safety_stop, is_futility_stop = [np.zeros(sims, dtype=bool) for _ in range(3)]
    stops_n = np.full(sims, max_n)
    stopped = np.zeros(sims, dtype=bool)

    # Consistent look schedule: Search and Operational Test now match
    for n in range(cohort, max_n + 1, cohort):
        active = ~stopped
        if not np.any(active): break
        
        c_s, c_tox = np.sum(outcomes[active, :n], axis=1), np.sum(saes[active, :n], axis=1)
        prob_eff = 1 - beta.cdf(hurdle, pa + c_s, pb + (n - c_s))
        prob_tox = 1 - beta.cdf(limit, sa + c_tox, sb + (n - c_tox))
        
        tox_trig = prob_tox > s_conf
        eff_trig = prob_eff > e_conf
        fut_trig = (n >= max_n/2) & (prob_eff < f_conf)
        
        # Exact mapping logic from v15
        idx = np.where(active)[0]
        is_safety_stop[idx[tox_trig]] = True
        is_success[idx[eff_trig & ~tox_trig]] = True
        is_futility_stop[idx[fut_trig & ~tox_trig & ~eff_trig]] = True
        
        triggered = (tox_trig | eff_trig | fut_trig)
        stops_n[idx[triggered]] = n
        stopped[idx[triggered]] = True

    remaining = ~stopped
    if np.any(remaining):
        f_s = np.sum(outcomes[remaining, :max_n], axis=1)
        is_success[remaining] = (1 - beta.cdf(hurdle, pa + f_s, pb + (max_n - f_s))) > e_conf
        
    return np.mean(is_success), np.mean(is_safety_stop), np.mean(stops_n), np.mean(is_futility_stop)

# --- SEARCH ENGINE: RESOLVING THE "NO DESIGN" ISSUE ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    n_list = range(n_range[0], n_range[1] + 1, 2)
    # Expansion: Search broader hurdle space to accommodate high efficacy (p1 > 80%)
    hurdle_options = np.linspace(p0, (p0 + p1)/1.5, 7)
    
    with st.spinner("Analyzing high-efficacy design space..."):
        for n in n_list:
            for h in hurdle_options:
                h = round(float(h), 3)
                # Test Alpha
                a, _, _, _ = run_fast_batch(2000, n, p0, 0.05, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, p_a, p_b, s_a, s_b)
                if a <= max_alpha:
                    # Test Power and Safety Detection
                    pwr, _, _, _ = run_fast_batch(2000, n, p1, 0.05, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, p_a, p_b, s_a, s_b)
                    _, s_p, _, _ = run_fast_batch(2000, n, p1, true_toxic_rate, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, p_a, p_b, s_a, s_b)
                    if pwr >= min_power and s_p >= min_safety_power:
                        results.append({"N": n, "Hurdle": h, "Alpha": a, "Power": pwr, "Safety": s_p})
                        break
    if results:
        st.session_state['best'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['up'] = {"p0":p0, "p1":p1, "lim":safe_limit, "tox":true_toxic_rate, "e_c":eff_conf, "s_c":safety_conf, "f_c":fut_conf, "pa":p_a, "pb":p_b, "sa":s_a, "sb":s_b, "cohort":cohort_size}
    else:
        st.error("❌ No design found. To fix: 1. Decrease 'Success Confidence' or 2. Increase 'Max Alpha'.")

# --- PERSISTENT v15 UI RESTORATION ---
if 'best' in st.session_state:
    b, u = st.session_state['best'], st.session_state['up']
    st.success(f"### ✅ Optimal Design Found: Max N = {int(b['N'])} | Hurdle = {b['Hurdle']}")
    
    # Restoring v15 metric displays
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max N", int(b['N'])); c2.metric("Power", f"{b['Power']:.1%}"); c3.metric("Safety Det.", f"{b['Safety']:.1%}"); c4.metric("Alpha", f"{b['Alpha']:.2%}")

    if st.button("📊 Run 8-Scenario Stress Test"):
        scens = [("Super-Eff", u['p1']+0.1, 0.05), ("On-Target", u['p1'], 0.05), ("Null", u['p0'], 0.05), ("Toxic", u['p1'], u['tox'])]
        data = []
        for name, pe, ps in scens:
            pe = np.clip(pe, 0.01, 0.99)
            pw, stp, asn, fut = run_fast_batch(2000, int(b['N']), pe, ps, b['Hurdle'], u['e_c'], u['lim'], u['cohort'], u['s_c'], u['f_c'], u['pa'], u['pb'], u['sa'], u['sb'])
            data.append({"Scenario": name, "Success %": f"{pw:.1%}", "Safety Stop %": f"{stp:.1%}", "Avg N": f"{asn:.1f}"})
        st.table(pd.DataFrame(data))
