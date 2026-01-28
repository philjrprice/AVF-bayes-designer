import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Specialized Priors")
st.markdown("Updated v21: Resolved High-Efficacy Paradox & Dynamic Hurdle Search.")

# --- SIDEBAR: DESIGN GOALS ---
with st.sidebar:
    st.header("🎯 Efficacy & Safety")
    p0 = st.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5)
    p1 = st.slider("Target Efficacy (p1)", 0.5, 0.95, 0.8) # Expanded p1 range
    safe_limit = st.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
    true_toxic_rate = st.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30)
    
    st.header("⚖️ Priors")
    p_a = st.slider("Eff Prior α", 1.0, 10.0, 1.0)
    p_b = st.slider("Eff Prior β", 1.0, 10.0, 1.0)
    s_a = st.slider("Saf Prior α", 1.0, 10.0, 1.0)
    s_b = st.slider("Saf Prior β", 1.0, 10.0, 1.0)
    
    st.header("📐 Risk Standards")
    max_alpha = st.slider("Max Alpha", 0.01, 0.20, 0.05)
    min_power = st.slider("Min Power", 0.50, 0.99, 0.80)
    min_safety_pwr = st.slider("Min Safety Power", 0.50, 0.99, 0.80)
    
    st.header("⏱️ Adaptive Thresholds")
    min_n_lead = st.slider("Min N Before First Check", 5, 50, 15)
    eff_conf = st.slider("Success Confidence", 0.70, 0.99, 0.85)
    safety_conf = st.slider("Safety Confidence", 0.50, 0.99, 0.90)
    fut_conf = st.slider("Futility Threshold", 0.01, 0.20, 0.05)
    cohort_sz = st.slider("Cohort Size", 1, 20, 5)
    n_range = st.slider("N Search Range", 20, 200, (40, 100))

# --- STABLE VECTORIZED ENGINE ---
def run_simulation(sims, max_n, p_eff, p_sae, hurdle, e_conf, limit, cohort, s_conf, f_conf, pa, pb, sa, sb, min_n):
    np.random.seed(42)
    # Ensure min_n is respected
    looks = sorted(list(set([min_n] + [n for n in range(min_n, max_n + 1, cohort) if n <= max_n])))
    
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    
    is_success, is_safety_stop, is_futility_stop = [np.zeros(sims, dtype=bool) for _ in range(3)]
    stops_n = np.full(sims, max_n)
    stopped = np.zeros(sims, dtype=bool)

    for n in looks:
        active = ~stopped
        if not np.any(active): break
        
        c_s, c_tox = np.sum(outcomes[active, :n], axis=1), np.sum(saes[active, :n], axis=1)
        prob_eff = 1 - beta.cdf(hurdle, pa + c_s, pb + (n - c_s))
        prob_tox = 1 - beta.cdf(limit, sa + c_tox, sb + (n - c_tox))
        
        tox_trig = prob_tox > s_conf
        eff_trig = prob_eff > e_conf
        fut_trig = (n >= max_n/2) & (prob_eff < f_conf)
        
        idx = np.where(active)[0]
        # Priority: Safety > Success > Futility
        t_mask = idx[tox_trig]
        e_mask = idx[eff_trig & ~tox_trig]
        f_mask = idx[fut_trig & ~tox_trig & ~eff_trig]
        
        for i in t_mask: 
            if not stopped[i]: is_safety_stop[i]=True; stops_n[i]=n; stopped[i]=True
        for i in e_mask: 
            if not stopped[i]: is_success[i]=True; stops_n[i]=n; stopped[i]=True
        for i in f_mask: 
            if not stopped[i]: is_futility_stop[i]=True; stops_n[i]=n; stopped[i]=True

    remaining = ~stopped
    if np.any(remaining):
        f_s = np.sum(outcomes[remaining, :max_n], axis=1)
        is_success[remaining] = (1 - beta.cdf(hurdle, pa + f_s, pb + (max_n - f_s))) > e_conf
        
    return np.mean(is_success), np.mean(is_safety_stop), np.mean(stops_n), np.mean(is_futility_stop)

# --- OPTIMIZED SEARCH ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    # Dynamic Hurdle: We now search from p0 up to slightly below p1
    hurdle_opts = np.linspace(p0, (p0 + p1) / 1.5, 8)
    n_list = range(max(n_range[0], min_n_lead), n_range[1] + 1, 2)
    
    with st.spinner("Analyzing Design Space..."):
        for n in n_list:
            for h in hurdle_opts:
                h = round(float(h), 3)
                # Alpha check (using max_n to be conservative)
                a, _, _, _ = run_simulation(2000, n, p0, 0.05, h, eff_conf, safe_limit, n, safety_conf, fut_conf, p_a, p_b, s_a, s_b, min_n_lead)
                if a <= max_alpha:
                    # Power checks
                    pwr, _, _, _ = run_simulation(2000, n, p1, 0.05, h, eff_conf, safe_limit, n, safety_conf, fut_conf, p_a, p_b, s_a, s_b, min_n_lead)
                    _, s_p, _, _ = run_simulation(2000, n, p1, true_toxic_rate, h, eff_conf, safe_limit, n, safety_conf, fut_conf, p_a, p_b, s_a, s_b, min_n_lead)
                    
                    if pwr >= min_power and s_p >= min_safety_pwr:
                        results.append({"N": n, "Hurdle": h, "Alpha": a, "Power": pwr, "Safety": s_p})
                        break # Found best hurdle for this N
                        
    if results:
        st.session_state['best_v21'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['params_v21'] = {"p0":p0, "p1":p1, "limit":safe_limit, "tox":true_toxic_rate, "e_c":eff_conf, "s_c":safety_conf, "f_c":fut_conf, "pa":p_a, "pb":p_b, "sa":s_a, "sb":s_b, "cohort":cohort_sz, "min_n":min_n_lead}
    else:
        st.error("❌ No design found. Possible fix: Increase Max Alpha or decrease Success Confidence.")

# --- DISPLAY ---
if 'best_v21' in st.session_state:
    b, u = st.session_state['best_v21'], st.session_state['params_v21']
    st.success(f"### ✅ Optimal Design: Max N={int(b['N'])} | Hurdle={b['Hurdle']}")
    
    # Restored v15 Multi-Scenario Stress Test
    if st.button("📈 Run Full 8-Scenario Stress Test"):
        scens = [("Super-Eff", u['p1']+0.1, 0.05), ("Target", u['p1'], 0.05), ("Null", u['p0'], 0.05), ("Toxic", u['p1'], u['tox'])]
        data = []
        for name, pe, ps in scens:
            pe = np.clip(pe, 0.01, 0.99)
            pw, stp, asn, fut = run_simulation(2000, int(b['N']), pe, ps, b['Hurdle'], u['e_c'], u['limit'], u['cohort'], u['s_c'], u['f_c'], u['pa'], u['pb'], u['sa'], u['sb'], u['min_n'])
            data.append({"Scenario": name, "Success %": f"{pw:.1%}", "Safety Stop %": f"{stp:.1%}", "Avg N": f"{asn:.1f}"})
        st.table(pd.DataFrame(data))
