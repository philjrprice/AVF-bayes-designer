import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Specialized Priors")
st.markdown("Updated v20: Final Stability & Range Auto-Correction.")

# --- SIDEBAR: INPUTS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5)
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7)
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Priors")
prior_alpha = st.sidebar.slider("Eff Prior α", 1.0, 10.0, 1.0, step=0.5)
prior_beta = st.sidebar.slider("Eff Prior β", 1.0, 10.0, 1.0, step=0.5)
s_prior_alpha = st.sidebar.slider("Saf Prior α", 1.0, 10.0, 1.0, step=0.5)
s_prior_beta = st.sidebar.slider("Saf Prior β", 1.0, 10.0, 1.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max Alpha", 0.005, 0.20, 0.01, step=0.005)
min_power = st.sidebar.slider("Min Power", 0.70, 0.99, 0.90)
min_safety_power = st.sidebar.slider("Min Safety Detection", 0.70, 0.99, 0.95)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Settings")
min_n_lead = st.sidebar.slider("Min N Lead-in", 5, 50, 20)
cohort_size = st.sidebar.slider("Cohort Size", 1, 20, 5)
n_range = st.sidebar.slider("N Search Range", 20, 200, (60, 120))
eff_conf = st.sidebar.slider("Success Confidence", 0.70, 0.99, 0.85)
safety_conf = st.sidebar.slider("Safety Confidence", 0.50, 0.99, 0.90)
fut_conf = st.sidebar.slider("Futility Threshold", 0.01, 0.20, 0.05)

# --- ENGINE ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, e_conf, limit, cohort_sz, s_conf, f_conf, p_a, p_b, s_a, s_b, min_n):
    np.random.seed(42)
    p_eff, p_sae = np.clip(p_eff, 0.001, 0.999), np.clip(p_sae, 0.001, 0.999)
    
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    stops_n = np.full(sims, max_n)
    is_success, is_safety_stop, is_futility_stop, already_stopped = [np.zeros(sims, dtype=bool) for _ in range(4)]

    # Calculate look points correctly
    look_points = [min_n]
    current_look = min_n + cohort_sz
    while current_look < max_n:
        look_points.append(current_look)
        current_look += cohort_sz

    for n in look_points:
        active = np.where(~already_stopped)[0]
        if len(active) == 0: break
        
        c_s = np.sum(outcomes[active, :n], axis=1)
        c_tox = np.sum(saes[active, :n], axis=1)
        
        prob_eff = 1 - beta.cdf(hurdle, p_a + c_s, p_b + (n - c_s))
        prob_tox = 1 - beta.cdf(limit, s_a + c_tox, s_b + (n - c_tox))
        
        tox_trig = prob_tox > s_conf
        eff_trig = prob_eff > e_conf
        fut_trig = (n >= max_n/2) & (prob_eff < f_conf)
        
        should_stop = tox_trig | eff_trig | fut_trig
        stopping_now = active[should_stop]
        
        is_safety_stop[active[tox_trig]] = True
        is_success[active[eff_trig & ~tox_trig]] = True
        is_futility_stop[active[fut_trig & ~tox_trig & ~eff_trig]] = True
        
        stops_n[stopping_now] = n
        already_stopped[stopping_now] = True

    # Final check
    remaining = np.where(~already_stopped)[0]
    if len(remaining) > 0:
        f_s = np.sum(outcomes[remaining, :max_n], axis=1)
        is_success[remaining] = (1 - beta.cdf(hurdle, p_a + f_s, p_b + (max_n - f_s))) > e_conf
    
    return np.mean(is_success), np.mean(is_safety_stop), np.mean(stops_n), np.mean(is_futility_stop)

# --- SEARCH ---
if st.button("🚀 Find Optimal Sample Size"):
    results = []
    # Auto-Correction: Ensure N starts at or after min_n_lead
    adj_start = max(n_range[0], min_n_lead)
    n_list = range(adj_start, n_range[1] + 1, 2)
    hurdle_options = np.linspace(p0, (p0 + p1)/2, 5)
    
    with st.spinner("Analyzing..."):
        for n in n_list:
            for h in hurdle_options:
                h = round(float(h), 3)
                a, _, _, _ = run_fast_batch(2000, n, p0, 0.05, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta, min_n_lead)
                if a <= max_alpha:
                    p, _, _, _ = run_fast_batch(2000, n, p1, 0.05, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta, min_n_lead)
                    _, s_p, _, _ = run_fast_batch(2000, n, p1, true_toxic_rate, h, eff_conf, safe_limit, cohort_size, safety_conf, fut_conf, prior_alpha, prior_beta, s_prior_alpha, s_prior_beta, min_n_lead)
                    if p >= min_power and s_p >= min_safety_power:
                        results.append({"N": n, "Hurdle": h, "Alpha": a, "Power": p, "Safety": s_p})
                        break
    
    if results:
        st.session_state['best_design'] = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['up'] = {"p_a": prior_alpha, "p_b": prior_beta, "s_a": s_prior_alpha, "s_b": s_prior_beta, "cohort": cohort_size, "min_n": min_n_lead, "p0": p0, "p1": p1, "toxic": true_toxic_rate}
    else:
        st.error("❌ No design found. Suggestions: 1. Increase Max N Range. 2. Decrease Success Confidence. 3. Ensure Lead-in N is not too high.")

# --- DISPLAY RESULTS ---
if 'best_design' in st.session_state:
    b, u = st.session_state['best_design'], st.session_state['up']
    st.success(f"### ✅ Optimal Design Found: N={int(b['N'])} | Hurdle={b['Hurdle']}")
    
    # Restored Multi-Scenario Stress Test
    if st.button("📈 Run Full Stress Test"):
        scenarios = [("1. Super-Eff", u['p1']+0.1, 0.05), ("2. On-Target", u['p1'], 0.05), ("4. Null", u['p0'], 0.05), ("7. Toxic", u['p1'], u['toxic'])]
        stress = []
        for name, pe, ps in scenarios:
            pw, stp, asn, fut = run_fast_batch(2000, int(b['N']), pe, ps, b['Hurdle'], eff_conf, safe_limit, u['cohort'], safety_conf, fut_conf, u['p_a'], u['p_b'], u['s_a'], u['s_b'], u['min_n'])
            stress.append({"Scenario": name, "Success %": f"{pw:.1%}", "Safety Stop %": f"{stp:.1%}", "Avg N": f"{asn:.1f}"})
        st.table(pd.DataFrame(stress))
