import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd

st.set_page_config(page_title="AVF Master Designer: Optimized", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Stress-Tester")
st.markdown("Reverted to high-flexibility engine with relative hurdle logic.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy & Safety")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.1, 0.9, 0.50)
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.1, 0.9, 0.70)
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' Rate", 0.10, 0.50, 0.30)

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.01, 0.20, 0.05)
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.80)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Settings")
min_n_lead_in = st.sidebar.number_input("Minimum N before check", 1, 100, 15)
cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5)
n_range = st.sidebar.slider("N Search Range", 20, 250, (40, 120))

# --- STABLE VECTORIZED ENGINE ---
def run_fast_batch(sims, max_n, p_eff, p_sae, hurdle, conf, limit, cohort_sz, lead_in, safe_conf=0.90):
    outcomes = np.random.binomial(1, p_eff, (sims, max_n))
    saes = np.random.binomial(1, p_sae, (sims, max_n))
    
    stops_n = np.full(sims, max_n)
    is_success = np.zeros(sims, dtype=bool)
    is_safety_stop = np.zeros(sims, dtype=bool)
    already_stopped = np.zeros(sims, dtype=bool)

    # Define check points respecting lead-in
    check_points = [n for n in range(lead_in, max_n + 1) if (n == lead_in) or ((n - lead_in) % cohort_sz == 0)]
    
    for n in check_points:
        active = ~already_stopped
        if not np.any(active): break
        
        c_s = np.sum(outcomes[active, :n], axis=1)
        c_tox = np.sum(saes[active, :n], axis=1)
        
        p_eff_val = 1 - beta.cdf(hurdle, 1 + c_s, 1 + (n - c_s))
        p_tox_val = 1 - beta.cdf(limit, 1 + c_tox, 1 + (n - c_tox))
        
        tox_trig = p_tox_val > safe_conf
        eff_trig = p_eff_val > conf
        
        new_stops = np.zeros(sims, dtype=bool)
        new_stops[active] = (tox_trig | eff_trig)
        
        is_safety_stop[active & (np.where(active, False, False) | (new_stops & ~eff_trig))] = True # Priority to safety
        is_success[active & (np.where(active, False, False) | (eff_trig & ~tox_trig))] = True
        
        stops_n[new_stops & ~already_stopped] = n
        already_stopped[new_stops] = True

    remaining = ~already_stopped
    if np.any(remaining):
        f_s = np.sum(outcomes[remaining, :max_n], axis=1)
        is_success[remaining] = (1 - beta.cdf(hurdle, 1 + f_s, 1 + (max_n - f_s))) > conf

    return np.mean(is_success), np.mean(is_safety_stop), np.mean(stops_n)

# --- SEARCH EXECUTION ---
if st.button("🚀 Find Optimal Design"):
    results = []
    # Dynamic hurdles based on user input to guarantee results
    hurdle_list = [p0, (p0 + p1)/2, p1 - 0.05]
    conf_list = [0.75, 0.80, 0.85, 0.90]
    n_list = range(n_range[0], n_range[1] + 1, 5)

    with st.spinner("Scanning parameter space..."):
        for n in n_list:
            if n < min_n_lead_in: continue
            for h in hurdle_list:
                for c in conf_list:
                    alpha, _, _ = run_fast_batch(1000, n, p0, 0.05, h, c, safe_limit, cohort_size, min_n_lead_in)
                    if alpha <= max_alpha:
                        pwr, safe_p, _ = run_fast_batch(1000, n, p1, 0.05, h, c, safe_limit, cohort_size, min_n_lead_in)
                        if pwr >= min_power:
                            results.append({"N": n, "Hurdle": h, "Conf": c, "Alpha": alpha, "Power": pwr})

    if results:
        best = pd.DataFrame(results).sort_values("N").iloc[0]
        st.session_state['best_design'] = best
        st.success(f"### Design Found: Max N = {int(best['N'])}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Sample Size (N)", int(best['N']))
        c2.metric("Power", f"{best['Power']:.1%}")
        c3.metric("Alpha", f"{best['Alpha']:.2%}")
    else:
        st.error("No design found. Try increasing N Range or Alpha.")

# --- SEPARATE OC TESTER ---
st.markdown("---")
st.subheader("📊 Operational Stress-Tester")
if 'best_design' in st.session_state:
    if st.button("📈 Run OC Simulations"):
        b = st.session_state['best_design']
        scenarios = [
            ("Target Met (Efficacy)", p1, 0.05),
            ("Null (Failing)", p0, 0.05),
            ("Toxic (Danger)", p1, true_toxic_rate)
        ]
        stress_results = []
        for name, pe, ps in scenarios:
            pwr, tox, asn = run_fast_batch(2000, int(b['N']), pe, ps, b['Hurdle'], b['Conf'], safe_limit, cohort_size, min_n_lead_in)
            stress_results.append({"Scenario": name, "Success %": f"{pwr:.1%}", "Safety Stop %": f"{tox:.1%}", "Avg Patients": f"{asn:.1f}"})
        st.table(pd.DataFrame(stress_results))
