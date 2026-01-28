import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd

st.set_page_config(page_title="Master Bayesian Trial Lab", layout="wide")

st.title("🔬 Master Bayesian Trial Lab: Efficacy & Safety")
st.markdown("This tool simulates dual-track monitoring: continuous safety checks and periodic efficacy looks.")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("1. Clinical Assumptions")
    p0 = st.slider("Null Response Rate (p0)", 0.1, 0.9, 0.50)
    p1 = st.slider("Target Efficacy (p1)", 0.1, 0.9, 0.70)
    tox_limit = st.slider("Max Tolerable Toxicity Rate", 0.1, 0.5, 0.33)
    
    st.header("2. Trial Structure")
    max_n = st.slider("Max Sample Size (N)", 20, 150, 80)
    start_at = st.slider("First Efficacy Look at n=", 5, 50, 20)
    look_every = st.slider("Efficacy Look Interval", 1, 10, 5)
    
    st.header("3. Bayesian Thresholds")
    eff_interim = st.slider("Interim Success Threshold", 0.95, 0.999, 0.99, format="%.3f")
    eff_final = st.slider("FINAL Success Threshold", 0.90, 0.99, 0.95, format="%.3f")
    fut_thresh = st.slider("Futility Threshold", 0.01, 0.20, 0.10)
    tox_thresh = st.slider("Safety Threshold (Prob > Max Tox)", 0.80, 0.99, 0.95)

# --- SIMULATION ENGINE ---
@st.cache_data
def run_master_simulation(p0, p1, tox_limit, max_n, start_at, step, e_int, e_fin, f_t, t_t):
    interims = list(range(start_at, max_n, step))
    if max_n not in interims: interims.append(max_n)
    
    # 1. Pre-calculate Master Table Boundaries
    table_data = []
    for n in range(1, max_n + 1):
        # Toxicity Bound (Calculated for every n)
        tox_bound = -1
        for x in range(n + 1):
            if (1 - stats.beta.cdf(tox_limit, 0.5 + x, 0.5 + n - x)) >= t_t:
                tox_bound = x
                break
        
        # Efficacy Bound (Only for interim n)
        f_lim, e_lim = None, None
        if n in interims:
            f_lim, e_lim = -1, n + 1
            thresh = e_fin if n == max_n else e_int
            for x in range(n + 1):
                prob = 1 - stats.beta.cdf(p0, 0.5 + x, 0.5 + n - x)
                if prob < f_t: f_lim = x
                if prob >= thresh and e_lim == n + 1: e_lim = x
        
        if n in interims or n < 10 or n % 10 == 0:
            table_data.append({
                "n": n, 
                "Stop for Safety (Tox ≥)": tox_bound if tox_bound != -1 else "—",
                "Futility (Resp ≤)": f_lim if f_lim is not None else "—",
                "Success (Resp ≥)": e_lim if e_lim is not None else "—"
            })

    # 2. Monte Carlo Simulation (Efficacy + Safety)
    def sim(p_eff, p_tox):
        resp, tox, curr_n = 0, 0, 0
        # Check every patient for safety, but check efficacy at steps
        for n_step in range(1, max_n + 1):
            resp += np.random.binomial(1, p_eff)
            tox += np.random.binomial(1, p_tox)
            curr_n = n_step
            
            # Continuous Safety Check
            prob_tox = 1 - stats.beta.cdf(tox_limit, 0.5 + tox, 0.5 + curr_n - tox)
            if prob_tox >= t_t: return "Safety Stop", curr_n
            
            # Periodic Efficacy Check
            if curr_n in interims:
                prob_eff = 1 - stats.beta.cdf(p0, 0.5 + resp, 0.5 + curr_n - resp)
                thresh = e_fin if curr_n == max_n else e_int
                if prob_eff < f_t: return "Futility Stop", curr_n
                if prob_eff >= thresh: return "Success Stop", curr_n
        
        return "Futility Stop", max_n

    n_sims = 4000
    res_null = [sim(p0, 0.20) for _ in range(n_sims)] # Assume 20% baseline tox
    res_alt = [sim(p1, 0.20) for _ in range(n_sims)]
    
    return {
        "alpha": np.mean([1 if r[0] == "Success Stop" else 0 for r in res_null]),
        "power": np.mean([1 if r[0] == "Success Stop" else 0 for r in res_alt]),
        "safety_stops": np.mean([1 if r[0] == "Safety Stop" else 0 for r in res_alt]),
        "asn": np.mean([r[1] for r in res_alt]),
        "table": pd.DataFrame(table_data)
    }

# --- UI DISPLAY ---
results = run_master_simulation(p0, p1, tox_limit, max_n, start_at, look_every, eff_interim, eff_final, fut_thresh, tox_thresh)

st.subheader("📊 Integrated Performance Readout")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall Alpha", f"{results['alpha']:.2%}")
c2.metric("Total Power", f"{results['power']:.2%}")
c3.metric("Prob of Safety Stop", f"{results['safety_stops']:.2%}", help="Chance trial stops for toxicity if true rate is 20%")
c4.metric("Avg Sample Size", f"{results['asn']:.1f}")

st.divider()
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Master Operational Table")
    st.dataframe(results['table'], use_container_width=True, hide_index=True)

with col_right:
    st.subheader("Final Protocol Specs")
    st.write(f"**Efficacy:** Pass if ≥ {results['table'].iloc[-1]['Success (Resp ≥)']} responders.")
    st.write(f"**Safety:** Suspend trial if Toxicity count hits the safety limit at any point.")
    st.write(f"**Efficiency:** The design expects to finish after ~{results['asn']:.0f} patients if the drug is effective.")
    
    st.download_button("Export Table to CSV", results['table'].to_csv(index=False), "master_trial_design.csv")
