import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd

st.set_page_config(page_title="Bayesian Trial Lab", layout="wide")

st.title("🔬 Bayesian Trial Lab: Calibrated Monitor")
st.markdown("This tool simulates thousands of trials to account for 'peeking' and provide accurate Alpha/Power stats.")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("1. Clinical Assumptions")
    p0 = st.slider("Null Rate (p0)", 0.1, 0.9, 0.50)
    p1 = st.slider("Target Efficacy (p1)", 0.1, 0.9, 0.70)
    
    st.header("2. Trial Structure")
    max_n = st.slider("Max Sample Size (N)", 20, 150, 80)
    start_at = st.slider("First Look at Patient #", 5, 50, 20)
    look_every = st.slider("Look Every 'X' Patients", 1, 10, 5)
    
    st.header("3. Bayesian Thresholds")
    eff_interim = st.slider("Interim Success Threshold", 0.95, 0.999, 0.99, format="%.3f")
    eff_final = st.slider("FINAL Success Threshold", 0.90, 0.99, 0.95, format="%.3f")
    fut_thresh = st.slider("Futility Threshold", 0.01, 0.20, 0.10)

# --- SIMULATION ENGINE ---
@st.cache_data # Cache results to make sliders feel smooth
def run_simulation(p0, p1, max_n, start_at, step, e_int, e_fin, f_t):
    interims = list(range(start_at, max_n, step))
    if max_n not in interims: interims.append(max_n)
    
    # 1. Pre-calc Boundaries
    bounds = []
    for n in interims:
        f_lim, e_lim = -1, n + 1
        thresh = e_fin if n == max_n else e_int
        for x in range(n + 1):
            prob = 1 - stats.beta.cdf(p0, 0.5 + x, 0.5 + n - x)
            if prob < f_t: f_lim = x
            if prob >= thresh and e_lim == n + 1: e_lim = x
        bounds.append({'n': n, 'f': f_lim, 'e': e_lim})

    # 2. Monte Carlo (5000 iterations for speed/accuracy balance)
    def sim(p_true):
        hits, curr_n = 0, 0
        for b in bounds:
            hits += np.random.binomial(b['n'] - curr_n, p_true)
            curr_n = b['n']
            if hits <= b['f']: return 0, curr_n # Futility stop
            if hits >= b['e']: return 1, curr_n # Success stop
        return 0, curr_n

    n_sims = 5000
    res_null = [sim(p0) for _ in range(n_sims)]
    res_alt = [sim(p1) for _ in range(n_sims)]
    
    return {
        "alpha": np.mean([r[0] for r in res_null]),
        "power": np.mean([r[0] for r in res_alt]),
        "asn_null": np.mean([r[1] for r in res_null]),
        "asn_alt": np.mean([r[1] for r in res_alt]),
        "table": pd.DataFrame(bounds).rename(columns={'n':'Patients (n)', 'f':'Futility (Resp ≤)', 'e':'Success (Resp ≥)'})
    }

# --- RUN & DISPLAY ---
results = run_simulation(p0, p1, max_n, start_at, look_every, eff_interim, eff_final, fut_thresh)

# Stats Readout Row
st.subheader("📊 Statistical Performance Readout")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall Alpha", f"{results['alpha']:.2%}", help="Cumulative False Positive rate accounting for all interim looks.")
c2.metric("Total Power", f"{results['power']:.2%}", help="Probability of success if the drug is truly p1 effective.")
c3.metric("Avg N (if Null)", f"{results['asn_null']:.1f}")
c4.metric("Avg N (if Effective)", f"{results['asn_alt']:.1f}")

# Layout for Table
st.divider()
col_tab, col_info = st.columns([3, 2])

with col_tab:
    st.subheader("Operational Decision Table")
    st.table(results['table'])

with col_info:
    st.subheader("Design Insights")
    st.write(f"**Success at Final Look:** You need {results['table'].iloc[-1]['Success (Resp ≥)']} responders out of {max_n} patients.")
    st.write(f"**Efficiency:** If the drug works, you will likely conclude the trial after **{results['asn_alt']:.0f}** patients.")
    
    if results['alpha'] > 0.06:
        st.warning("⚠️ Alpha is high. Try increasing the 'Interim Success Threshold' or 'Final Success Threshold'.")
    elif results['power'] < 0.80:
        st.error("❌ Power is low. Consider increasing 'Max Sample Size (N)' or lowering the 'Final Success Threshold'.")
    else:
        st.success("✅ This design meets standard regulatory power/alpha levels.")

st.download_button("Download Protocol Table", results['table'].to_csv(index=False), "trial_design.csv", "text/csv")
