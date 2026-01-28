import streamlit as st
import numpy as np
import scipy.stats as stats
import pandas as pd

# Page Config
st.set_page_config(page_title="Bayesian Trial Designer", layout="wide")

st.title("🔬 Bayesian Single-Arm Monitor Designer")
st.markdown("""
This tool designs a Bayesian single-arm trial with continuous monitoring for efficacy and futility.
It uses a Beta-Binomial conjugate prior model.
""")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Trial Parameters")
    p0 = st.number_input("Null Response Rate (p0)", value=0.50, step=0.05)
    p1 = st.number_input("Expected Efficacy (p1)", value=0.70, step=0.05)
    
    st.divider()
    max_n = st.slider("Max Sample Size (N)", 20, 150, 80)
    look_every = st.slider("Monitor Every 'X' Patients", 1, 10, 5)
    start_at = st.slider("Start Monitoring at Patient #", 5, 30, 20)
    
    st.divider()
    eff_thresh = st.slider("Success Threshold (Prob > p0)", 0.90, 0.999, 0.99, format="%.3f")
    fut_thresh = st.slider("Futility Threshold (Prob > p0)", 0.01, 0.20, 0.10)

# --- Core Functions ---
def get_boundaries(p0, max_n, start_at, step, eff_t, fut_t):
    # ... previous code ...
    for n in interims:
        fut_limit = -1
        eff_limit = n + 1
        for x in range(n + 1):
            post_prob = 1 - stats.beta.cdf(p0, a_prior + x, b_prior + n - x)
            if post_prob < fut_t:
                fut_limit = x
            # For the final patient, we can be slightly less strict (e.g., 0.95) 
            # to maintain power, while interims stay at 0.99.
            current_thresh = eff_t if n < max_n else 0.95 
            if post_prob > current_thresh and eff_limit == n + 1:
                eff_limit = x
    
    boundaries = []
    # Use Jeffreys Prior (0.5, 0.5)
    a_prior, b_prior = 0.5, 0.5
    
    for n in interims:
        fut_limit = -1
        eff_limit = n + 1
        for x in range(n + 1):
            # Posterior Probability: P(theta > p0 | x, n)
            post_prob = 1 - stats.beta.cdf(p0, a_prior + x, b_prior + n - x)
            if post_prob < fut_t:
                fut_limit = x
            if post_prob > eff_t and eff_limit == n + 1:
                eff_limit = x
        boundaries.append({'n': n, 'Futility_If_Responders_<=': fut_limit, 'Success_If_Responders_>=': eff_limit})
    return pd.DataFrame(boundaries)

def simulate_trials(p_true, b_df, p0, n_sims=5000):
    success_count = 0
    total_n = 0
    a_p, b_p = 0.5, 0.5
    
    for _ in range(n_sims):
        curr_n = 0
        hits = 0
        stopped = False
        for _, row in b_df.iterrows():
            n_new = row['n'] - curr_n
            hits += np.random.binomial(n_new, p_true)
            curr_n = row['n']
            
            if hits <= row['Futility_If_Responders_<=']:
                stopped = True; break
            if hits >= row['Success_If_Responders_>=']:
                success_count += 1
                stopped = True; break
        
        if not stopped: # Final look check
            prob = 1 - stats.beta.cdf(p0, a_p + hits, b_p + curr_n - hits)
            if prob >= 0.99: # Match the eff_thresh logic
                success_count += 1
        total_n += curr_n
        
    return (success_count / n_sims), (total_n / n_sims)

# --- Logic Execution ---
df_bounds = get_boundaries(p0, max_n, start_at, look_every, eff_thresh, fut_thresh)

# Simulation
with st.spinner("Running Simulations..."):
    alpha, asn_null = simulate_trials(p0, df_bounds, p0)
    power, asn_alt = simulate_trials(p1, df_bounds, p0)

# --- UI Layout ---
col1, col2, col3 = st.columns(3)
col1.metric("Alpha (Type I Error)", f"{alpha:.2%}")
col2.metric("Power (at {0:.0%})".format(p1), f"{power:.2%}")
col3.metric("Avg Sample Size (if effective)", f"{asn_alt:.1f}")

st.subheader("Operational Decision Table")
st.dataframe(df_bounds, use_container_width=True)

st.info(f"**Final Analysis Rule:** At N={max_n}, you need {df_bounds.iloc[-1]['Success_If_Responders_>=']} or more responders to declare success.")
