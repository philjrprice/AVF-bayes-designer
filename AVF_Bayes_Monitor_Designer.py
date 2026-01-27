import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AVF Trial Designer & OC Tool", layout="wide")

st.title("🧬 Antivenom Trial Designer: OC & Sample Size Calculator")
st.markdown("""
This tool uses Monte Carlo simulations to calculate the **Operating Characteristics (OC)** of your trial. 
It optimizes for the smallest N that meets your Power and Safety requirements.
""")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy Objectives")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, help="Efficacy of a 'failure' drug.")
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, help="Efficacy of your 'dream' drug.")

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Safety Objectives")
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
true_safe_rate = st.sidebar.slider("Assumed 'Safe' SAE Rate", 0.01, 0.15, 0.05)
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.25)

st.sidebar.markdown("---")
st.sidebar.header("📐 Constraints & Risks")
max_alpha = st.sidebar.slider("Max False Positive Rate (Alpha)", 0.01, 0.20, 0.10)
min_power = st.sidebar.slider("Min Statistical Power", 0.70, 0.95, 0.85)
n_range = st.sidebar.slider("Sample Size Search Range", 20, 150, (40, 100))

# --- SIMULATION ENGINE ---
def run_trial_sims(n, p_eff, p_sae, eff_hurdle, eff_conf, safe_limit, safe_conf, sims=5000):
    """Simulates 'sims' number of trials to check for success or safety stops."""
    # Simulate Successes and SAEs
    successes = np.random.binomial(n, p_eff, sims)
    sae_counts = np.random.binomial(n, p_sae, sims)
    
    # Bayesian Efficacy Check
    prob_eff = 1 - beta.cdf(eff_hurdle, 1 + successes, 1 + (n - successes))
    is_success = prob_eff > eff_conf
    
    # Bayesian Safety Check
    prob_safe = 1 - beta.cdf(safe_limit, 1 + sae_counts, 1 + (n - sae_counts))
    is_safety_stop = prob_safe > safe_conf
    
    return np.mean(is_success), np.mean(is_safety_stop)

if st.button("🚀 Optimize Design & Calculate Safety OCs"):
    results = []
    
    with st.spinner("Simulating 10,000+ trial iterations..."):
        # Grid Search across N, Hurdles, and Confidence levels
        for n in range(n_range[0], n_range[1] + 1, 5):
            for hurdle in [0.55, 0.60, 0.65]:
                for conf in [0.70, 0.74, 0.80]:
                    # 1. SCENARIO: DRUG IS A FAILURE (p0) BUT SAFE
                    alpha, _ = run_trial_sims(n, p0, true_safe_rate, hurdle, conf, safe_limit, 0.90)
                    
                    if alpha <= max_alpha:
                        # 2. SCENARIO: DRUG IS A SUCCESS (p1) AND SAFE
                        power, _ = run_trial_sims(n, p1, true_safe_rate, hurdle, conf, safe_limit, 0.90)
                        
                        # 3. SCENARIO: DRUG IS TOXIC
                        _, tox_stop_prob = run_trial_sims(n, p1, true_toxic_rate, hurdle, conf, safe_limit, 0.90)
                        
                        results.append({
                            "N": n, "Hurdle": hurdle, "Conf": conf,
                            "Alpha": alpha, "Power": power, "Safety_Power": tox_stop_prob
                        })

    if results:
        df = pd.DataFrame(results)
        valid = df[df['Power'] >= min_power].sort_values("N")
        
        if not valid.empty:
            best = valid.iloc[0]
            st.success(f"### Optimal Design Found: N = {int(best['N'])}")
            
            # --- TOP METRICS ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Recommended N", int(best['N']))
            c2.metric("Efficacy Power", f"{best['Power']:.1%}")
            c3.metric("False Positive (Alpha)", f"{best['Alpha']:.1%}")
            c4.metric("Safety Stop Prob.", f"{best['Safety_Power']:.1%}")

            # --- DETAILED ANALYSIS ---
            st.markdown("---")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("📊 Efficacy Profile")
                st.write(f"**Hurdle:** {best['Hurdle']:.0%} | **Confidence Req:** {best['Conf']:.0%}")
                st.info(f"If
