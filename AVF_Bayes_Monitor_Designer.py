import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AVF Trial Designer & OC Tool", layout="wide")

st.title("🧬 Antivenom Trial Designer: OC & Sample Size Calculator")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy Objectives")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5)
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Safety Objectives")
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.25)

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk & Standards")
max_alpha = st.sidebar.slider("Max False Positive Rate (Alpha)", 0.01, 0.20, 0.10)
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.95, 0.85)
# NEW: Safety Power Adjustment
min_safety_power = st.sidebar.slider("Min Safety Power (Detection)", 0.70, 0.99, 0.90, 
                                     help="90% is the standard for high-risk trials.")

n_range = st.sidebar.slider("Sample Size Search Range", 20, 150, (40, 100))

# --- SIMULATION ENGINE ---
def run_trial_sims(n, p_eff, p_sae, eff_hurdle, eff_conf, safe_limit, safe_conf=0.90, sims=5000):
    successes = np.random.binomial(n, p_eff, sims)
    sae_counts = np.random.binomial(n, p_sae, sims)
    
    # Efficacy: P(rate > hurdle) > conf
    prob_eff = 1 - beta.cdf(eff_hurdle, 1 + successes, 1 + (n - successes))
    is_success = prob_eff > eff_conf
    
    # Safety: P(rate > limit) > 0.90
    prob_toxic = 1 - beta.cdf(safe_limit, 1 + sae_counts, 1 + (n - sae_counts))
    is_safety_stop = prob_toxic > safe_conf
    
    return np.mean(is_success), np.mean(is_safety_stop)

if st.button("🚀 Optimize Design"):
    results = []
    with st.spinner("Searching for design meeting all standards..."):
        for n in range(n_range[0], n_range[1] + 1, 2):
            for hurdle in [0.55, 0.60, 0.65]:
                for conf in [0.70, 0.74, 0.80]:
                    # Check Alpha (False Positive)
                    alpha, _ = run_trial_sims(n, p0, 0.05, hurdle, conf, safe_limit)
                    
                    if alpha <= max_alpha:
                        # Check Efficacy Power
                        power, _ = run_trial_sims(n, p1, 0.05, hurdle, conf, safe_limit)
                        # Check Safety Power (Detection of toxic rate)
                        _, tox_stop_prob = run_trial_sims(n, p1, true_toxic_rate, hurdle, conf, safe_limit)
                        
                        # FILTER: Only keep designs meeting both Efficacy and Safety Power targets
                        if power >= min_power and tox_stop_prob >= min_safety_power:
                            results.append({
                                "N": n, "Hurdle": hurdle, "Conf": conf,
                                "Alpha": alpha, "Power": power, "Safety_Power": tox_stop_prob
                            })

    if results:
        df = pd.DataFrame(results)
        best = df.sort_values("N").iloc[0]
        
        st.success(f"### ✅ Optimal Design Found: N = {int(best['N'])}")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recommended N", int(best['N']))
        m2.metric("Efficacy Power", f"{best['Power']:.1%}")
        m3.metric("Safety Detection", f"{best['Safety_Power']:.1%}")
        m4.metric("Risk (Alpha)", f"{best['Alpha']:.1%}")

def run_scenario_matrix(n, eff_hurdle, eff_conf, safe_limit, p1, p_toxic):
    # Define the 13 scenarios based on user inputs
    scenarios = [
        ("High Eff / Safe", p1 + 0.1, 0.05),
        ("On-Target / Safe", p1, 0.05),
        ("Low Eff / Safe", p1 - 0.1, 0.05),
        ("High Eff / BL Safe", p1 + 0.1, 0.12),
        ("On-Target / BL Safe", p1, 0.12),
        ("Low Eff / BL Safe", p1 - 0.1, 0.12),
        ("High Eff / BL Unsafe", p1 + 0.1, 0.18),
        ("On-Target / BL Unsafe", p1, 0.18),
        ("Low Eff / BL Unsafe", p1 - 0.1, 0.18),
        ("Futile (Null)", 0.50, 0.05),
        ("High Eff / Toxic", p1 + 0.1, p_toxic),
        ("Target Eff / Toxic", p1, p_toxic),
        ("Low Eff / Toxic", p1 - 0.1, p_toxic),
    ]
    
    table_data = []
    for name, p_eff, p_sae in scenarios:
        # Simulate 2000 trials for this specific scenario
        sims = 2000
        successes = np.random.binomial(n, p_eff, sims)
        saes = np.random.binomial(n, p_sae, sims)
        
        # Calculate Bayesian Outcomes
        prob_eff = 1 - beta.cdf(eff_hurdle, 1 + successes, 1 + (n - successes))
        prob_safe = 1 - beta.cdf(safe_limit, 1 + saes, 1 + (n - saes))
        
        # Determine Status
        is_success = prob_eff > eff_conf
        is_safety_stop = prob_safe > 0.90
        is_futility = (prob_eff < 0.10) # Example futility rule
        
        table_data.append({
            "Scenario": name,
            "Success %": f"{(np.mean(is_success) * 100):.1f}%",
            "Safety Stop %": f"{(np.mean(is_safety_stop) * 100):.1f}%",
            "Futility %": f"{(np.mean(is_futility) * 100):.1f}%"
        })
    return pd.DataFrame(table_data)
      
        # Validation Label
        if best['Safety_Power'] >= 0.90:
            st.info("⭐ **Meets High-Safety Standard:** This design provides >90% protection against toxic outcomes.")
        
        # Visualization
        fig = px.scatter(df, x="Alpha", y="Power", color="N", size="Safety_Power",
                         title="Search Space: Efficacy Power vs Risk (Bubble Size = Safety Detection)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No design found. Try increasing the 'Sample Size Search Range' or lowering the 'Min Safety Power'.")

if results:
    # ... (after finding the 'best' design)
    st.subheader("📋 Full Operational Stress Test (OC Table)")
    oc_table = run_scenario_matrix(best['N'], best['Hurdle'], best['Conf'], safe_limit, p1, true_toxic_rate)
    st.table(oc_table)
