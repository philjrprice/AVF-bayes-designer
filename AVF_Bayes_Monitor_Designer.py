import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AVF Master Designer: Adaptive Suite", layout="wide")

st.title("🧬 Master Designer: Adaptive OC & Stress-Tester")
st.markdown("This tool optimizes for the best $N$ and then stress-tests the design using **Adaptive Interim Looks** to calculate the Average Sample Size (Avg N).")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy Objectives")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5)
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Safety Objectives")
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30)

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.005, 0.20, 0.01, step=0.005)
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.90)
min_safety_power = st.sidebar.slider("Min Safety Power (Detection)", 0.70, 0.99, 0.95)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Settings")
cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5, help="Checks for early stopping every X patients.")
n_range = st.sidebar.slider("N Search Range", 20, 150, (40, 100))

# --- CORE ADAPTIVE SIMULATION ENGINE ---
def run_adaptive_trial(max_n, p_eff, p_sae, eff_hurdle, eff_conf, safe_limit, cohort_size, safe_conf=0.90):
    """Simulates a single trial with interim looks and returns (N_at_stop, Outcome_String, Is_Success_Flag)."""
    # Create the data stream for one trial
    outcomes = np.random.binomial(1, p_eff, max_n)
    saes = np.random.binomial(1, p_sae, max_n)
    
    # Define interim look points
    interim_points = list(range(cohort_size, max_n, cohort_size)) + [max_n]
    
    for n in interim_points:
        curr_succ = np.sum(outcomes[:n])
        curr_saes = np.sum(saes[:n])
        
        # Bayesian Posteriors
        prob_eff = 1 - beta.cdf(eff_hurdle, 1 + curr_succ, 1 + (n - curr_succ))
        prob_toxic = 1 - beta.cdf(safe_limit, 1 + curr_saes, 1 + (n - curr_saes))
        
        # 1. Safety Check (Primary)
        if prob_toxic > safe_conf:
            return n, "Safety Stop", False
        
        # 2. Success Check (Efficacy)
        if prob_eff > eff_conf:
            return n, "Success", True
            
        # 3. Futility Check (Optional: If halfway through and prob < 5%)
        if n >= max_n/2 and prob_eff < 0.05:
            return n, "Futility Stop", False
            
    return max_n, "Final Analysis", (prob_eff > eff_conf)

def run_simulation_batch(sims, max_n, p_eff, p_sae, eff_hurdle, eff_conf, safe_limit, cohort_size):
    """Runs a batch of trials to get averages and probabilities."""
    results = [run_adaptive_trial(max_n, p_eff, p_sae, eff_hurdle, eff_conf, safe_limit, cohort_size) for _ in range(sims)]
    
    n_stops = [r[0] for r in results]
    successes = [r[2] for r in results]
    safety_stops = [1 for r in results if r[1] == "Safety Stop"]
    
    return np.mean(successes), np.mean(safety_stops), np.mean(n_stops)

# --- EXECUTION ---
if st.button("🚀 Run Adaptive Design & Stress-Test"):
    results = []
    with st.spinner("Searching for optimal Adaptive N..."):
        # Grid search for the best fixed parameters first
        for n in range(n_range[0], n_range[1] + 1, 2):
            for hurdle in [0.55, 0.60, 0.65]:
                for conf in [0.70, 0.74, 0.80, 0.85]:
                    # Standard check (no early stopping for optimization speed)
                    alpha, _ = run_simulation_batch(1000, n, p0, 0.05, hurdle, conf, safe_limit, n) # Fixed look at N
                    if alpha <= max_alpha:
                        power, _ , _ = run_simulation_batch(1000, n, p1, 0.05, hurdle, conf, safe_limit, n)
                        _, tox_stop, _ = run_simulation_batch(1000, n, p1, true_toxic_rate, hurdle, conf, safe_limit, n)
                        
                        if power >= min_power and tox_stop >= min_safety_power:
                            results.append({"N": n, "Hurdle": hurdle, "Conf": conf, "Alpha": alpha, "Power": power, "Safety_Power": tox_stop})

    if results:
        df = pd.DataFrame(results)
        best = df.sort_values("N").iloc[0]
        
        st.success(f"### ✅ Optimal Adaptive Design Found: Max N = {int(best['N'])}")
        
        # Display Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max Enrollment (N)", int(best['N']))
        c2.metric("Efficacy Power", f"{best['Power']:.1%}")
        c3.metric("Safety Stop Prob.", f"{best['Safety_Power']:.1%}")
        c4.metric("Risk (Alpha)", f"{best['Alpha']:.2%}")

        # --- FULL STRESS TEST TABLE ---
        st.markdown("---")
        st.subheader("📋 Adaptive Operational Stress Test (OC Table)")
        st.info(f"Checking every **{cohort_size}** patients for early stopping.")
        
        scenarios = [
            ("1. High Eff / Safe", p1 + 0.1, 0.05),
            ("2. On-Target / Safe", p1, 0.05),
            ("10. Futile (Null)", p0, 0.05),
            ("11. High Eff / Toxic", p1 + 0.1, true_toxic_rate),
            ("12. Target Eff / Toxic", p1, true_toxic_rate),
            ("13. Low Eff / Toxic", p1 - 0.1, true_toxic_rate),
        ]
        
        stress_data = []
        for name, pe, ps in scenarios:
            pow_val, stop_val, asn_val = run_simulation_batch(2000, best['N'], pe, ps, best['Hurdle'], best['Conf'], safe_limit, cohort_size)
            stress_data.append({
                "Scenario Name": name,
                "Success %": f"{pow_val*100:.1f}%",
                "Safety Stop %": f"{stop_val*100:.1f}%",
                "Avg N at Stop (ASN)": f"{asn_val:.1f}",
                "True Efficacy": f"{pe*100:.0f}%",
                "True SAE Rate": f"{ps*100:.0f}%"
            })
        
        st.table(pd.DataFrame(stress_data))
        
        st.warning(f"💡 **Observation:** Notice how the **Avg N at Stop** drops significantly in Toxic scenarios (11-13). This is the 'Adaptive Benefit' of checking every {cohort_size} patients.")
    else:
        st.error("No design found meeting these strict criteria. Try increasing the N range or slightly relaxing the Alpha/Power targets.")
    
