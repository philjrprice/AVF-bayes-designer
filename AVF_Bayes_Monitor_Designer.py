import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd

# --- SIDEBAR UPDATES ---
st.sidebar.header("⏱️ Adaptive Settings")
cohort_size = st.sidebar.slider("Interim Cohort Size", 1, 20, 5, 
                                help="Frequency of safety/efficacy checks (e.g., check every 5 patients).")

# --- UPDATED SIMULATION ENGINE ---
def run_adaptive_trial(max_n, p_eff, p_sae, eff_hurdle, eff_conf, safe_limit, cohort_size, safe_conf=0.90):
    """Simulates a single trial with interim looks."""
    # We check at every 'cohort_size' interval
    interim_steps = range(cohort_size, max_n + 1, cohort_size)
    
    # Generate all potential outcomes upfront for this one trial
    all_outcomes = np.random.binomial(1, p_eff, max_n)
    all_saes = np.random.binomial(1, p_sae, max_n)
    
    for n in interim_steps:
        current_successes = np.sum(all_outcomes[:n])
        current_saes = np.sum(all_saes[:n])
        
        # Bayesian Checks
        prob_eff = 1 - beta.cdf(eff_hurdle, 1 + current_successes, 1 + (n - current_successes))
        prob_toxic = 1 - beta.cdf(safe_limit, 1 + current_saes, 1 + (n - current_saes))
        
        # 1. Check Safety First
        if prob_toxic > safe_conf:
            return n, "Safety Stop", False
        
        # 2. Check Efficacy (Success)
        if prob_eff > eff_conf:
            return n, "Success", True
            
        # 3. Check Futility (Optional: lower bound check)
        if n > (max_n / 2) and prob_eff < 0.05:
            return n, "Futility", False
            
    # If we reach max_n without a stop
    return max_n, "Final Analysis", (prob_eff > eff_conf)

def run_stress_test_with_asn(n_target, eff_hurdle, eff_conf, safe_limit, p1, p_toxic, cohort_size):
    scenarios = [
        ("1. High Eff / Safe", p1 + 0.1, 0.05),
        ("2. On-Target / Safe", p1, 0.05),
        ("10. Futile (Null)", 0.50, 0.05),
        ("11. High Eff / Toxic", p1 + 0.1, p_toxic),
        ("12. Target Eff / Toxic", p1, p_toxic),
    ]
    
    table_data = []
    sims = 1000 # Reduced sims for speed in loop
    for name, pe, ps in scenarios:
        results = [run_adaptive_trial(n_target, pe, ps, eff_hurdle, eff_conf, safe_limit, cohort_size) for _ in range(sims)]
        
        stops_n = [r[0] for r in results]
        success_flags = [r[2] for r in results]
        safety_stops = [1 for r in results if r[1] == "Safety Stop"]
        
        table_data.append({
            "Scenario Name": name,
            "Success %": f"{(sum(success_flags)/sims)*100:.1f}%",
            "Safety Stop %": f"{(len(safety_stops)/sims)*100:.1f}%",
            "Avg N at Stop (ASN)": f"{np.mean(stops_n):.1f}"
        })
    return pd.DataFrame(table_data)

# --- DISPLAY LOGIC (Inside the main 'if results' block) ---
# When displaying the table:
# st.table(run_stress_test_with_asn(best['N'], best['Hurdle'], best['Conf'], safe_limit, p1, true_toxic_rate, cohort_size))
