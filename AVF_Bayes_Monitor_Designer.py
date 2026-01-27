import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AVF Master Designer & Stress-Tester", layout="wide")

st.title("🧬 Master Designer: OC Simulation & Stress-Tester")
st.markdown("This tool optimizes trial parameters and then stress-tests the design across 13 clinical scenarios.")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Efficacy Objectives")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, help="Standard of care efficacy.")
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, help="Your 'On-Target' goal.")

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Safety Objectives")
safe_limit = st.sidebar.slider("SAE Upper Limit (%)", 0.05, 0.30, 0.15)
true_toxic_rate = st.sidebar.slider("Assumed 'Toxic' SAE Rate", 0.10, 0.50, 0.30)

st.sidebar.markdown("---")
st.sidebar.header("📐 Risk Standards")
max_alpha = st.sidebar.slider("Max False Positive (Alpha)", 0.01, 0.20, 0.10)
min_power = st.sidebar.slider("Min Efficacy Power", 0.70, 0.99, 0.85)
min_safety_power = st.sidebar.slider("Min Safety Power (Detection)", 0.70, 0.99, 0.95)

n_range = st.sidebar.slider("N Search Range", 20, 150, (40, 100))

# --- CORE SIMULATION ENGINE ---
def run_trial_sims(n, p_eff, p_sae, eff_hurdle, eff_conf, safe_limit, safe_conf=0.90, sims=2000):
    successes = np.random.binomial(n, p_eff, sims)
    saes = np.random.binomial(n, p_sae, sims)
    
    # Efficacy Check
    prob_eff = 1 - beta.cdf(eff_hurdle, 1 + successes, 1 + (n - successes))
    is_success = prob_eff > eff_conf
    
    # Safety Check
    prob_toxic = 1 - beta.cdf(safe_limit, 1 + saes, 1 + (n - saes))
    is_safety_stop = prob_toxic > safe_conf
    
    return np.mean(is_success), np.mean(is_safety_stop)

def run_full_stress_test(n, eff_hurdle, eff_conf, safe_limit, p1, p_toxic):
    scenarios = [
        ("1. High Eff / Safe", p1 + 0.1, 0.05),
        ("2. On-Target / Safe", p1, 0.05),
        ("3. Low Eff / Safe", p1 - 0.1, 0.05),
        ("4. High Eff / BL Safe", p1 + 0.1, 0.12),
        ("5. On-Target / BL Safe", p1, 0.12),
        ("6. Low Eff / BL Safe", p1 - 0.1, 0.12),
        ("7. High Eff / BL Unsafe", p1 + 0.1, 0.18),
        ("8. On-Target / BL Unsafe", p1, 0.18),
        ("9. Low Eff / BL Unsafe", p1 - 0.1, 0.18),
        ("10. Futile (Null)", 0.50, 0.05),
        ("11. High Eff / Toxic", p1 + 0.1, p_toxic),
        ("12. Target Eff / Toxic", p1, p_toxic),
        ("13. Low Eff / Toxic", p1 - 0.1, p_toxic),
    ]
    
    table_data = []
    for name, pe, ps in scenarios:
        suc, saf = run_trial_sims(n, pe, ps, eff_hurdle, eff_conf, safe_limit)
        table_data.append({
            "Scenario Name": name,
            "Success %": f"{suc*100:.1f}%",
            "Safety Stop %": f"{saf*100:.1f}%",
            "True Efficacy": f"{pe*100:.0f}%",
            "True SAE Rate": f"{ps*100:.0f}%"
        })
    return pd.DataFrame(table_data)

# --- EXECUTION ---
if st.button("🚀 Run Master Design & Stress-Test"):
    results = []
    with st.spinner("Searching for optimal design..."):
        for n in range(n_range[0], n_range[1] + 1, 2):
            for hurdle in [0.55, 0.60, 0.65]:
                for conf in [0.70, 0.74, 0.80]:
                    alpha, _ = run_trial_sims(n, p0, 0.05, hurdle, conf, safe_limit)
                    if alpha <= max_alpha:
                        power, _ = run_trial_sims(n, p1, 0.05, hurdle, conf, safe_limit)
                        _, tox_stop = run_trial_sims(n, p1, true_toxic_rate, hurdle, conf, safe_limit)
                        
                        if power >= min_power and tox_stop >= min_safety_power:
                            results.append({"N": n, "Hurdle": hurdle, "Conf": conf, "Alpha": alpha, "Power": power, "Safety_Power": tox_stop})

    if results:
        df = pd.DataFrame(results)
        best = df.sort_values("N").iloc[0]
        
        st.success(f"### ✅ Optimal Design Found: N = {int(best['N'])}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recommended N", int(best['N']))
        c2.metric("Efficacy Power", f"{best['Power']:.1%}")
        c3.metric("Safety Detection", f"{best['Safety_Power']:.1%}")
        c4.metric("Risk (Alpha)", f"{best['Alpha']:.1%}")

        st.markdown("---")
        st.subheader("📋 Full Operational Stress Test (OC Table)")
        st.write(f"Testing design (Hurdle: {best['Hurdle']:.0%}, Conf: {best['Conf']:.0%}) across 13 scenarios:")
        
        stress_df = run_full_stress_test(best['N'], best['Hurdle'], best['Conf'], safe_limit, p1, true_toxic_rate)
        st.table(stress_df)

        st.info("💡 **Clinical Tip:** Look at Scenarios 11-13. If your Safety Stop % is near 100%, this design is highly 'legitimate' for ethics committees.")
    else:
        st.error("No design found. Try increasing the N Range or lowering Safety Power requirements.")
