import streamlit as st
import numpy as np
from scipy.stats import beta
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Trial Designer & OC Calculator", layout="wide")

st.title("🧬 Clinical Trial Designer (OC & Sample Size Calculator)")

# --- SIDEBAR: DESIGN GOALS ---
st.sidebar.header("🎯 Design Objectives")
p0 = st.sidebar.slider("Null Efficacy (p0)", 0.3, 0.7, 0.5, help="Efficacy of a 'failure' drug.")
p1 = st.sidebar.slider("Target Efficacy (p1)", 0.5, 0.9, 0.7, help="Efficacy of your 'dream' drug.")

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Risk Tolerance")
max_alpha = st.sidebar.slider("Max False Positive Rate (Alpha)", 0.01, 0.20, 0.10)
min_power = st.sidebar.slider("Min Statistical Power", 0.70, 0.95, 0.80)

st.sidebar.markdown("---")
st.sidebar.header("📐 Constraints")
n_range = st.sidebar.slider("Sample Size Search Range", 20, 200, (40, 100))

# --- SIMULATION ENGINE ---
def run_simulation(n, p_true, hurdle, conf_req, sims=5000):
    # Simulate 'sims' number of trials
    successes = np.random.binomial(n, p_true, sims)
    # Bayesian posterior check: P(rate > hurdle | data) > conf_req
    # Using flat prior (1,1)
    p_success = 1 - beta.cdf(hurdle, 1 + successes, 1 + (n - successes))
    return np.mean(p_success > conf_req)

if st.button("🚀 Run Grid Search & Optimize Design"):
    results = []
    
    with st.spinner("Simulating thousands of trials..."):
        # We test different Hurdles and N values
        for n in range(n_range[0], n_range[1] + 1, 5):
            for hurdle in [0.55, 0.60, 0.65]:
                for conf in [0.70, 0.74, 0.80, 0.85]:
                    # Calculate Alpha (Risk with bad drug)
                    alpha = run_simulation(n, p0, hurdle, conf)
                    if alpha <= max_alpha:
                        # Calculate Power (Success with good drug)
                        power = run_simulation(n, p1, hurdle, conf)
                        results.append({
                            "N": n,
                            "Hurdle": hurdle,
                            "Conf_Req": conf,
                            "Alpha": alpha,
                            "Power": power
                        })

    if results:
        df = pd.DataFrame(results)
        # Find the smallest N that meets power requirements
        valid_designs = df[df['Power'] >= min_power].sort_values("N")
        
        if not valid_designs.empty:
            best = valid_designs.iloc[0]
            
            st.success(f"### ✅ Optimal Design Found: N = {int(best['N'])}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Required N", int(best['N']))
            c2.metric("Success Hurdle", f"{best['Hurdle']:.0%}")
            c3.metric("Min. Confidence", f"{best['Conf_Req']:.0%}")

            st.write(f"**Operating Characteristics:** Under this design, if your drug is {p1:.0%} effective, you have a **{best['Power']:.1%} chance** of success. If it is only {p0:.0%} effective, the risk of a false positive is only **{best['Alpha']:.1%}**.")
            
            # --- VISUALIZATION ---
            st.subheader("Design Landscape")
            fig = px.scatter(df, x="Alpha", y="Power", color="N", size="N", 
                             hover_data=['Hurdle', 'Conf_Req'],
                             title="Power vs. Risk for Different Designs")
            fig.add_vline(x=max_alpha, line_dash="dash", line_color="red")
            fig.add_hline(y=min_power, line_dash="dash", line_color="green")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No design met your Power requirements in this N range. Try increasing the search range or lowering the Power requirement.")
    else:
        st.error("No designs stayed below the Max Alpha. Try a stricter Confidence Requirement or a higher Hurdle.")
