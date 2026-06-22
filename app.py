import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ==========================
# LOAD FILES
# ==========================

model = joblib.load(
    "dengue_xgb_model.pkl"
)

feature_names = joblib.load(
    "feature_names.pkl"
)

threshold = joblib.load(
    "ews_threshold.pkl"
)

ews_data = pd.read_csv(
    "Dengue_EWS_Result.csv"
)

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Dengue Early Warning System",
    layout="wide"
)

# HEADER
st.title(
    "🦟 Dengue Early Warning System"
)

st.markdown(
"""
Machine Learning-Based Forecasting and Early Warning System
for Dengue Surveillance
"""
)

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "📊 Dashboard",

    "🔮 Prediction",

    "🚨 Early Warning",

    "🧠 Explainable AI",

    "ℹ️ About"

])

# Tab 1 - Dashboard

with tab1:

# Matrics
  col1, col2, col3 = st.columns(3)

col1.metric(
        "Total Records",
        len(ews_data)
    )

col2.metric(
        "Mean Cases",
        int(
            ews_data["Actual_Cases"].mean()
        )
    )

col3.metric(
        "Max Cases",
        int(
            ews_data["Actual_Cases"].max()
        )
    )
  
# Actual vs Prediction

fig = go.Figure()

fig.add_trace(

        go.Scatter(

            y=ews_data["Actual_Cases"],

            name="Actual"

        )
    )

fig.add_trace(

        go.Scatter(

            y=ews_data["Predicted_Cases"],

            name="Prediction"

        )
    )

st.plotly_chart(
        fig,
        use_container_width=True
    )

# Tab 2 - Prediction

with tab2:
  
# Sidebar Input
    st.subheader(
        "Climate Variables Input"
    )

    user_input = {}

for feature in feature_names:

        user_input[feature] = st.number_input(

            feature,

            value=0.0
        )

# Prediction Button
pred_cases = None

if st.button(
        "Predict Dengue Cases"
    ):
    input_df = pd.DataFrame(
        [user_input]
    )
    pred_log = model.predict(input_df)
    pred_cases = int(pred_log[0])

if pred_cases is not None:
    st.metric(
        "Predicted Cases",
        f"{pred_cases:,.0f}"
    )

# TAB 3 — Early Warning System

with tab3:
      
# Risk Function

    def classify_risk(x):

        if x < threshold["normal"]:

            return "NORMAL"

        elif x < threshold["alert"]:

            return "ALERT"

        elif x < threshold["outbreak"]:

            return "OUTBREAK"

        else:

            return "SEVERE OUTBREAK"

# Example

latest_prediction = (
        ews_data[
            "Predicted_Cases"
        ].iloc[-1]
    )

risk = classify_risk(
        latest_prediction
    )

# Risk Card

st.subheader(
        f"Current Status: {risk}"
    )

if risk == "NORMAL":

        st.success(
            "🟢 NORMAL"
        )

elif risk == "ALERT":

        st.warning(
            "🟡 ALERT"
        )

elif risk == "OUTBREAK":

        st.error(
            "🟠 OUTBREAK"
        )

else:

        st.error(
            "🔴 SEVERE OUTBREAK"
        )

# Threshold Plot

fig = go.Figure()

fig.add_hline(

        y=threshold["normal"],

        annotation_text="Normal"
    )

fig.add_hline(

        y=threshold["alert"],

        annotation_text="Alert"
    )

fig.add_hline(

        y=threshold["outbreak"],

        annotation_text="Outbreak"
    )

st.plotly_chart(
        fig,
        use_container_width=True
    )


# TAB 4 — Explainable AI

with tab4:
    # Prepare feature importance DataFrame
    try:
        # sklearn-style
        importances = model.feature_importances_
    except Exception:
        # try XGBoost booster
        try:
            booster = model.get_booster()
            score = booster.get_score(importance_type="weight")
            # map to feature_names
            importances = [score.get(f, 0) for f in feature_names]
        except Exception:
            importances = [0] * len(feature_names)

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    importance.to_csv("feature_importance.csv", index=False)
     
    fig = px.bar(importance.head(20), x="Importance", y="Feature", orientation="h")

st.plotly_chart(
    fig,
    use_container_width=True
)

st.write(
    "Top Risk Drivers"
)

st.dataframe(
    importance.head(10)
)

# TAB 5 — About

with tab5:
     
     st.markdown("""

### Project Information

Dengue Early Warning System

Machine Learning Model:
XGBoost

Features:
Meteorological Variables

Output:
- Predicted Dengue Cases
- Risk Classification
- Early Warning System

Author:
A Alfarobi

""")
     