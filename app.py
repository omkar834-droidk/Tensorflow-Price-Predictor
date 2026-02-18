import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Laptop Budget Advisor",
    page_icon="💻",
    layout="wide"
)

# -----------------------------
# Load Model & Files
# -----------------------------
try:
    model = load_model("laptop_price_model.keras")
    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)
    with open("dropdowns.pkl", "rb") as f:
        dropdowns = pickle.load(f)
except:
    st.error("⚠ Required model files not found.")
    st.stop()

# -----------------------------
# Header Section
# -----------------------------
st.title("💻 Smart Laptop Price & Budget Advisor")
st.markdown("AI-powered price prediction with financial affordability analysis.")
st.divider()

# -----------------------------
# INPUT SECTION (LEFT PANEL STYLE)
# -----------------------------
st.subheader("🔧 Laptop Specifications")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Brand", dropdowns["Company"])
    type_name = st.selectbox("Laptop Type", dropdowns["TypeName"])
    cpu = st.selectbox("CPU Brand", dropdowns["Cpu_brand"])
    gpu = st.selectbox("GPU Brand", dropdowns["Gpu_brand"])
    os = st.selectbox("Operating System", dropdowns["OpSys"])

with col2:
    ram = st.selectbox("RAM (GB)", dropdowns["Ram"])
    inches = st.number_input("Screen Size (Inches)", 10.0, 20.0, step=0.1)
    ssd = st.number_input("SSD (GB)", 0, 4000, step=128)
    hdd = st.number_input("HDD (GB)", 0, 4000, step=256)
    weight = st.number_input("Weight (kg)", 0.5, 5.0, step=0.1)

st.divider()

st.subheader("💰 Financial Details")
user_income = st.number_input("Enter Your Monthly Income (₹)", 10000, 1000000, step=5000)

st.divider()

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔮 Analyze & Predict", use_container_width=True):

    with st.spinner("Analyzing specifications and affordability..."):

        input_data = {
            "Company": company,
            "TypeName": type_name,
            "Cpu_brand": cpu,
            "Gpu_brand": gpu,
            "OpSys": os,
            "Ram": ram,
            "Inches": inches,
            "SSD": ssd,
            "HDD": hdd,
            "Weight": weight
        }

        input_df = pd.DataFrame([input_data])
        encoded_df = pd.get_dummies(input_df)
        encoded_df = encoded_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(encoded_df)[0][0]
        predicted_price = int(prediction)

        income_ratio = predicted_price / user_income
        affordability_percent = (predicted_price / user_income) * 100

        if income_ratio <= 0.5:
            status = "✅ Affordable"
            message = "This laptop fits comfortably within your income."
            color = "green"
        elif income_ratio <= 1:
            status = "⚠ Stretch Budget"
            message = "This may stretch your budget. EMI recommended."
            color = "orange"
        else:
            status = "❌ Not Recommended"
            message = "This laptop is expensive compared to your income."
            color = "red"

        emi_months = [6, 9, 12, 24]
        emi_values = [predicted_price / m for m in emi_months]

        emi_df = pd.DataFrame({
            "EMI Duration (Months)": emi_months,
            "Monthly EMI (₹)": [int(e) for e in emi_values]
        })

    st.success("📊 Analysis Complete")

    # -----------------------------
    # KPI METRICS ROW
    # -----------------------------
    m1, m2, m3 = st.columns(3)

    m1.metric("💻 Predicted Price", f"₹ {predicted_price:,}")
    m2.metric("💰 Monthly Income", f"₹ {user_income:,}")
    m3.metric("📊 Price vs Income", f"{affordability_percent:.1f}%")

    st.markdown(f"### :{color}[{status}]")
    st.write(message)

    st.divider()

    # -----------------------------
    # EMI TABLE
    # -----------------------------
    st.subheader("💳 EMI Payment Options")
    st.dataframe(emi_df, use_container_width=True)

    st.divider()

    # -----------------------------
    # EMI BAR CHART
    # -----------------------------
    st.subheader("📈 EMI Comparison Chart")

    fig, ax = plt.subplots()
    ax.bar(emi_months, emi_values)
    ax.set_xlabel("EMI Duration (Months)")
    ax.set_ylabel("Monthly EMI (₹)")
    ax.set_title("Monthly EMI vs Duration")
    st.pyplot(fig)

    st.divider()

    # -----------------------------
    # Affordability Progress
    # -----------------------------
    st.subheader("📉 Affordability Indicator")

    st.progress(min(int(affordability_percent), 100))

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("© 2026 Smart Laptop Budget Advisor | AI + Financial Intelligence 🚀")
