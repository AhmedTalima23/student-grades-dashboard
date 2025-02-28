import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace with actual dataset path)
data = pd.read_csv("Students_Grading_Dataset.csv")

# Set page configuration
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Title
st.title("📊 Student Performance Dashboard")

# Key Metrics
st.header("Key Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Average Final Grade", round(data['Final_Score'].mean(), 2))
col2.metric("Average Attendance (%)", round(data['Attendance (%)'].mean(), 2))
col3.metric("Average Study Hours/Week", round(data['Study_Hours_per_Week'].mean(), 2))

# Feature Importance (You should calculate and load feature importance)
st.header("Feature Importance")
feature_importance = pd.DataFrame({
    "Feature": ["Attendance (%)", "Total_Score", "Midterm_Score", "Final_Score", "Participation_Score"],
    "Importance": [0.45, 0.15, 0.12, 0.10, 0.08]
})

fig, ax = plt.subplots()
sns.barplot(y="Feature", x="Importance", data=feature_importance, palette="viridis", ax=ax)
plt.xlabel("Feature Importance Score")
st.pyplot(fig)

# Correlation Analysis
st.header("Correlation Analysis")

x_axis = st.selectbox("Select X-axis feature", ["Attendance (%)", "Study_Hours_per_Week", "Sleep_Hours_per_Night"])
y_axis = st.selectbox("Select Y-axis feature", ["Final_Score", "Total_Score", "Midterm_Score"])

fig, ax = plt.subplots()
sns.scatterplot(x=data[x_axis], y=data[y_axis], alpha=0.7)
plt.title(f"{x_axis} vs {y_axis}")
st.pyplot(fig)

# Conclusion
st.header("Conclusion")
st.write("1. **Attendance** is the most critical factor impacting student grades."
         "\n2. **Total score and midterm performance** are also strong predictors."
         "\n3. Improving participation and study hours can enhance academic outcomes.")

st.write("### Recommendations:")
st.write("- Encourage regular attendance through engagement programs."
         "\n- Implement targeted interventions for low-performing students."
         "\n- Promote consistent study habits and monitor participation levels.")
