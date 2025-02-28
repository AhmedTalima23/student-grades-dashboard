import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Students_Grading_Dataset.csv")

    # Drop non-numeric columns
    df.drop(['Student_ID', 'First_Name', 'Last_Name', 'Email'], axis=1, inplace=True)

    # Encode categorical columns
    categorical_cols = ['Gender', 'Department', 'Extracurricular_Activities', 'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level', 'Grade']
    df[categorical_cols] = df[categorical_cols].apply(LabelEncoder().fit_transform)

    return df

data = load_data()

# Title
st.title("📊 Advanced Student Performance Dashboard")

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
selected_department = st.sidebar.selectbox("Select Department", data['Department'].unique())
filtered_data = data[data['Department'] == selected_department]

# Summary Statistics
st.subheader("📈 Data Overview")
st.write(filtered_data.describe())

# Correlation Heatmap
st.subheader("🔬 Correlation Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(filtered_data.corr(), annot=True, cmap='coolwarm')
st.pyplot(plt)

# Grade Distribution
st.subheader("🎓 Grade Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Grade', data=filtered_data, palette='viridis', ax=ax)
st.pyplot(fig)

# Feature Importance
st.subheader("📊 Feature Importance")
X = filtered_data.drop("Grade", axis=1)
y = filtered_data["Grade"]

model = RandomForestClassifier()
model.fit(X, y)

importance_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)
fig, ax = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=importance_df, palette='mako', ax=ax)
st.pyplot(fig)

st.success("✅ Dashboard Loaded Successfully!")
