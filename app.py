import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
def load_data():
    df = pd.read_csv("Students_Grading_Dataset.csv")

    # Drop non-numeric columns
    df.drop(['Student_ID', 'First_Name', 'Last_Name', 'Email'], axis=1, inplace=True)

    # Encode categorical columns
    categorical_columns = ['Gender', 'Department', 'Extracurricular_Activities', 'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level', 'Grade']
    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Scale numeric columns
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df

df = load_data()

# Title
st.title("Advanced Student Performance Dashboard")

# Dataset overview
st.header("Dataset Overview")
st.write(df.head())

# Feature Importance Calculation
st.header("Feature Importance")
X = df.drop("Grade", axis=1)
y = df["Grade"]

model = RandomForestClassifier()
model.fit(X, y)

feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

st.bar_chart(feature_importance.set_index("Feature"))

# Correlation Heatmap
st.header("Correlation Heatmap")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
st.pyplot(plt)

# Interactive Filters
st.sidebar.header("Filter Options")
selected_department = st.sidebar.selectbox("Select Department", df['Department'].unique())
filtered_df = df[df['Department'] == selected_department]

st.write(f"Filtered Data for Department: {selected_department}")
st.write(filtered_df)

# Distribution of Grades
st.header("Grade Distribution")
plt.figure(figsize=(8, 5))
sns.countplot(x='Grade', data=df, palette='viridis')
st.pyplot(plt)

# Study Hours vs Grade
st.header("Study Hours vs Grade")
plt.figure(figsize=(8, 5))
sns.boxplot(x='Grade', y='Study_Hours_per_Week', data=df)
st.pyplot(plt)

st.success("Dashboard is running smoothly!")
