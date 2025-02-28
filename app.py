import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
st.title("📊 Advanced Student Performance Dashboard")


@st.cache_data
def load_data():
    return pd.read_csv("your_fixed_dataset.csv")

df = load_data()

    # Display dataset
    if st.checkbox("Show Raw Data"):
        st.write(df.head())

    # Data Preprocessing
    df.drop(['Student_ID', 'First_Name', 'Last_Name', 'Email'], axis=1, inplace=True)

    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Interactive Filters
    st.sidebar.header("Filters")

    department_mapping = {idx: name for idx, name in enumerate(label_encoders['Department'].classes_)}
    department_names = [department_mapping[val] for val in df['Department'].unique()]

    selected_departments = st.sidebar.multiselect("Select Department", department_names)

    if selected_departments:
        selected_indices = [key for key, value in department_mapping.items() if value in selected_departments]
        df = df[df['Department'].isin(selected_indices)]

    st.sidebar.subheader("Range Filters")
    age_range = st.sidebar.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
    df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]

    # Key Metrics
    st.header("📈 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Total Score", round(df['Total_Score'].mean(), 2))
    col2.metric("Average Study Hours", round(df['Study_Hours_per_Week'].mean(), 2))
    col3.metric("Average Stress Level", round(df['Stress_Level (1-10)'].mean(), 2))

    # Feature Importance
    st.header("🔍 Feature Importance")
    X = df.drop('Grade', axis=1)
    y = df['Grade']

    model = RandomForestClassifier()
    model.fit(X, y)

    feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importance["Importance"], y=feature_importance["Feature"], palette="magma", ax=ax)
    ax.set_title("Feature Importance for Predicting Student Grades")
    st.pyplot(fig)

    # Correlation Heatmap
    st.header("📊 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Interactive Scatter Plot
    st.header("📌 Interactive Scatter Plot")
    x_axis = st.selectbox("Select X-axis", X.columns)
    y_axis = st.selectbox("Select Y-axis", X.columns)

    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x_axis], y=df[y_axis], hue=df['Grade'], palette="viridis", ax=ax)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    st.pyplot(fig)

    # Download Processed Data
    st.header("📥 Download Processed Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="processed_student_data.csv", mime="text/csv")

    st.success("Dashboard Updated Successfully!")
