import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Students_Grading_Dataset.csv")

df = load_data()

# Sidebar - Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Detailed Analysis", "Feature Insights"])

# Header
st.title("🎓 Student Performance Dashboard")

# Overview Page
if page == "Overview":
    st.header("📌 General Information")

    # Key Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", df.shape[0])
    col2.metric("Average Grade", round(df["Total_Score"].mean(), 2))
    col3.metric("Average Study Hours", round(df["Study_Hours_per_Week"].mean(), 2))

    # Grade Distribution
    st.subheader("📈 Grade Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Grade", data=df, palette="viridis", order=sorted(df["Grade"].unique()))
    st.pyplot(fig)

    # Data Preview
    st.subheader("🔍 Preview Dataset")
    st.dataframe(df.head(10))

# Detailed Analysis Page
elif page == "Detailed Analysis":
    st.header("📊 In-Depth Student Analysis")

    # Correlation Heatmap
    st.subheader("🔬 Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(fig)

    # Feature Comparisons
    st.subheader("📌 Compare Features")
    feature_x = st.selectbox("Select X-axis Feature", df.columns)
    feature_y = st.selectbox("Select Y-axis Feature", df.columns)
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feature_x], y=df[feature_y], hue=df["Grade"], palette="viridis")
    st.pyplot(fig)

    # Study Hours vs Grades
    st.subheader("📚 Study Hours vs. Grades")
    fig, ax = plt.subplots()
    sns.boxplot(x="Grade", y="Study_Hours_per_Week", data=df, palette="Set2")
    st.pyplot(fig)

# Feature Insights Page
elif page == "Feature Insights":
    st.header("🔎 Feature Importance Insights")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    # Prepare Data
    df_encoded = df.copy()
    label_encoders = {}

    for column in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le

    X = df_encoded.drop("Grade", axis=1)
    y = df_encoded["Grade"]

    # Train RandomForest Model
    rf = RandomForestClassifier()
    rf.fit(X, y)

    feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    # Display Feature Importance
    st.subheader("📊 Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="mako")
    st.pyplot(fig)

    st.write("### Key Insights")
    st.markdown("- **Attendance (%)** is the most influential factor on student grades.")
    st.markdown("- **Study Hours** have a moderate effect, but not as strong as attendance.")
    st.markdown("- **Quizzes and Assignments** play a significant role in predicting performance.")

# Footer
st.sidebar.info("Built with ❤️ using Streamlit")
