# 📊 Advanced Student Performance Dashboard

## 📌 Overview
This project is an **interactive Streamlit dashboard** built in **Python** to analyze and visualize student performance data.  
It provides **real-time insights** into student grades, study patterns, stress levels, and demographics.  

The dashboard also uses **machine learning (Random Forest)** to show feature importance in predicting student grades.  

---

## ⚡ Features
- ✅ Load and preprocess student dataset  
- ✅ Interactive **filters** (by Department & Age range)  
- ✅ **Key Metrics**: Average Score, Study Hours, Stress Level  
- ✅ **Feature Importance** (Random Forest Classifier)  
- ✅ **Correlation Heatmap** of dataset features  
- ✅ **Histograms** for categorical features (decoded values)  
- ✅ **Pie Charts** for Gender & Grade distributions  
- ✅ **Download processed dataset** as CSV  

---

## 📂 Project Structure
├── Students_Grading_Dataset.csv # Input dataset
├── app.py # Main Streamlit app
└── README.md # Project documentation

yaml
Copy code

---

## 🛠️ Tech Stack
- **Python 3.8+**  
- **Streamlit** – interactive web app framework  
- **Pandas** – data manipulation  
- **Matplotlib & Seaborn** – visualization  
- **Scikit-learn** – machine learning (Random Forest, Label Encoding)  

---

## ▶️ How to Run
1. Install dependencies:
   ```bash
   pip install streamlit pandas matplotlib seaborn scikit-learn
Place your dataset file in the same folder:

Students_Grading_Dataset.csv

Run the Streamlit app:

bash
Copy code
streamlit run app.py
Open the local URL (usually http://localhost:8501) in your browser.

📂 Example Dashboard Screenshots

🔹 Key Metrics

- Average Score
- Average Study Hours
- Average Stress Level

🔹 Feature Importance
Random Forest identifies which features affect student grades most.

🔹 Correlation Heatmap
Visualizes relationships between performance metrics.

🔹 Histograms & Pie Charts
Distribution of Gender, Departments, and Grades.
