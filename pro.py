import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import tempfile
import os

st.set_page_config(page_title="Data Solution", layout="wide")

st.title("Data Solution")
st.markdown("""
This app performs automated data analysis, visualization, and predictive modeling. 
Upload your dataset and follow the steps to get insights!
""")

st.sidebar.title("Navigation")
options = st.sidebar.radio("Select Step:", 
                          ["Upload Data", "Data Cleaning", "EDA", 
                           "Visualization", "Prediction", "Insights"])

# Session state init
for var in ['df', 'cleaned_df', 'target', 'model_type', 'model']:
    if var not in st.session_state:
        st.session_state[var] = None

# Upload Data
if options == "Upload Data":
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("Data uploaded!")
            st.write(df.head())
        except Exception as e:
            st.error(f"Error: {e}")

# Data Cleaning
elif options == "Data Cleaning" and st.session_state.df is not None:
    df = st.session_state.df
    st.header("Data Cleaning")
    issues = []

    if df.isna().sum().sum() > 0:
        issues.append("Missing values detected")
    if df.duplicated().sum() > 0:
        issues.append("Duplicate rows detected")

    st.warning("Issues:" if issues else "No major issues.")
    for issue in issues:
        st.write(f"- {issue}")

    options = st.multiselect("Select cleaning steps:", [
        "Remove Duplicates",
        "Fill Missing (Numeric)",
        "Fill Missing (Categorical)",
        "Remove Rows with Missing",
        "Remove Columns with >30% Missing",
        "Convert Text to Numeric",
        "Standardize Column Names"
    ])

    if st.button("Clean Data"):
        cleaned = df.copy()

        if "Remove Duplicates" in options:
            cleaned = cleaned.drop_duplicates()
        if "Fill Missing (Numeric)" in options:
            num = cleaned.select_dtypes(include=np.number).columns
            imputer = SimpleImputer(strategy='mean')
            cleaned[num] = imputer.fit_transform(cleaned[num])
        if "Fill Missing (Categorical)" in options:
            cat = cleaned.select_dtypes(include='object').columns
            for col in cat:
                cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
        if "Remove Rows with Missing" in options:
            cleaned = cleaned.dropna()
        if "Remove Columns with >30% Missing" in options:
            threshold = len(cleaned) * 0.7
            cleaned = cleaned.dropna(axis=1, thresh=threshold)
        if "Convert Text to Numeric" in options:
            for col in cleaned.select_dtypes(include='object'):
                try:
                    cleaned[col] = pd.to_numeric(cleaned[col])
                except:
                    pass
        if "Standardize Column Names" in options:
            cleaned.columns = cleaned.columns.str.lower().str.replace(' ', '_')

        st.session_state.cleaned_df = cleaned
        st.success("Data cleaned!")
        st.write(cleaned.head())

# EDA using Sweetviz
elif options == "EDA" and st.session_state.cleaned_df is not None:
    df = st.session_state.cleaned_df
    st.header("Exploratory Data Analysis")

    if st.button("Generate Full EDA Report"):
        try:
            report = sv.analyze(df)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
                report_path = tmpfile.name
                report.show_html(filepath=report_path, open_browser=False)

            with open(report_path, 'r') as f:
                html = f.read()
            st.components.v1.html(html, height=1000, scrolling=True)

            with open(report_path, 'rb') as f:
                st.download_button("Download EDA Report", data=f, file_name="eda_report.html", mime="text/html")

        except Exception as e:
            st.error(f"Error generating report: {e}")

# Visualization
elif options == "Visualization" and st.session_state.cleaned_df is not None:
    df = st.session_state.cleaned_df
    st.header("Visualization")
    viz = st.selectbox("Choose Chart Type", [
        "Scatter", "Line", "Bar", "Histogram", "Box", "Pie", "Heatmap"
    ])

    num = df.select_dtypes(include=np.number).columns
    cat = df.select_dtypes(include='object').columns

    if viz == "Scatter" and len(num) >= 2:
        x = st.selectbox("X", num)
        y = st.selectbox("Y", num)
        color = st.selectbox("Color", [None] + list(cat))
        fig = px.scatter(df, x=x, y=y, color=color)
        st.plotly_chart(fig)

    elif viz == "Line" and len(num) >= 2:
        x = st.selectbox("X Axis", num)
        y = st.selectbox("Y Axis", num)
        fig = px.line(df, x=x, y=y)
        st.plotly_chart(fig)

    elif viz == "Bar" and len(cat) >= 1 and len(num) >= 1:
        x = st.selectbox("Category", cat)
        y = st.selectbox("Value", num)
        fig = px.bar(df, x=x, y=y)
        st.plotly_chart(fig)

    elif viz == "Histogram":
        col = st.selectbox("Select Numeric Column", num)
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)

    elif viz == "Box":
        y = st.selectbox("Numeric Column", num)
        x = st.selectbox("Category (Optional)", [None] + list(cat))
        fig = px.box(df, x=x, y=y)
        st.plotly_chart(fig)

    elif viz == "Pie":
        col = st.selectbox("Select Categorical Column", cat)
        fig = px.pie(df, names=col)
        st.plotly_chart(fig)

    elif viz == "Heatmap":
        if len(num) > 1:
            corr = df[num].corr()
            fig = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig)

# Prediction
elif options == "Prediction" and st.session_state.cleaned_df is not None:
    df = st.session_state.cleaned_df
    st.header("Predictive Modeling")

    target = st.selectbox("Select Target Variable", df.columns)
    st.session_state.target = target

    y = df[target]
    X = df.drop(columns=[target])
    for col in X.select_dtypes(include='object'):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if y.dtype in ['int64', 'float64'] and y.nunique() > 10:
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.success(f"Regression Model | R²: {model.score(X_test, y_test):.2f} | MSE: {mean_squared_error(y_test, y_pred):.2f}")
    else:
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.success(f"Classification Model | Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    st.session_state.model = model

    # Feature importance
    st.subheader("Feature Importance")
    feat_imp = pd.DataFrame({
        'Feature': df.drop(columns=[target]).columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=feat_imp, ax=ax)
    st.pyplot(fig)

# Insights
elif options == "Insights" and st.session_state.cleaned_df is not None:
    df = st.session_state.cleaned_df
    st.header("Data Storytelling & Insights")
    st.write(f"📊 Your dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
    
    num = df.select_dtypes(include=np.number).columns
    cat = df.select_dtypes(include='object').columns

    if len(num):
        st.write("### 📈 Numeric Insights")
        for col in num:
            st.write(f"**{col}**: min = {df[col].min()}, max = {df[col].max()}, mean = {df[col].mean():.2f}, std = {df[col].std():.2f}")

    if len(cat):
        st.write("### 🧩 Categorical Insights")
        for col in cat:
            vc = df[col].value_counts().head(3)
            st.write(f"**{col}** top values:")
            for val, cnt in vc.items():
                st.write(f" - {val}: {cnt} times")

    if len(num) >= 2:
        corr = df[num].corr().unstack().sort_values(ascending=False)
        corr = corr[corr < 1].drop_duplicates()
        if not corr.empty:
            st.write("### 🔗 Correlation Highlights")
            top_pos = corr.idxmax(), corr.max()
            top_neg = corr.idxmin(), corr.min()
            st.write(f"Most positive: {top_pos[0]} → r={top_pos[1]:.2f}")
            st.write(f"Most negative: {top_neg[0]} → r={top_neg[1]:.2f}")
