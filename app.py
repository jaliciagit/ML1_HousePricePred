import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("boston_housing.csv")

@st.cache_resource
def train_model(df):
    X = df.drop(columns="MEDV")
    y = df["MEDV"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler().fit(X_tr)
    model = LinearRegression().fit(X_tr, y_tr)

    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    coef = pd.DataFrame(
        {
            "Coefficient": model.coef_,
            "Std. coefficient": LinearRegression()
            .fit(scaler.transform(X_tr), y_tr)
            .coef_,
        },
        index=X.columns,
    )

    return X, y, X_tr, X_te, y_tr, y_te, scaler, model, pred_tr, pred_te, coef

def metrics(y_true, y_pred, p):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2,
        "Adj_R2": 1 - (1 - r2) * (n - 1) / (n - p - 1),
    }

st.title("🏠 House Price Prediction")
st.write("Multiple Linear Regression on the Boston Housing Dataset")

try:
    df = load_data()
except FileNotFoundError:
    st.error("boston_housing.csv was not found.")
    st.info("Place boston_housing.csv in the same folder as app.py, then run the app again.")
    st.stop()

required_target = "MEDV"
if required_target not in df.columns:
    st.error("The dataset must contain a 'MEDV' target column.")
    st.stop()

X, y, X_tr, X_te, y_tr, y_te, scaler, model, pred_tr, pred_te, coef = train_model(df)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ["Prediction", "Dataset & EDA", "Model Evaluation", "Regression Analysis"]
)

if page == "Prediction":
    st.header("Predict Median House Value")
    st.write("Enter the 13 predictor values used by the trained regression model.")

    cols = st.columns(2)
    inputs = {}

    for i, feature in enumerate(X.columns):
        default = float(X[feature].median())
        if feature == "CHAS":
            default = int(round(default))
            value = cols[i % 2].number_input(
                feature, value=default, step=1, format="%d"
            )
        else:
            value = cols[i % 2].number_input(
                feature, value=default, format="%.4f"
            )
        inputs[feature] = value

    if st.button("Predict House Price", type="primary"):
        input_df = pd.DataFrame([inputs], columns=X.columns)
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted MEDV: {prediction:.2f}")
        st.metric(
            "Estimated Median House Value",
            f"${prediction * 1000:,.0f}"
        )
        st.caption("MEDV is expressed in thousands of dollars in the dataset.")

elif page == "Dataset & EDA":
    st.header("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Basic Statistics")
    st.dataframe(df.describe().T.round(3), use_container_width=True)

    st.subheader("MEDV Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["MEDV"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of MEDV")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation Matrix")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=.5, ax=ax
    )
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)
    plt.close(fig)

elif page == "Model Evaluation":
    st.header("Model Evaluation")

    train_metrics = metrics(y_tr, pred_tr, X.shape[1])
    test_metrics = metrics(y_te, pred_te, X.shape[1])

    metrics_df = pd.DataFrame(
        {"Train": train_metrics, "Test": test_metrics}
    )
    st.dataframe(metrics_df.round(4), use_container_width=True)

    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_te, pred_te, alpha=.7)
    lo, hi_ = y.min(), y.max()
    ax.plot([lo, hi_], [lo, hi_], "r--")
    ax.set_xlabel("Actual MEDV")
    ax.set_ylabel("Predicted MEDV")
    ax.set_title("Actual vs Predicted (Test Set)")
    st.pyplot(fig)
    plt.close(fig)

    resid = y_te - pred_te

    st.subheader("Residual Diagnostics")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(pred_te, resid, alpha=.7)
    ax.axhline(0, color="r", ls="--")
    ax.set_xlabel("Predicted MEDV")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predicted")
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        f"Residual mean: {resid.mean():.3f} | "
        f"Std: {resid.std():.3f} | "
        f"Skew: {resid.skew():.3f}"
    )
    st.write(f"Shapiro-Wilk p-value: {stats.shapiro(resid)[1]:.4g}")

elif page == "Regression Analysis":
    st.header("Regression Coefficients")

    st.dataframe(
        coef.sort_values("Std. coefficient").round(4),
        use_container_width=True
    )

    st.subheader("Standardised Regression Coefficients")
    c = coef["Std. coefficient"].sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(c.index, c.values)
    ax.set_xlabel("Standardised coefficient")
    ax.set_title("Standardised Regression Coefficients")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation with MEDV")
    corr = df.corr()["MEDV"].drop("MEDV").sort_values()
    st.dataframe(corr.round(3).to_frame("Correlation"), use_container_width=True)

    st.subheader("Model Information")
    st.write(f"Intercept: {model.intercept_:.4f}")
    st.write(f"Training records: {len(X_tr)}")
    st.write(f"Testing records: {len(X_te)}")
