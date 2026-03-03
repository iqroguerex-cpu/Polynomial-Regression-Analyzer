import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Polynomial Regression Analyzer",
    page_icon="📈",
    layout="wide"
)

# =========================
# DARK PROFESSIONAL UI
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
div[data-testid="stMetricValue"] {
    color: #00FFAA;
}
</style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")
sns.set_style("darkgrid")

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("Position_Salaries.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Model Controls")

degree = st.sidebar.slider("Polynomial Degree", 1, 10, 4)

# =========================
# LINEAR REGRESSION
# =========================
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

r2_linear = r2_score(y, y_lin_pred)

# =========================
# POLYNOMIAL REGRESSION
# =========================
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

y_poly_pred = poly_reg.predict(X_poly)

r2_poly = r2_score(y, y_poly_pred)

# =========================
# HEADER
# =========================
st.title("📈 Polynomial vs Linear Regression Analyzer")
st.caption("Visual Comparison of Model Complexity")

st.divider()

# =========================
# METRICS
# =========================
col1, col2 = st.columns(2)

col1.metric("Linear R²", f"{r2_linear:.3f}")
col2.metric("Polynomial R²", f"{r2_poly:.3f}")

st.divider()

# =========================
# VISUALIZATION
# =========================
st.subheader("Model Comparison")

X_grid = np.arange(X.min(), X.max(), 0.1)
X_grid = X_grid.reshape(-1, 1)

X_grid_poly = poly.transform(X_grid)

fig, ax = plt.subplots(figsize=(8,6))

# Actual Data
ax.scatter(X, y, color="#FF6F61", label="Actual Data", s=60)

# Linear Line
ax.plot(X, y_lin_pred, color="#00BFFF", linewidth=2, label="Linear Regression")

# Polynomial Curve
ax.plot(X_grid, poly_reg.predict(X_grid_poly),
        color="#00FFAA",
        linewidth=3,
        label=f"Polynomial (Degree {degree})")

ax.set_xlabel("Position Level")
ax.set_ylabel("Salary")
ax.legend()

st.pyplot(fig)

st.divider()

# =========================
# EXPLANATION SECTION
# =========================
st.subheader("Model Interpretation")

if degree == 1:
    st.info("Degree 1 is Linear Regression. Model may underfit complex data.")
elif degree <= 3:
    st.success("Moderate complexity. Captures some curvature.")
elif degree <= 6:
    st.warning("High complexity. Watch for overfitting.")
else:
    st.error("Very high complexity. Likely overfitting.")

st.markdown("""
- Linear regression fits a straight line.
- Polynomial regression adds powers of X (X², X³, ...).
- Higher degree increases flexibility.
- Too high degree causes overfitting.
""")
