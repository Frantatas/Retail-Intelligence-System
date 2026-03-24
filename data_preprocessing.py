import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("supermarket_sales.csv")

print(df.head())

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# -------------------------
# Feature Engineering
# -------------------------
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month
df["Weekend"] = (df["Date"].dt.dayofweek >= 5).astype(int)
df["Discount"] = np.random.randint(0, 50, size=len(df))
df["Sales"] = df["Quantity"] * df["Unit price"] * (1 - df["Discount"] / 100)
# -------------------------
# Targets
# -------------------------
# Regression target
y_reg = df["Sales"]

# Classification target: HIGH vs LOW sale
median_sales = df["Sales"].median()
df["Sales_Class"] = df["Sales"].apply(lambda x: 1 if x > median_sales else 0)
y_clf = df["Sales_Class"]

# -------------------------
# Features (match new UI)
# -------------------------
X = df[
    [
        "Quantity",
        "Unit price",
        "Month",
        "Customer type",
        "Gender",
        "Product line",
        "Payment",
        "Weekend",
        "Discount"
    ]
]

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Convert True/False columns to 0/1
for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

print("\nProcessed columns:")
print(X.columns)

print("\nFirst 5 rows:")
print(X.head())

# -------------------------
# Train/Test Split
# -------------------------
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

print("\nRegression train shape:", X_train_reg.shape)
print("Regression test shape:", X_test_reg.shape)
print("Classification train shape:", X_train_clf.shape)
print("Classification test shape:", X_test_clf.shape)

# -------------------------
# 1. Linear Regression
# -------------------------
print("\nTraining Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)
y_pred_lr = lr_model.predict(X_test_reg)

print("\n=== Linear Regression ===")
print("MSE:", mean_squared_error(y_test_reg, y_pred_lr))
print("R2 Score:", r2_score(y_test_reg, y_pred_lr))

# -------------------------
# 2. Logistic Regression
# -------------------------
print("\nTraining Logistic Regression...")
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_clf, y_train_clf)
y_pred_log = log_model.predict(X_test_clf)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_log))
print(classification_report(y_test_clf, y_pred_log))

# -------------------------
# 3. Decision Tree
# -------------------------
print("\nTraining Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train_clf, y_train_clf)
y_pred_dt = dt_model.predict(X_test_clf)

print("\n=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_dt))
print(classification_report(y_test_clf, y_pred_dt))

# -------------------------
# 4. Random Forest
# -------------------------
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_clf, y_train_clf)
y_pred_rf = rf_model.predict(X_test_clf)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_rf))
print(classification_report(y_test_clf, y_pred_rf))

# -------------------------
# 5. Neural Network
# -------------------------
print("\nScaling data for Neural Network...")
scaler = StandardScaler()
X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

print("Training Neural Network...")
nn_model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    max_iter=500,
    learning_rate_init=0.001,
    random_state=42
)
nn_model.fit(X_train_clf_scaled, y_train_clf)
y_pred_nn = nn_model.predict(X_test_clf_scaled)

print("\n=== Neural Network ===")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_nn))
print(classification_report(y_test_clf, y_pred_nn))

# -------------------------
# Save models and metadata
# -------------------------
joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(log_model, "log_model.pkl")
joblib.dump(dt_model, "dt_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(nn_model, "nn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\nModels and feature columns saved successfully!")

print("\n=== MODEL SUMMARY ===")
print("Linear Regression R2:", r2_score(y_test_reg, y_pred_lr))
print("Logistic Accuracy:", accuracy_score(y_test_clf, y_pred_log))
print("Decision Tree Accuracy:", accuracy_score(y_test_clf, y_pred_dt))
print("Random Forest Accuracy:", accuracy_score(y_test_clf, y_pred_rf))
print("Neural Network Accuracy:", accuracy_score(y_test_clf, y_pred_nn))