# ============================================================
# CUSTOMER CHURN PREDICTION USING MACHINE LEARNING
# File: customer_churn_prediction.py
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------
df = pd.read_excel("Telco_customer_churn.xlsx")
print("Dataset Loaded Successfully")
print(df.head())
print(df.columns)


# ------------------------------------------------------------
# 3. BASIC EXPLORATION
# ------------------------------------------------------------
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Churn distribution
sns.countplot(x='Churn Label', data=df)
plt.title("Churn Distribution")
plt.show()

# ------------------------------------------------------------
# 4. DATA CLEANING & PREPROCESSING
# ------------------------------------------------------------

# Convert TotalCharges to numeric
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df['Total Charges'].fillna(df['Total Charges'].median(), inplace=True)

# Drop customerID (not useful for prediction)
df.drop('CustomerID', axis=1, inplace=True)

# Encode categorical variables
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split features and target
X = df.drop('Churn Value', axis=1)
y = df['Churn Value']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------------------------
# 5. MODEL TRAINING
# ------------------------------------------------------------

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ------------------------------------------------------------
# 6. MODEL EVALUATION
# ------------------------------------------------------------

print("\n--- Logistic Regression Performance ---")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))

print("\n--- Random Forest Performance ---")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d')
plt.title("Confusion Matrix - Random Forest")
plt.show()

# ROC-AUC Score
rf_prob = rf_model.predict_proba(X_test)[:, 1]
print("Random Forest ROC-AUC Score:", roc_auc_score(y_test, rf_prob))

# ------------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ------------------------------------------------------------
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.show()

# ------------------------------------------------------------
# 8. SAVE MODEL (OPTIONAL)
# ------------------------------------------------------------
import joblib
joblib.dump(rf_model, "churn_model.pkl")
print("Model saved as churn_model.pkl")

# ============================================================
# END OF SCRIPT
# ============================================================