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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier



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

# Remove leakage columns
leakage_cols = ['Churn Label', 'Churn Score', 'Churn Reason', 'CLTV']
df = df.drop(columns=leakage_cols)

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

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=200, random_state=42)    
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# XGBoost Classifier
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# ------------------------------------------------------------
# 6. MODEL EVALUATION
# ------------------------------------------------------------

print("\n--- Logistic Regression Performance ---")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))

print("\n--- Random Forest Performance ---")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

print("\n--- Decision Tree Performance ---")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

print("\n--- Gradient Boosting Performance ---")
print("Accuracy:", accuracy_score(y_test, gb_pred))
print(classification_report(y_test, gb_pred))

print("\n--- XGBoost Performance ---")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))


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
joblib.dump(log_model, "logistic_model.pkl")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(dt_model, "decision_tree_model.pkl")
joblib.dump(gb_model, "gradient_boosting_model.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")


# ------------------------------------------------------------
# 9. SAVE ALL MODEL EVALUATIONS TO TEXT FILE
# ------------------------------------------------------------

results = []

# Logistic Regression
results.append("=== Logistic Regression ===")
results.append(f"Accuracy: {accuracy_score(y_test, log_pred):.4f}")
results.append(classification_report(y_test, log_pred))

# Random Forest
results.append("\n=== Random Forest ===")
results.append(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
results.append(classification_report(y_test, rf_pred))

# Decision Tree
results.append("\n=== Decision Tree ===")
results.append(f"Accuracy: {accuracy_score(y_test, dt_pred):.4f}")
results.append(classification_report(y_test, dt_pred))

# Gradient Boosting
results.append("\n=== Gradient Boosting ===")
results.append(f"Accuracy: {accuracy_score(y_test, gb_pred):.4f}")
results.append(classification_report(y_test, gb_pred))

# XGBoost
results.append("\n=== XGBoost ===")
results.append(f"Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
results.append(classification_report(y_test, xgb_pred))

# ------------------------------------------------------------
# Identify Best Model
# ------------------------------------------------------------
model_scores = {
    "Logistic Regression": accuracy_score(y_test, log_pred),
    "Random Forest": accuracy_score(y_test, rf_pred),
    "Decision Tree": accuracy_score(y_test, dt_pred),
    "Gradient Boosting": accuracy_score(y_test, gb_pred),
    "XGBoost": accuracy_score(y_test, xgb_pred)
}

best_model = max(model_scores, key=model_scores.get)
best_score = model_scores[best_model]

results.append("\n=== BEST MODEL ===")
results.append(f"Best Model: {best_model}")
results.append(f"Accuracy: {best_score:.4f}")

# ------------------------------------------------------------
# Write to text file
# ------------------------------------------------------------
with open("model_evaluation_results.txt", "w") as f:
    for line in results:
        f.write(line + "\n")

print("All evaluation results saved to model_evaluation_results.txt")

# ============================================================
# END OF SCRIPT
# ============================================================