# **Customer Churn Prediction – Project Documentation**

## **1. Introduction**
Customer churn is one of the most critical challenges in the telecom industry. Retaining existing customers is significantly more cost‑effective than acquiring new ones. This project builds a machine‑learning system to predict which customers are likely to churn, enabling proactive retention strategies.  
Using the Telco Customer Churn dataset, the project implements a complete end‑to‑end predictive analytics pipeline including data cleaning, leakage removal, preprocessing, model training, evaluation, and saving trained models.

---

## **2. Business Problem**
Telecom companies lose millions annually due to customer churn. Understanding who is likely to churn—and why—helps companies:

- Reduce revenue loss  
- Improve customer satisfaction  
- Target at‑risk customers with personalized offers  
- Optimize marketing and retention budgets  

This project provides a data‑driven solution to identify churn risk early.

---

## **3. Project Objectives**
- Analyze customer behavior and churn patterns  
- Build multiple machine‑learning models  
- Compare model performance  
- Identify the best model for churn prediction  
- Save trained models for future use  
- Provide insights into key churn drivers  

---

## **4. Dataset Overview**
The dataset includes customer demographics, service usage, billing information, and churn labels.

**Key columns include:**
- CustomerID  
- Gender  
- SeniorCitizen  
- Partner / Dependents  
- Tenure  
- Contract Type  
- Payment Method  
- MonthlyCharges / TotalCharges  
- Churn Value (target variable)

---

## **5. Data Preprocessing**

### **5.1 Handling Missing Values**
- Converted *TotalCharges* to numeric  
- Filled missing values using median  

### **5.2 Removing Irrelevant Columns**
- Removed *CustomerID* (identifier only)

### **5.3 Leakage Removal**
To prevent artificially inflated accuracy, the following leakage columns were removed:

- Churn Label  
- Churn Score  
- Churn Reason  
- CLTV  

These columns directly reveal churn and would cause unrealistic model performance.

### **5.4 Encoding Categorical Variables**
- Applied Label Encoding to all object‑type columns

### **5.5 Feature Scaling**
- Applied StandardScaler to numerical features

### **5.6 Train–Test Split**
- 80% training  
- 20% testing  

---

## **6. Machine Learning Models Used**
Five models were trained and evaluated:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  

**Evaluation Metrics:**
- Accuracy  
- Precision  
- Recall  
- F1‑Score  
- ROC‑AUC  

---

## **7. Model Evaluation**
A consolidated evaluation summary is stored in:

```
Evaluation/model_evaluation_results.txt
```

This file includes:

- Full classification reports  
- Accuracy of each model  
- Identification of the best model  

---

## **8. Best Model**
The script automatically identifies the best model based on accuracy.

**Example Output:**
- **Best Model:** Logistic Regression  
- **Accuracy:** 0.87  

This model is saved for future use.

---

## **9. Saving Trained Models**
All trained models are stored in the folder:

```
churn_Models/
```

**Saved files include:**
- logistic_model.pkl  
- random_forest_model.pkl  
- decision_tree_model.pkl  
- gradient_boosting_model.pkl  
- xgboost_model.pkl  

These models can be loaded later for deployment or prediction.

---

## **10. Project Structure**

```
Predictive_Analysis/
│
├── churn_Models/
│       ├── logistic_model.pkl
│       ├── random_forest_model.pkl
│       ├── decision_tree_model.pkl
│       ├── gradient_boosting_model.pkl
│       ├── xgboost_model.pkl
│
├── customer_churn_prediction/
│       ├── customer_churn_prediction.py
│       ├── Teleco_customer_churn.xlsx
│
├── churn_analysis.ipynb
│
├── Evaluation/
│       ├── model_evaluation_results.txt
│
├── Visualisation/
│       ├── Churn_Distribution.png
│       ├── Confusion_matrix_Random_Forest.png
│       ├── Important_Features.png
│
├── README/
│       ├── requirements.txt
│       └── README.txt
```

---

## **11. Technologies Used**
- Python  
- Pandas  
- NumPy  
- Scikit‑Learn  
- XGBoost  
- Matplotlib  
- Seaborn  
- Jupyter Notebook  

---

## **12. Key Insights**
- Contract type, tenure, and monthly charges are strong churn indicators.  
- Customers with month‑to‑month contracts churn more frequently.  
- Long‑term customers (high tenure) are less likely to churn.  
- Electronic check payment method correlates with higher churn.  

---

## **13. Future Enhancements**
- Add SHAP explainability for model interpretation  
- Deploy the model using Flask or FastAPI  
- Build a Streamlit dashboard for interactive predictions  
- Perform hyperparameter tuning for improved accuracy  
- Add customer segmentation using clustering  

---

## **14. Conclusion**
This project successfully builds a robust churn‑prediction system using multiple machine‑learning models. The best model is identified, saved, and ready for deployment. The workflow demonstrates strong analytical, modeling, and engineering skills suitable for real‑world telecom applications.

---