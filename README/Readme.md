CUSTOMER CHURN PREDICTION - PROJECT DOCUMNETATION

1. INTRODUCTION
Customer churn is one of the most critical challenges in the telecom industry. Retaining existing customers is significantly more cost‑effective than acquiring new ones. This project builds a machine‑learning system to predict which customers are likely to churn, enabling proactive retention strategies.
The project uses the Telco Customer Churn dataset and implements a full end‑to‑end predictive analytics pipeline: data cleaning, leakage removal, preprocessing, model training, evaluation, and saving trained models.

2. BUISNESS PROBLEM
Telecom companies lose millions annually due to customer churn. Understanding who is likely to churn and why allows companies to:
- Reduce revenue loss
- Improve customer satisfaction
- Target at‑risk customers with personalized offers
- Optimize marketing and retention budgets
This project provides a data‑driven solution to identify churn risk early.

3. PROJECT OBJECTIVES
- Analyze customer behavior and churn patterns
- Build multiple machine‑learning models
- Compare model performance
- Identify the best model for churn prediction
- Save trained models for future use
- Provide insights into key churn drivers

4. DATASET OVERVIEW
The dataset includes customer demographics, service usage, billing information, and churn labels.
Key columns include:
- CustomerID
- Gender
- SeniorCitizen
- Partner / Dependents
- Tenure
- Contract type
- Payment method
- MonthlyCharges / TotalCharges
- Churn Value (target variable)

5. DATA PREPROCESSING
Several steps were performed to prepare the data:
5.1 HANDELING MISSING VALUES
- Total Charges converted to numeric
- Missing values filled with median
5.2 REMOVING IRRELEVENT COLUMNS
- CustomerID removed (identifier only)
5.3 LEAKAGE APPROVAL
To prevent artificially high accuracy, the following columns were removed:
- Churn Label
- Churn Score
- Churn Reason
- CLTV
These columns directly reveal churn and would cause 100% accuracy if left in the dataset.
5.4 ENCODING CATAGORAL VARIABLES
Label encoding applied to all object‑type columns.
5.5 FEATURE SCALING
StandardScaler applied to numerical features.
5.6 TRAIN-TEST SPLIT
Dataset split into:
- 80% training
- 20% testing

6. MACHINE LEARNING MODEL USED
Five models were trained and evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
Each model was EVALUATED using:
- Accuracy
- Precision
- Recall
- F1‑Score
- ROC‑AUC

7. MODEL EVALUATION
A consolidated evaluation summary is saved in:
model_evaluation_results.txt

This file includes:
- Full classification reports
- Accuracy of each model
- Best model identification

8. BEST MODEL
The script automatically identifies the best model based on accuracy.
Example:
- Best Model: XGBoost
- Accuracy: 0.87
This model is saved for future use.

9. SAVING TRAINED MODELS
All trained models are stored in the folder:
churn_Models/
Saved files include:
- logistic_model.pkl
- random_forest_model.pkl
- decision_tree_model.pkl
- gradient_boosting_model.pkl
- xgboost_model.pkl
These can be loaded later for deployment or prediction.

10. PROJECT STRUCTURE
Predictive_Analysis/
│
├── churn_Models/
│       ├── logistic_model.pkl
│       ├── random_forest_model.pkl
│       ├── decision_tree_model.pkl
│       ├── gradient_boosting_model.pkl
│       ├── xgboost_model.pkl
│
├── customer_churn_prediction
|       ├── customer_churn_prediction.Py
|       ├── Teleco_customer_churn.xlsx
├── churn_analysis.ipynb
├── Evaluation
|       ├──model_evaluation_results.txt
├── Visualisation
|       ├── Churn_Distribution.png
|       ├── Confusion_matrix_Random_Forest.png
|       ├──Important_Features.png
├──README
|       ├── requirements.txt
|       └── README.txt

11. TECHNOLOGY USED
- Python
- Pandas
- NumPy
- Scikit‑Learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

12. KEY INSIGHTS
- Contract type, tenure, and monthly charges are strong churn indicators.
- Customers with month‑to‑month contracts churn more frequently.
- Long‑term customers (high tenure) are less likely to churn.
- Electronic check payment method correlates with higher churn.

13. FUTURE ENHANCEMENTS
- Add SHAP explainability for model interpretation
- Deploy the model using Flask or FastAPI
- Build a Streamlit dashboard for interactive predictions
- Perform hyperparameter tuning for improved accuracy
- Add customer segmentation using clustering

14. CONCLUSION
This project successfully builds a robust churn‑prediction system using multiple machine‑learning models. The best model is identified, saved, and ready for deployment. The workflow demonstrates strong analytical, modeling, and engineering skills suitable for real‑world telecom applications.
