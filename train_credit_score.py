import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib
import json
import pprint
import time

start_time = time.perf_counter()

#^^ Define Data, Path, and Parameters

MODEL_FILENAME = './models/credit_score_xgb_model.joblib'
ARTIFACTS_FILENAME = './models/preprocessing_artifacts.json'
DATA_FILENAME = './data/train.csv'

NUM_TREES = 10000
LEARNING_RATE = 0.05

columns_to_drop = [
    'ID',
    'Customer_ID',
    'Name',
    'SSN',
    'Month',
    'Type_of_Loan',
    'Credit_History_Age'
]

numerical_obj_cols = [
    'Age',
    'Annual_Income',
    'Num_of_Loan',
    'Interest_Rate',
    'Num_of_Delayed_Payment',
    'Changed_Credit_Limit',
    'Amount_invested_monthly', 
    'Monthly_Balance',
    'Num_Credit_Card',
    'Delay_from_due_date',
    'Credit_Utilization_Ratio',
    'Num_Credit_Inquiries',
    'Total_EMI_per_month',
    'Outstanding_Debt'
]

numerical_cols_to_impute = numerical_obj_cols + ['Monthly_Inhand_Salary', 'Num_Credit_Inquiries']

nominal_cols = ['Occupation', 'Credit_Mix', 'Payment_Behaviour']

#^^ --- Data Loading and Initial Drop ---
try:
    df = pd.read_csv(DATA_FILENAME)
except FileNotFoundError:
    print(f"Error: {DATA_FILENAME} not found.")
    exit()

df_dropped = df.drop(columns=columns_to_drop, axis=1)
df = df_dropped.copy()

#^^ --- Clean and Impute Numerical Columns ---
def clean_non_numeric(series):
    #>-< CLean non-numeric strings ('_', ',') and convert to numeric
    series = series.astype(str).str.replace(r'[$,]', '', regex=True)
    series = pd.to_numeric(series, errors='coerce')
    return series

for col in numerical_obj_cols:
    df[col] = clean_non_numeric(df[col])

#>-< Handle Age Outliers
df['Age'] = df['Age'].clip(lower=18, upper=100)
df['Num_Credit_Card'] = df['Num_Credit_Card'].clip(lower=0, upper=10)
df['Num_of_Loan'] = df['Num_of_Loan'].clip(lower=0, upper=10)
df['Num_Credit_Inquiries'] = df['Num_Credit_Inquiries'].clip(lower=0, upper=100)

#>-< Calculate Medians
imputation_medians = {}
for col in numerical_cols_to_impute:
    median_val = df[col].median()
    imputation_medians[col] = median_val
    df[col].fillna(median_val, inplace=True)

#^^ --- Encode Categorical Columns ---

#>-< Target Variable Encoding (Ordinal) - NOTE: This column will be dropped later
score_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
df['Credit_Score_Encoded'] = df['Credit_Score'].map(score_mapping)
df.drop('Credit_Score', axis=1, inplace=True)

#>-< Ordinal Encoding for Payment_of_Min_Amount
payment_mapping = {'No': 0, 'NM': 1, 'Yes': 2}
df['Payment_of_Min_Amount_Encoded'] = df['Payment_of_Min_Amount'].map(payment_mapping)
df.drop('Payment_of_Min_Amount', axis=1, inplace=True)

#>-< One-Hot Encoding for remaining nominal categorical features
df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

#^^ --- Separate Features (X) and Target (y) and Train Model ---
X = df_encoded.drop('Credit_Score_Encoded', axis=1)
y = df_encoded['Credit_Score_Encoded']

#>-< Save feature list (column names)
feature_names = list(X.columns)

#>-< Combine all artifacts into a single dictionary
artifacts = {
    'feature_names': feature_names,
    'imputation_medians': imputation_medians,
    'payment_mapping': payment_mapping,
    'target_mapping': score_mapping, # useful for inverse mapping later
    'nominal_cols': nominal_cols
}

#^^ --- Split the data ---

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#^^ --- Train the XGBoost Model ---
xgb_model = XGBClassifier(
    objective='multi:softmax',      #^^ indicate classification
    num_class=3,                    #^^ obvious aint it
    n_estimators=NUM_TREES,         #^^ # of trees
    learning_rate=LEARNING_RATE,    #^^ LEARNING_RATE*value of tree
    max_depth=10,
    eval_metric='mlogloss',
    random_state=404,               #^^ seed, set it to make sure same data produce same model
    tree_method='hist'
)

print("Starting XGBoost model training...")
xgb_model.fit(X_train, y_train)
print("Model training complete.")
try:
    pprint.pprint(numerical_obj_cols)
except Exception as e:
    print(e)

#^^ --- Save Model and Preprocessing Artifacts ---

#>-< Save the trained model
joblib.dump(xgb_model, MODEL_FILENAME)
print(f"Trained model saved to {MODEL_FILENAME}")

#>-< Save preprocessing artifacts
with open(ARTIFACTS_FILENAME, 'w') as f:
    json.dump(artifacts, f, indent=4)
print(f"Preprocessing artifacts saved to {ARTIFACTS_FILENAME}")

#^^ --- Evaluate Models ---
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Poor (0)', 'Standard (1)', 'Good (2)'])
print("\n--- Model Evaluation Results ---")
print(f"Accuracy Score: {accuracy:.4f}")
print("\nClassification Report:\n", report)

#>-< Calculate execution time
end_time = time.perf_counter()
duration = end_time - start_time

print(f"Execution time: {duration:.4f} seconds")

#^^ Log Model Parameters for Future Analysis
try:
    with open('log.txt', 'a') as f:
        # pprint.pprint(numerical_obj_cols, stream=f, indent=4)
        f.write("Parameter:\n")
        f.write(f"    NUM_TREES = {NUM_TREES}\n")
        f.write(f"    LEARNING_RATE = {LEARNING_RATE}\n")
        f.write("\n--- Model Evaluation Results ---\n")
        f.write(f"Accuracy Score: {accuracy:.4f}\n")
        f.write(f"\nClassification Report:\n{report}")
        f.write(f"\nExecution time: {duration:.4f} seconds\n\n")
        f.write("^^--------------------------------------------------------^^\n\n")
except:
    print("ERROR")

