import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib
import json
import pprint
import time
from typing import List, Dict, Any

start_time = time.perf_counter()

#^^ Define Data, Path, and Parameters

MODEL_FILENAME = './models/credit_score_xgb_model.joblib'
ARTIFACTS_FILENAME = './models/preprocessing_artifacts.json'
DATA_FILENAME = './data/test.csv'

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
    'Num_of_Delayed_Payment',
    'Changed_Credit_Limit',
    'Amount_invested_monthly', 
    'Monthly_Balance',
    'Outstanding_Debt'
]

numerical_cols_to_impute = numerical_obj_cols + ['Monthly_Inhand_Salary', 'Num_Credit_Inquiries']

nominal_cols = ['Occupation', 'Credit_Mix', 'Payment_Behaviour']

def load_data(csv_path):
    #^^ --- Data Loading and Initial Drop ---
    try:
        df = pd.read_csv(DATA_FILENAME)
    except FileNotFoundError:
        print(f"Error: {DATA_FILENAME} not found.")
        exit()

    return df

#^^ --- Clean and Impute Numerical Columns ---
def clean_non_numeric(series):
    #>-< CLean non-numeric strings ('_', ',') and convert to numeric
    series = series.astype(str).str.replace(r'[$,]', '', regex=True)
    series = pd.to_numeric(series, errors='coerce')
    return series

def preprocess_new_data(raw_data_df, artifacts):
    df = raw_data_df.copy()

    for col in numerical_obj_cols:
        df[col] = clean_non_numeric(df[col])

    df['Age'] = df['Age'].clip(lower=18, upper=100)
    df['Num_Credit_Card'] = df['Num_Credit_Card'].clip(lower=0, upper=10)
    df['Num_of_Loan'] = df['Num_of_Loan'].clip(lower=0, upper=10)
    df['Num_Credit_Inquiries'] = df['Num_Credit_Inquiries'].clip(lower=0, upper=100)
    
    medians = artifacts['imputation_medians']
 
    df.fillna(value=medians, inplace=True)

    payment_map = artifacts['payment_mapping']
    df['Payment_of_Min_Amount_Encoded'] = df['Payment_of_Min_Amount'].map(payment_map)
    df.drop('Payment_of_Min_Amount', axis=1, inplace=True)

    nominal_cols = artifacts['nominal_cols']
    df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    trained_features = artifacts['feature_names']
    
    df_aligned = df_encoded.reindex(columns=trained_features, fill_value=0)

    return df_aligned

def csv_to_list(csv_file):
    try:
        customer_list = csv_file.to_dict(orient='records')
        return customer_list
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return []

def load_artifact():
    #^^ --- Load Artifacts ---
    try:
        xgb_model = joblib.load(MODEL_FILENAME)
        print(f"Successfully loaded model from {MODEL_FILENAME}.")
    except FileNotFoundError:
        print(f"Error: Model file {MODEL_FILENAME} not found. Run train_and_save_model.py first.")
        exit()

    try:
        with open(ARTIFACTS_FILENAME, 'r') as f:
            artifacts = json.load(f)
        print(f"Successfully loaded artifacts from {ARTIFACTS_FILENAME}.")
    except FileNotFoundError:
        print(f"Error: Artifacts file {ARTIFACTS_FILENAME} not found. Run train_and_save_model.py first.")
        exit()

    return xgb_model, artifacts
    

def get_manual_input() -> Dict[str, Any]:
    """Prompts the user for key customer data and returns it as a dictionary."""
    print("\n--- Manual Data Entry ---")
    
    # Default values for non-critical features to ensure the DataFrame is complete
    customer_data = {
        'ID': 'Manual_01',
        'Customer_ID': 'CUS_Manual',
        'Name': 'N/A',
        'SSN': 'N/A',
        'Month': 'January',
        'Type_of_Loan': 'Unknown Loan',
        'Credit_History_Age': '1 Year and 0 Months',
        'Num_Bank_Accounts': 3,
        'Interest_Rate': 15.0,
        'Delay_from_due_date': 5,
        'Changed_Credit_Limit': 5.0,
        'Credit_Utilization_Ratio': 30.0,
        'Total_EMI_per_month': 150.0,
        'Outstanding_Debt': 800.0,
        'Amount_invested_monthly': 200.0,
        'Monthly_Inhand_Salary': 3000.0,
        'Num_Credit_Inquiries': 2,
    }
    
    # Collect key inputs for prediction
    key_inputs = {
        'Age': (int, "Age (e.g., 30): "),
        'Occupation': (str, "Occupation (e.g., Scientist, Doctor, Teacher): "),
        'Annual_Income': (float, "Annual Income (e.g., 50000.00): "),
        'Num_Credit_Card': (int, "Number of Credit Cards (e.g., 4): "),
        'Num_of_Loan': (int, "Number of Loans (e.g., 2): "),
        'Num_of_Delayed_Payment': (int, "Number of Delayed Payments (e.g., 3): "),
        'Credit_Mix': (str, "Credit Mix (Good, Standard, or Bad): "),
        'Payment_of_Min_Amount': (str, "Payment of Min Amount (Yes, No, or NM): "),
        'Payment_Behaviour': (str, "Payment Behaviour (e.g., High_spent_Small_value_payments): "),
        'Monthly_Balance': (float, "Monthly Balance (e.g., 450.50): "),
    }
    
    for key, (dtype, prompt) in key_inputs.items():
        while True:
            try:
                user_input = input(prompt).strip()
                if not user_input:
                    # Allow user to skip for less critical features, which will use the default/imputed value
                    if key in ['Occupation', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount']:
                         print(f"Using default or imputed value for {key}.")
                         break
                    print(f"Input for {key} cannot be empty. Please try again.")
                    continue
                
                # Special handling for numerical inputs that might contain symbols like '$' or ','
                if dtype in [int, float]:
                    user_input = user_input.replace('$', '').replace(',', '')
                
                value = dtype(user_input)
                customer_data[key] = value
                break
            except ValueError:
                print(f"Invalid input. Please enter a valid {dtype.__name__} for {key}.")

    return customer_data

if __name__ == "__main__":
    print("Start Credit Score Prediction Tool")
    
    mode = input("Select prediction mode (1 for CSV load, 2 for Manual Input): ").strip()
    
    converted_list = []
    
    if mode == '1':
        raw_data = load_data(DATA_FILENAME)
        if raw_data is not None:
            converted_list = csv_to_list(raw_data)
        else:
            print("Cannot proceed without data. Exiting.")
            exit()
    elif mode == '2':
        customer_data = get_manual_input()
        converted_list = [customer_data]
        print("Manual record prepared for prediction.")
    else:
        print("Invalid mode selected. Defaulting to CSV load.")
        raw_data = load_data(DATA_FILENAME)
        if raw_data is not None:
            converted_list = csv_to_list(raw_data)
        else:
            print("Cannot proceed without data. Exiting.")
            exit()

    xgb_model, artifacts = load_artifact()

    inverse_score_map = {v: k for k, v in artifacts['target_mapping'].items()}

    predicted_scores = []

    for i, customer_data in enumerate(converted_list):
        print(f"\n--- Predicting Credit Score for Customer {i+1} ---")
        
        #>-< Convert to a DataFrame
        raw_df = pd.DataFrame([customer_data])

        X_new = preprocess_new_data(raw_df, artifacts)

        prediction = xgb_model.predict(X_new)[0]
        predicted_score = inverse_score_map.get(prediction, 'Unknown')

        predicted_scores.append(predicted_score)

        print(f"Input features (preprocessed shape): {X_new.shape}")
        print(f"Model Prediction (Encoded): {int(prediction)}")
        print(f"Model Prediction (Decoded): {predicted_score}")
        with open('test.txt', 'a') as f:
            f.write("\n")
            f.write(predicted_score)
    
    
    
    
    
    
    
    
    
    
    
    predicted_csv = raw_data
    predicted_csv['predicted'] = predicted_scores

    try:
        predicted_csv.to_csv('data/result.csv', index=False)
    except Exception as e:
        print(e)
        
    #>-< Calculate execution time
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"Execution time: {duration:.4f} seconds")