"""_summary_
    Data Treatement class has as goal to analyse, treat and possibly clean any dirty data
    in the targeted database    
"""
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def eval_data(df):
    # CHECK LABEL DISTRIBUTION
    print("="*60)
    print("DATA DISTRIBUTION CHECK")
    print("="*60)
    
    # Assuming last column is the label (0=normal, 1=attack)
    label_counts = df.iloc[:, -1].value_counts()
    print(f"\nLabel distribution:")
    print(label_counts)
    print(f"\nPercentages:")
    print(label_counts / len(df) * 100)
    
    total = len(df)
    class_0 = (df.iloc[:, -1] == 0).sum()
    class_1 = (df.iloc[:, -1] == 1).sum()
    
    print(f"\nClass 0 (normal): {class_0} ({class_0/total*100:.2f}%)")
    print(f"Class 1 (attack): {class_1} ({class_1/total*100:.2f}%)")
    
    imbalance_ratio = max(class_0, class_1) / min(class_0, class_1)
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 10:
        print("⚠️  WARNING: Severe class imbalance detected!")
        print("   Your model may learn to always predict the majority class.")
    
    print("="*60)
    
def check_for_leakage(df_train, df_test):
    """
    Check for potential data leakage
    """
    print("\n" + "="*60)
    print("DATA LEAKAGE CHECK")
    print("="*60)
    
    # Check 1: Temporal overlap
    train_time_range = (df_train['stime'].min(), df_train['stime'].max())
    test_time_range = (df_test['stime'].min(), df_test['stime'].max())
    
    print(f"\nTrain time range: {train_time_range}")
    print(f"Test time range: {test_time_range}")
    
    if train_time_range[1] >= test_time_range[0]:
        print("⚠️  WARNING: Temporal overlap detected!")
    else:
        print("✓ No temporal overlap")
    
    # Check 2: Feature overlap (e.g., same IPs)
    train_saddrs = set(df_train['daddr'].unique())
    test_saddrs = set(df_test['daddr'].unique())
    
    overlap_ratio = len(train_saddrs & test_saddrs) / len(test_saddrs)
    print(f"\nDestination IP overlap: {overlap_ratio*100:.2f}%")
    
    if overlap_ratio > 0.9:
        print("⚠️  WARNING: High IP overlap - model may memorize specific IPs")
    
    # Check 3: Label distribution similarity
    train_attack_rate = df_train.iloc[:, -1].mean()
    test_attack_rate = df_test.iloc[:, -1].mean()
    
    print(f"\nTrain attack rate: {train_attack_rate*100:.2f}%")
    print(f"Test attack rate: {test_attack_rate*100:.2f}%")
    
    if abs(train_attack_rate - test_attack_rate) > 0.2:
        print("⚠️  WARNING: Very different label distributions")
    
    print("="*60)

def encoded(df):
    df = df.copy()
    df = df.fillna(0)
    cols_to_dec = ["sourceIP", "destinationIP", "Protocol"]
    for col in df:
        if col in cols_to_dec:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def describe_database(df):
    print("\n"+"="*60)
    print("DATASET TYPE CLASSIFICATION")
    print("="*60)
    
    for col in df.columns:
        # Get pandas dtype for the column
        pandas_dtype = df[col].dtype
        
        # Get Python type of first non-null value
        first_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        python_type = type(first_value).__name__ if first_value is not None else "NoneType"
        
        # Determine status
        if pandas_dtype == 'object':
            status = "⚠️  WARNING: object type!"
        elif df[col].isnull().any():
            status = f"⚠️  Has {df[col].isnull().sum()} nulls"
        else:
            status = "OK"
        
        print(f"{col:<20} {str(pandas_dtype):<15} {python_type:<15} {status}")
    print("="*60)
    return

if __name__ == '__main__':
    df = pd.read_csv("../dataset_sdn.csv")
    print(f"taille totale des donnes {len(df)}")
    df = encoded(df)
    describe_database(df)
    
    # ----------- data base test functions --------------------------------
    #quart = len(df) // 4
    #df_quart = df.iloc[:quart]
    #df_test = df.iloc[quart:2*quart]
    ##verify_type(df)
    #eval_data(df)
    #check_for_leakage(df_quart, df_test)
    # ---------------------------------------------------------------------