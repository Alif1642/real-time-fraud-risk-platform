"""
Generate sample IEEE-CIS fraud detection data for development/testing.
Run this script to create train_transaction.csv and train_identity.csv in data/raw/
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_sample_data(n_rows: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate mock IEEE-CIS format data."""
    np.random.seed(42)
    
    # Transaction data
    transaction_data = {
        'TransactionID': range(n_rows),
        'isFraud': np.random.binomial(1, 0.033, n_rows),  # ~3.3% fraud rate (realistic)
        'TransactionDT': np.random.randint(86400, 172800, n_rows),
        'TransactionAmt': np.random.exponential(100, n_rows).round(2),
        'ProductCD': np.random.choice(['W', 'H', 'S', 'C', 'R'], n_rows),
        'card1': np.random.randint(10000, 50000, n_rows),
        'card2': np.random.randint(100, 500, n_rows),
        'card3': np.random.randint(100, 600, n_rows),
        'card4': np.random.choice(['visa', 'mastercard', 'amex'], n_rows),
        'card5': np.random.randint(100, 500, n_rows),
        'card6': np.random.choice(['debit', 'credit'], n_rows),
        'addr1': np.random.randint(1000, 999000, n_rows),
        'addr2': np.random.randint(1, 60, n_rows),
        'dist1': np.random.exponential(50, n_rows).round(2),
        'dist2': np.random.exponential(100, n_rows).round(2),
        'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', np.nan], n_rows),
        'R_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'outlook.com', np.nan], n_rows),
    }
    
    # Identity data (subset of transactions)
    identity_data = {
        'TransactionID': range(n_rows // 2),
        'id_01': np.random.randint(1, 100, n_rows // 2),
        'id_02': np.random.randint(1, 100, n_rows // 2),
        'id_11': np.random.choice(['NotFound', 'Found', 'Unknown'], n_rows // 2),
        'id_12': np.random.choice(['NotFound', 'Found', 'Unknown'], n_rows // 2),
        'DeviceType': np.random.choice(['desktop', 'mobile'], n_rows // 2),
        'DeviceInfo': np.random.choice(['Windows', 'Mac', 'Android', 'iOS'], n_rows // 2),
    }
    
    df_transaction = pd.DataFrame(transaction_data)
    df_identity = pd.DataFrame(identity_data)
    
    return df_transaction, df_identity


def main():
    # Create data/raw directory if it doesn't exist
    data_raw_dir = Path('data/raw')
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print("Generating sample data...")
    df_transaction, df_identity = generate_sample_data(n_rows=10000)
    
    # Save to CSV
    transaction_path = data_raw_dir / 'train_transaction.csv'
    identity_path = data_raw_dir / 'train_identity.csv'
    
    df_transaction.to_csv(transaction_path, index=False)
    df_identity.to_csv(identity_path, index=False)
    
    print(f"✓ Created {transaction_path} ({len(df_transaction)} rows)")
    print(f"✓ Created {identity_path} ({len(df_identity)} rows)")
    print(f"\nDataset info:")
    print(f"  - Fraud cases: {df_transaction['isFraud'].sum()} ({df_transaction['isFraud'].mean()*100:.1f}%)")
    print(f"  - Transaction range: ${df_transaction['TransactionAmt'].min():.2f} - ${df_transaction['TransactionAmt'].max():.2f}")


if __name__ == '__main__':
    main()
