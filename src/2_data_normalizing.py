import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')

scaler = joblib.load('models/data/scaler.pkl')
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#converting to df here because later scripts threw errors with the numpy array (commit only in part!)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_train_scaled.to_csv('data/processed/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/processed/X_test_scaled.csv', index=False)

print("2_data_normalizing complete")