import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mineral_processing = pd.read_csv('data/raw/raw.csv') 
y = mineral_processing['silica_concentrate']
X = mineral_processing.drop(columns=['silica_concentrate'])

#drop date column as it is causing execution errors, also likely not relevant for model training
X = mineral_processing.drop(columns=["date"])
X_train, X_test, y_train, y_test = train_test_split(X, y)

X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

#I read fitting the scaler early and saving it for reproducability is good practise, so here we go good practising :)
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'models/data/scaler.pkl')

print("1_data_splitting complete")