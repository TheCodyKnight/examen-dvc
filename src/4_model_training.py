import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor

best_params = joblib.load('models/data/best_params.pkl')

X_train_scaled = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

model = DecisionTreeRegressor(**best_params)
model.fit(X_train_scaled, y_train)
joblib.dump(model, 'models/models/regressor.pkl')

print("4_model_training complete")