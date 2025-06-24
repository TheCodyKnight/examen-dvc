import pandas as pd
import os
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

X_test_scaled = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/processed/y_test.csv')
model = joblib.load('models/models/regressor.pkl')
y_predict = model.predict(X_test_scaled)

#not specifically asked for in the "Submission" section, but it was mentioned that "we will have a new dataset in data containing the predictions"
#I got a bit confused by the term "dataset", but I figured this would be what is expected:
predictions = pd.DataFrame({'Actual': y_test.values.ravel(), 'Predicted': y_predict})
#got an error I didn't know without the values.ravel, I learned it is due to the csv imports and exports I am doing
predictions.to_csv('models/data/prediction.csv', index=False)

mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
medae = median_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
metrics = {'mean_squared_error': mse, 'mean_absolute_error': mae, 'median_absolute_error': medae, 'r2_score': r2}
print(metrics)
f = open('metrics/scores.json', 'w')
json.dump(metrics, f)
f.close()

print("5_model_evaluating complete")