import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

X_train_scaled = pd.read_csv('../data/processed/X_train_scaled.csv')
y_train = pd.read_csv('../data/processed/y_train.csv')
model = DecisionTreeRegressor()
parameter_grid = {
    'max_depth': [3, 5, 7, 10, 15, 18, 20, None],
    'min_samples_split': [2, 5, 10, 13, 15],
    'min_samples_leaf': [1, 2, 3, 4, 6, 8],
}

grid_search_cv = GridSearchCV(model, parameter_grid)
grid_search_cv.fit(X_train_scaled, y_train)
best_params = grid_search_cv.best_params_
joblib.dump(best_params, '../models/data/best_params.pkl')

print(best_params)
print("3_gridsearch complete")