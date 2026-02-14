import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# регрессоры
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# --------- Общая функция ----------
def evaluate_pipeline(pipe: Pipeline, X_train, y_train,
                      X_val, y_val):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    mae  = mean_absolute_error(y_val, preds)
    r2   = r2_score(y_val, preds)
    return mae, r2

# --------- Пайплайн ----------
def build_pipeline(preprocessor, regressor):
    return Pipeline(steps=[
        ('preprocess', preprocessor),
        ('regressor',  regressor)
    ])

# --------- Создание предобработки ----------
def create_preprocessor(numeric_features, categorical_features):
    numeric_tf = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_tf = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer(
        transformers=[
            ('num', numeric_tf, numeric_features),
            ('cat', categorical_tf, categorical_features)
        ],
        remainder='drop'
    )

# --------- Модели ----------
def get_models():
    return {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),
        'DecisionTree': DecisionTreeRegressor(max_depth=None, random_state=42),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=20,
            min_samples_split=2, min_samples_leaf=1,
            random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'XGBRegressor': xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    }

# --------- GridSearch ----------
def tune_xgb(X, y):
    param_grid = {
        'n_estimators': [200, 400, 600],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4,6,8],
        'subsample': [0.7,0.8,0.9]
    }
    gs = GridSearchCV(
        xgb.XGBRegressor(objective='reg:squarederror',
                         random_state=42,
                         n_jobs=-1),
        param_grid, cv=5, scoring='neg_mean_absolute_error',
        verbose=1, n_jobs=-1
    )
    gs.fit(X, y)
    return gs.best_estimator_
