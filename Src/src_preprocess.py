import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ---------- Utility functions ----------
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


# ---------- Feature Engineering ----------
class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Извлекает часы, день недели и месяц из столбца datetime."""
    def __init__(self, column='pickup_datetime'):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        ts = pd.to_datetime(X_[self.column])
        X_['hour']     = ts.dt.hour
        X_['dayofweek']= ts.dt.dayofweek
        X_['month']    = ts.dt.month
        return X_.drop(columns=[self.column])


class DistanceCalculator(BaseEstimator, TransformerMixin):
    """Добавляет столбец расстояния (Haversine) между координатами."""
    def __init__(self, lat1='pickup_latitude', lon1='pickup_longitude',
                 lat2='dropoff_latitude', lon2='dropoff_longitude'):
        self.lat1, self.lon1 = lat1, lon1
        self.lat2, self.lon2 = lat2, lon2

    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        R = 6371.0  # радиус Земли в км
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)

        a = np.sin(dphi/2.0)**2 + \
            np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['distance_km'] = self.haversine(
            X_[self.lon1], X_[self.lat1],
            X_[self.lon2], X_[self.lat2]
        )
        # можно удалить исходные координаты
        return X_.drop(columns=[self.lat1, self.lon1,
                                self.lat2, self.lon2])


class DropMissing(BaseEstimator, TransformerMixin):
    """Обрабатывает пропуски: числовые – среднее, категориальные – mode."""
    def __init__(self, strategy='median'):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in X_.select_dtypes(include=['float64', 'int64']).columns:
            X_[col] = X_[col].fillna(X_[col].median())
        for col in X_.select_dtypes(include='object').columns:
            X_[col] = X_[col].fillna(X_[col].mode()[0])
        return X_
