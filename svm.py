import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# Cargar datos
file_path = r"C:\Users\ROG STRIX\Documents\cirrhosis.csv"
df = pd.read_csv(file_path)

# Verificar y manejar datos faltantes
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Detectar y manejar outliers
def remove_outliers(X, y):
    iso_forest = IsolationForest(contamination=0.05)
    yhat = iso_forest.fit_predict(X)
    mask = yhat != -1
    return X[mask], y[mask]

# Asumo que 'Status' es tu variable objetivo
X = df.drop('Status', axis=1)  # Cambia 'Status' al nombre real de tu columna objetivo si es diferente
y = df['Status']  # Cambia 'Status' al nombre real de tu columna objetivo si es diferente

# Aplicar preprocesamiento
X_processed = preprocessor.fit_transform(X)

# Remover outliers
X_processed, y = remove_outliers(X_processed, y)

# Selección de variables
feature_selector = SelectKBest(score_func=chi2, k=10)

# Pipeline final
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', feature_selector),
    ('classifier', SVC())
])

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Búsqueda de hiperparámetros
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': [1, 0.1, 0.01, 0.001],
    'classifier__kernel': ['linear', 'rbf']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)

# Entrenamiento y evaluación
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Mejores hiperparámetros:")
print(grid_search.best_params_)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
