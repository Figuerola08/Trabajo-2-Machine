import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Cargar datos
file_path = r"C:\Users\ROG STRIX\Documents\cirrhosis.csv"
df = pd.read_csv(file_path)

# Imputar valores faltantes
imputer_numeric = SimpleImputer(strategy='mean')
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_features] = imputer_numeric.fit_transform(df[numeric_features])

imputer_categorical = SimpleImputer(strategy='most_frequent')
categorical_features = df.select_dtypes(include=['object']).columns
df[categorical_features] = imputer_categorical.fit_transform(df[categorical_features])

# Codificación de variables categóricas
df_encoded = pd.get_dummies(df)

# Imprime las columnas para verificar la correcta
print("Columnas en DataFrame codificado:", df_encoded.columns.tolist())

# Identifica la columna objetivo correcta, ajustando si es necesario
target_column = 'Status_C'  # Ajusta según los nombres de columna impresos

if target_column not in df_encoded.columns:
    raise ValueError(f"Columna {target_column} no encontrada. Verifica los nombres de las columnas.")

X = df_encoded.drop(target_column, axis=1)
y = df_encoded[target_column].astype(int)

# Selección de características usando RandomForest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)
important_features = X.columns[rf_model.feature_importances_ > np.median(rf_model.feature_importances_)]

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(X[important_features], y, test_size=0.2, random_state=42)

# Configuración de la búsqueda de hiperparámetros
param_grid = {
    'C': [0.1, 10, 100],
    'gamma': [1, 0.01],
    'kernel': ['linear', 'rbf']
}
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid_search.fit(X_train, y_train)

# Evaluación del modelo
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Resultados optimizados: \nPrecisión: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")
print("Mejores hiperparámetros:", grid_search.best_params_)