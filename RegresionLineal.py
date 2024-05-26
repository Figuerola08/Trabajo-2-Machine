import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Cargar datos
file_path = "cirrhosis.csv"
df = pd.read_csv(file_path)

# Verificar datos faltantes
print("Datos faltantes por columna:\n", df.isnull().sum())

# Imputar valores faltantes para todas las columnas numéricas detectadas
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
imputer_numeric = SimpleImputer(strategy='mean')
df[numeric_features] = imputer_numeric.fit_transform(df[numeric_features])

# Imputar valores faltantes para todas las columnas categóricas detectadas
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
imputer_categorical = SimpleImputer(strategy='most_frequent')
df[categorical_features] = imputer_categorical.fit_transform(df[categorical_features])

# Transformar 'Status' en variable numérica antes de codificar
status_mapping = {'C': 0, 'CL': 1, 'D': 2}
df['Status'] = df['Status'].map(status_mapping)

# Separar la columna 'Status' antes de codificar las otras variables categóricas
status = df['Status']
df = df.drop(columns=['Status'])

# Codificación de variables categóricas usando One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=[col for col in categorical_features if col != 'Status'])

# Añadir nuevamente la columna 'Status' al DataFrame codificado
df_encoded['Status'] = status

# Verificar si la columna 'Status' está presente
if 'Status' not in df_encoded.columns:
    raise KeyError("La columna 'Status' no se encuentra en el DataFrame después de la codificación. Verifica el nombre de la columna en el archivo CSV.")

# Asumiendo que 'Status' es la variable objetivo
X = df_encoded.drop(['Status'], axis=1)
y = df_encoded['Status']

# Selección de variables relevantes usando RandomForestRegressor para Feature Importance
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)
feature_importances = rf_model.feature_importances_
important_features = [feature for feature, importance in zip(X.columns, feature_importances) if importance > np.mean(feature_importances)]

X_train, X_test, y_train, y_test = train_test_split(X[important_features], y, test_size=0.2, random_state=42)

# Entrenamiento de un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción y evaluación del modelo
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

residuals = y_test - y_pred

# Crear la figura de la distribución de errores
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='blue')
plt.title('Distribución de Errores del Modelo de Regresión Lineal')
plt.xlabel('Error Residual')
plt.ylabel('Frecuencia')
plt.axvline(x=0, color='red', linestyle='--')
plt.grid(True)
plt.show()

print(f"Error cuadrático medio (MSE): {mse}")
print(f"Coeficiente de determinación (R²): {r2}")
