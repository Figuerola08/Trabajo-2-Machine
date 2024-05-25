import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Cargar datos
file_path = r"C:\Users\ROG STRIX\Documents\cirrhosis.csv"
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

# Codificación de variables categóricas usando One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_features)

# Detección y manejo de outliers
for col in numeric_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df_encoded[col])
    plt.title(f'Detección de outliers en {col}')
    plt.show()

    # Calcular percentiles
    Q1 = df_encoded[col].quantile(0.25)
    Q3 = df_encoded[col].quantile(0.75)
    IQR = Q3 - Q1

    # Definir límites
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Reemplazar outliers por los límites
    df_encoded[col] = np.where(df_encoded[col] < lower_bound, lower_bound, df_encoded[col])
    df_encoded[col] = np.where(df_encoded[col] > upper_bound, upper_bound, df_encoded[col])

# Asumiendo que 'Bilirubin' es la variable objetivo
if 'Bilirubin' in df_encoded.columns:
    X = df_encoded.drop(['Bilirubin'], axis=1)
    y = df_encoded['Bilirubin']
    
    # Selección de variables relevantes usando RandomForestRegressor para Feature Importance
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X, y)
    feature_importances = rf_model.feature_importances_
    important_features = [feature for feature, importance in zip(X.columns, feature_importances) if importance > np.mean(feature_importances)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamiento de un modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicción y evaluación del modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Error cuadrático medio (MSE): {mse}")
    print(f"Coeficiente de determinación (R²): {r2}")

else:
    print("Columna 'Bilirubin' no encontrada en el DataFrame. Asegúrate de que el nombre esté escrito correctamente.")
