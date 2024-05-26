import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Cargar datos
file_path = "cirrhosis.csv"
df = pd.read_csv(file_path)

# Verificar las columnas del DataFrame
print("Columnas del DataFrame:", df.columns)

# Eliminar la columna ID ya que no aporta información relevante
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Verificar y eliminar filas duplicadas
df.drop_duplicates(inplace=True)

# Imputar valores faltantes
imputer_numeric = SimpleImputer(strategy='mean')
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_features] = imputer_numeric.fit_transform(df[numeric_features])

imputer_categorical = SimpleImputer(strategy='most_frequent')
categorical_features = df.select_dtypes(include=(['object'])).columns
df[categorical_features] = imputer_categorical.fit_transform(df[categorical_features])

# Transformar 'Status' en variable numérica antes de codificar
status_mapping = {'C': 0, 'CL': 1, 'D': 2}
df['Status'] = df['Status'].map(status_mapping)

# Codificación de variables categóricas
df_encoded = pd.get_dummies(df)

# Verificar si la columna 'Status' está presente
if 'Status' not in df_encoded.columns:
    raise KeyError("La columna 'Status' no se encuentra en el DataFrame después de la codificación. Verifica el nombre de la columna en el archivo CSV.")

# Identificar la columna objetivo correcta
target_column = 'Status'

# Eliminar características altamente correlacionadas con la columna objetivo que podrían causar fuga de datos
potential_leakage_features = []  # Añadir otras características si es necesario
X = df_encoded.drop(columns=potential_leakage_features + [target_column])
y = df_encoded[target_column].astype(int)

# Verificar el balance de clases
print("Distribución de clases en la columna objetivo antes del balanceo:")
print(y.value_counts())

# Balancear las clases usando SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Distribución de clases en la columna objetivo después del balanceo:")
print(y_resampled.value_counts())

# Selección de características usando RandomForest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)
important_features = X.columns[rf_model.feature_importances_ > np.median(rf_model.feature_importances_)]

# Normalización de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled[important_features])

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Configuración de la búsqueda de hiperparámetros
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=0, cv=10)
grid_search.fit(X_train, y_train)

# Evaluación del modelo
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Resultados optimizados: \nPrecisión: {accuracy}\nPrecisión: {precision}\nRecall: {recall}\nF1 Score: {f1}")
print("Mejores hiperparámetros:", grid_search.best_params_)

# Validación cruzada adicional
scores = cross_val_score(grid_search.best_estimator_, X_scaled, y_resampled, cv=10)
print("Scores de validación cruzada:", scores)
print("Precisión media de validación cruzada:", scores.mean())

# Curvas de aprendizaje
train_sizes, train_scores, test_scores = learning_curve(grid_search.best_estimator_, X_scaled, y_resampled, cv=10)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Entrenamiento")
plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validación")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Precisión")
plt.legend()
plt.title("Curvas de Aprendizaje")
plt.show()

# Análisis de importancia de características
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("Importancias de características:")
for f in range(len(important_features)):
    print(f"{important_features[f]}: {importances[indices[f]]}")
