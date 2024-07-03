#librerias
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

# Ruta local del archivo CSV en la carpeta "Integrador"
local_file_path = r'C:\Users\Arturo Marin\Desktop\RedesNeuronales\creditcard.csv'

# Carga los datos desde el archivo local
data = pd.read_csv(local_file_path)

# separacion de caracteristicas (x) y etiquetas (y)
x = data.drop('Class', axis=1)
y = data['Class']

# division de los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# escalar las caracteristicas para normalizar 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# construccion del modelo de red neuronal 

# define el modelo 

model = Sequential ([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') 
])

# compilacion del modelo 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# entrenamiento del modelo
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# realizacion de predicciones con el modelo sobre el conjunto de prueba 

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5) # convertir las probabilidades en etiquetas binarias

# generar la matriz de confusion 

conf_matrix = confusion_matrix(y_test, y_pred)

# mostrar la matriz de confusion 

print("Matriz de Confusion: ")
print(conf_matrix)

# Aplicar precision como metrica para medir el rendimiento del modelo

from sklearn.metrics import accuracy_score

precision = accuracy_score(y_test, y_pred)

print(f"Precision del modelo: {precision}")

#Visualizar el desbalanceo del Dataset
import matplotlib.pyplot as plt

# Calcular la cantidad de muestras por clase
class_counts = data['Class'].value_counts()

# Mostrar la cantidad de datos por clase
for class_name, count in class_counts.items():
    print(f'Clase: {class_name}, Cantidad de Datos: {count}')

# Crear el gráfico de barras
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color=['blue', 'red'])
plt.title('Desbalance de Clases')
plt.xlabel('Clase')
plt.ylabel('Cantidad de Muestras')
plt.xticks([0, 1], ['Normal', 'Fraud'])
plt.show()

#Que se observan en los datos visualizados?
from collections import Counter

# Calcular la distribución de clases actual
counter = Counter(y)
print("Distribución de clases antes del submuestreo:", counter)

# Identificar la cantidad mínima de muestras entre las clases
clase_menor = min(counter, key=counter.get)
cantidad_menor = counter[clase_menor]

# Realizar el submuestreo para igualar la cantidad de muestras de la clase minoritaria
clases = list(counter.keys())
for clase in clases:
    indices_clase = np.where(y == clase)[0]
    indices_muestra_a_descartar = indices_clase[cantidad_menor:]
    x_submuestreo = np.delete(x, indices_muestra_a_descartar, axis=0)
    y_submuestreo = np.delete(y, indices_muestra_a_descartar)

# Verificar la distribución de clases después del submuestreo
counter_submuestreo = Counter(y_submuestreo)
print("Distribución de clases después del submuestreo:", counter_submuestreo)

# Calcular la matriz de correlación
correlation_matrix = data.corr()

# Generar un mapa de calor
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap="RdYlGn")
plt.title('Mapa de Calor')
plt.show()

# Encontrar la correlación con la variable de salida
correlation_with_output = correlation_matrix['Class'].sort_values(ascending=False)
print("Correlación de las variables con la clase de salida:\n", correlation_with_output)