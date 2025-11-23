import numpy as np
from sklearn.linear_model import LinearRegression

peso = np.array([60.0, 65.0, 72.3, 75.0, 80.0])
altura = np.array([1.60, 1.65, 1.70, 1.73, 1.80])

modelo = LinearRegression()
modelo.fit(altura.reshape(-1,1), peso)
y_pred = modelo.predict(altura.reshape(-1,1))

test_altura = np.array([1.58, 1.62, 1.69, 1.76, 1.82]).reshape(-1, 1)

predicciones = modelo.predict(test_altura)
rss = np.sum((peso - y_pred)**2)

print("Alturas de prueba:", test_altura.flatten())
print("Predicciones de peso:", predicciones)
print("RSS:", rss)