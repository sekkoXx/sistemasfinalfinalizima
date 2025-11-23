import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 1. Cargar puntos_3d.npy en la variable d3
# ============================================================

# Se carga el archivo .npy y queda guardado como un arreglo numpy.
d3 = np.load("puntos_3d.npy")

# Mostrar estructura para verificar que son puntos 3D.
print("Forma del arreglo:", d3.shape)
print("Primeros puntos:\n", d3[:5])

# ============================================================
# 2. Crear un DataFrame vacío y luego asignarle las columnas
# ============================================================

# La pauta pide explícitamente usar: df = pd.DataFrame()
df = pd.DataFrame(d3)

# Ahora asignamos manualmente los nombres de las columnas
df.columns = ["X", "Y", "Z"]

print("\nDataFrame creado:")
print(df.head())

# ============================================================
# 3. Calcular el centroide del conjunto de puntos
# ============================================================

# El centroide es el promedio de las coordenadas X, Y y Z.
centroide = df.mean()

print("\nCentroide calculado:")
print(centroide)

# ============================================================
# 4. Graficar la nube de puntos 3D y el centroide en azul
# ============================================================

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Graficar todos los puntos en color rojo (como en la pauta).
ax.scatter(df["X"], df["Y"], df["Z"], color="red", s=25, label="Puntos")

# Graficar el centroide en color azul y más grande.
ax.scatter(centroide["X"], centroide["Y"], centroide["Z"],
           color="blue", s=200, marker="o", label="Centroide")

# Etiquetas y título del gráfico.
ax.set_title("Nube de puntos 3D con centroide")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.show()
