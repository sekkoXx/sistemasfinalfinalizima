import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ===========================
# 1. CARGAR DATOS
# ===========================
A = np.load("A.npy")          # Puntos (1000 x 2)
labels_true = np.load("_.npy")  # Etiquetas reales

print("Shape de los datos:", A.shape)
print("Cantidad de datos:", len(A))
print("Ejemplos de puntos:\n", A[:5])
print("Ejemplos de etiquetas reales:\n", labels_true[:10])

# ===========================
# FIGURA A: Scatter sin colores
# ===========================
plt.figure(figsize=(6,5))
plt.scatter(A[:,0], A[:,1], s=20, color="red")
plt.title("Figura A – Datos sin colorear")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# ===========================
# FIGURA B: Scatter coloreado por etiquetas reales
# ===========================
plt.figure(figsize=(6,5))
plt.scatter(A[:,0], A[:,1], c=labels_true, cmap="viridis", s=20)
plt.title("Figura B – Datos coloreados por etiqueta real")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# ===========================
# 3. K-MEANS
# ===========================
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(A)

centroides = kmeans.cluster_centers_
labels_kmeans = kmeans.labels_

print("\nCentroides encontrados:\n", centroides)

# ===========================
# FIGURA C – KMeans + Centroides
# ===========================
plt.figure(figsize=(6,5))
plt.scatter(A[:,0], A[:,1], c=labels_kmeans, cmap="rainbow", s=20)
plt.scatter(centroides[:,0], centroides[:,1], 
            marker="*", color="black", s=300, label="Centroides")
plt.title("Figura C – KMeans + Centroides")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

# ===========================
# 5. PREDICCIÓN TEST SET
# ===========================
test = np.array([[2,5], [3.2,6.5], [7,2.5], [9,3.2], [9,-6], [11,-8]])
pred = kmeans.predict(test)

print("\nData Test:")
print(test)
print("\nPredicciones de clase:")
print(pred)

# Mostrar resultados ordenados
for i, punto in enumerate(test):
    print(f"Punto {punto} → Clase {pred[i]}")
