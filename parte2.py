import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# ===========================================
# 1. CARGAR DATOS
# ===========================================
X = np.load("X.npy")             # Datos de 1000 puntos
labels_true = np.load("Kmeans.npy")   # Etiquetas reales (tomas si lees esto le puse mal el nombre al archivo mira la carpeta meanshift)

print("Shape de X:", X.shape)
print("Cantidad de datos:", len(X))
print("Ejemplos de X:\n", X[:5])
print("Ejemplos de etiquetas reales:\n", labels_true[:10])

# ===========================================
# FIGURA A: Scatter simple
# ===========================================
plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], s=20, color="red")
plt.title("Figura A – Datos sin colorear (MeanShift)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# ===========================================
# FIGURA B: Scatter coloreado por etiquetas reales
# ===========================================
plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], c=labels_true, cmap="viridis", s=20)
plt.title("Figura B – Datos coloreados por etiqueta real")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

# ===========================================
# 3. MEANSHIFT (detección automática de cantidad de clusters)
# ===========================================

# Bandwidth adaptativa
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels_ms = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters = len(np.unique(labels_ms))

print("\nClusters detectados por MeanShift:", n_clusters)
print("Centroides:\n", cluster_centers)

# ===========================================
# FIGURA C – MeanShift + Centroides
# ===========================================
plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], c=labels_ms, cmap="rainbow", s=20)
plt.scatter(cluster_centers[:,0], cluster_centers[:,1],
            marker='*', s=300, color='black', label="Centroides")
plt.title("Figura C – MeanShift + Centroides")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

# ===========================================
# 5. PREDICCIÓN TEST SET (MeanShift permite predict)
# ===========================================
# MeanShift no trae predict(), pero podemos asignar un cluster según distancia al centroide más cercano
test = np.array([[-7,-6], [1.5,-6.5], [7.9,0.5], [5.5,10]])

def predict_meanshift(points, centers):
    preds = []
    for p in points:
        d = np.linalg.norm(centers - p, axis=1)
        preds.append(np.argmin(d))
    return np.array(preds)

pred = predict_meanshift(test, cluster_centers)

print("\nData Test:")
print(test)
print("\nPredicciones (clusters más cercanos):")
print(pred)

for i, p in enumerate(test):
    print(f"Punto {p} → Clase {pred[i]}")
