import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 7.- Filtro de Kalman para predecir la posición de una partícula
# ============================================================

# 1) Definimos el eje temporal
dt = 0.1                  # paso de tiempo
t = np.arange(0, 10, dt)   # de 0 a 10 segundos
N = len(t)

# 2) Definimos una trayectoria "real" (puedes adaptarla a la de la guía)
#    Ejemplo: movimiento con aceleración constante (parabólico)
a = 0.5      # aceleración
v0 = 1.0     # velocidad inicial
x0 = 0.0     # posición inicial

x_real = 0.5 * a * t**2 + v0 * t + x0  # trayectoria real de la partícula

# 3) Generamos mediciones ruidosas (lo que "ve" el sensor)
np.random.seed(42)  # para que los resultados sean reproducibles
ruido_std = 2.0
mediciones = x_real + np.random.normal(0, ruido_std, size=N)

# ------------------------------------------------------------
# Filtro de Kalman 1D (posición) con modelo simple:
#   x_k = x_{k-1}              (modelo de movimiento muy simple)
#   z_k = x_k + ruido
# ------------------------------------------------------------

# Parámetros del Kalman
Q = 1e-2    # Varianza del proceso (incertidumbre del modelo)
R = ruido_std**2  # Varianza de la medición (sensor)

# Variables de Kalman
x_kalman = np.zeros(N)   # estimación filtrada
P = np.zeros(N)          # varianza del error de estimación

# Valores iniciales
x_kalman[0] = mediciones[0]  # empezamos "creyendo" la primera medición
P[0] = 1.0                   # incertidumbre inicial

for k in range(1, N):
    # 1) Predicción
    x_pred = x_kalman[k-1]       # x_k|k-1
    P_pred = P[k-1] + Q          # P_k|k-1

    # 2) Actualización (corrección)
    K = P_pred / (P_pred + R)    # Ganancia de Kalman
    x_kalman[k] = x_pred + K * (mediciones[k] - x_pred)
    P[k] = (1 - K) * P_pred

# ============================================================
# 8.- Filtro Complementario aplicado a la salida del Kalman
# ============================================================

# Función de filtro complementario:
# Combina Kalman (lento pero suave) con medición (rápida pero ruidosa)
def filtro_complementario(med, kalman, alpha):
    """
    med: mediciones ruidosas z_k
    kalman: salida del Filtro de Kalman
    alpha: peso del Kalman (0 <= alpha <= 1)
    """
    return alpha * kalman + (1 - alpha) * med

# Probamos varios alpha para ver cuál se comporta mejor
alphas = [0.2, 0.5, 0.8]

comp_por_alpha = {}
mse_por_alpha = {}

for alpha in alphas:
    x_comp = filtro_complementario(mediciones, x_kalman, alpha)
    comp_por_alpha[alpha] = x_comp
    
    # Error cuadrático medio respecto a la trayectoria real
    mse = np.mean((x_real - x_comp)**2)
    mse_por_alpha[alpha] = mse

# Elegimos el mejor alpha (menor MSE)
alpha_mejor = min(mse_por_alpha, key=mse_por_alpha.get)
x_comp_mejor = comp_por_alpha[alpha_mejor]

print("MSE por alpha:")
for a, mse in mse_por_alpha.items():
    print(f"  alpha = {a}: MSE = {mse:.4f}")
print(f"\nAlpha seleccionado (mejor): {alpha_mejor}")

# ============================================================
# GRÁFICOS PARA EL INFORME Y DEFENSA
# ============================================================

# --- Figura 1: Trayectoria real vs mediciones vs Kalman ---
plt.figure(figsize=(10,5))
plt.plot(t, x_real, label="Trayectoria real", linewidth=2)
plt.scatter(t, mediciones, s=10, alpha=0.5, label="Mediciones ruidosas")
plt.plot(t, x_kalman, label="Filtro de Kalman", linewidth=2)
plt.title("Filtro de Kalman – Posición de la partícula")
plt.xlabel("Tiempo [s]")
plt.ylabel("Posición [u.a.]")
plt.legend()
plt.grid(True)
plt.show()

# --- Figura 2: Kalman vs Filtro Complementario (mejor alpha) ---
plt.figure(figsize=(10,5))
plt.plot(t, x_real, label="Trayectoria real", linewidth=2)
plt.plot(t, x_kalman, label="Kalman", linewidth=2, linestyle="--")
plt.plot(t, x_comp_mejor, label=f"Filtro Complementario (alpha={alpha_mejor})", linewidth=2)
plt.title("Filtro Complementario aplicado a la salida del Kalman")
plt.xlabel("Tiempo [s]")
plt.ylabel("Posición [u.a.]")
plt.legend()
plt.grid(True)
plt.show()
