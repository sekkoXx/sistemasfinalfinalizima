import numpy as np
import matplotlib.pyplot as plt

# Tiempo
dt = 0.1
t = np.arange(0, 100, dt)
N = len(t)

# Trayectoria 
x_real = 50 - 10 * t

# Mediciones ruidosas
np.random.seed(42)
ruido_std = 20
mediciones = x_real + np.random.normal(0, ruido_std, size=N)

# Parámetros del Kalman
Q = 1e-2
R = ruido_std**2

x_kalman = np.zeros(N)
P = np.zeros(N)

x_kalman[0] = mediciones[0]
P[0] = 1.0

# Filtro de Kalman
for k in range(1, N):
    x_pred = x_kalman[k-1]
    P_pred = P[k-1] + Q

    K = P_pred / (P_pred + R)
    x_kalman[k] = x_pred + K * (mediciones[k] - x_pred)
    P[k] = (1 - K) * P_pred

# Filtro complementario
def filtro_complementario(med, kalman, alpha):
    return alpha * kalman + (1 - alpha) * med

alphas = [0.2, 0.5, 0.8]
comp_por_alpha = {}
mse_por_alpha = {}

for alpha in alphas:
    x_comp = filtro_complementario(mediciones, x_kalman, alpha)
    comp_por_alpha[alpha] = x_comp
    mse_por_alpha[alpha] = np.mean((x_real - x_comp)**2)

alpha_mejor = min(mse_por_alpha, key=mse_por_alpha.get)
x_comp_mejor = comp_por_alpha[alpha_mejor]

print("MSE por alpha:", mse_por_alpha)
print("Mejor alpha:", alpha_mejor)

# Gráfico 1 (Kalman)
plt.figure(figsize=(10,5))
plt.title("Filtro de Kalman")
plt.plot(t, x_real, label="Real Track")
plt.scatter(t, mediciones, s=8, alpha=0.5, label="Mediciones")
plt.plot(t, x_kalman, linewidth=2, label="Predicción Kalman Filter")
plt.legend()
plt.grid(True)
plt.show()

# Gráfico 2 (Complementario)
plt.figure(figsize=(10,5))
plt.title("Filtro Complementario")
plt.plot(t, x_real, label="Real Track")
plt.plot(t, x_kalman, label="Kalman")
plt.plot(t, x_comp_mejor, label=f"Complementario (alpha={alpha_mejor})")
plt.legend()
plt.grid(True)
plt.show()
