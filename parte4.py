import numpy as np 
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import math


A = np.array([2, 1, 0, 2, 0, 1, 1, 1]).reshape(1, -1)
B = np.array([2, 1, 1, 1, 1, 0, 1, 1]).reshape(1, -1)
P = np.array([1, 2, 3, 0, 4, 6, 7, 9]).reshape(1, -1)
Q = np.array([2, 4, 3, 1, 8, 2, 4, 1]).reshape(1, -1)
S = np.array([2, 1, 4, 7, 1, 4, 5, 6]).reshape(1, -1)
T = np.array([3, 3, 3, 6, 1, 1, 7, 8]).reshape(1, -1)

cos_sim_AB = cosine_similarity(A, B)[0][0]
cos_dis_AB = cosine_distances(A, B)[0][0]

cos_sim_PQ = cosine_similarity(P, Q)[0][0]
cos_dis_PQ = cosine_distances(P, Q)[0][0]

cos_sim_ST = cosine_similarity(S, T)[0][0]
cos_sim_ST = cosine_distances(S, T)[0][0]

def con_angle(cos_sim):
    return math.degrees(math.acos(cos_sim))

angle_AB = con_angle(cos_sim_AB)
angle_PQ = con_angle(cos_sim_PQ)
angle_ST = con_angle(cos_sim_ST)

print(f"Similitud del coseno AB: {cos_sim_AB:.2f}, PQ: {cos_sim_PQ:.2f}, ST: {cos_sim_ST:.2f}")
print(f"Distancia del coseno AB: {cos_dis_AB:.2f}, PQ: {cos_dis_PQ:.2f}, ST: {cos_sim_ST:.2f}")
print(f"Angulo de AB: {angle_AB:.2f}°, PQ: {angle_PQ:.2f}°, ST: {angle_ST:.2f}°")