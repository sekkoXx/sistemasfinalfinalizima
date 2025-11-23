from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
iris = load_iris()

x = iris.data
y = iris.target

kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(x)

cluster = kmeans.labels_

plt.scatter(x[:, 0], x[:, 1], c=cluster, cmap='viridis', marker='o')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal With')
plt.title('Cluster Kmeans en Iris')
plt.show()
