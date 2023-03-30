import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
z = np.sin(x)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.savefig('plot.png')