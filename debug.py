import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y)
plt.savefig("dummy_name.png")

