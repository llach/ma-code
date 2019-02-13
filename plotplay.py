import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
random_numbers = np.random.rand(100000)
x = -np.log(-np.log(np.random.rand(100000)))
# x = np.log(random_numbers / (1 - random_numbers))
n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)
plt.grid(True)
plt.show()