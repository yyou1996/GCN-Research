import numpy as np  
import matplotlib.pyplot as plt  


distance = np.loadtxt('distance.txt')
plt.plot(distance)
plt.show()