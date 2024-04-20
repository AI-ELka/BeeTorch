import matplotlib.pyplot as plt


X1 = [0.1175,0.1028,0.0981,0.0941,0.0905]
X2 = [0.3147,23.0483,2734.1946,325615.0312,38780928.0000]

X = [i for i in range(5)]

plt.plot(X,X1)
plt.plot(X,X2)
plt.show()