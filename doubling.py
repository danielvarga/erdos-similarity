import numpy as np
import matplotlib.pyplot as plt


N = 100000
k = 10
x_range = np.linspace(0, 1, N)
x = x_range.copy()
xs =  []
for i in range(k):
    xs.append(x)
    x, _ = np.modf(2 * x)

xs = np.array(xs).T
xs = np.sort(xs, axis=1)
cyc = np.zeros((N, k + 1))
cyc[:, :-1] = xs
cyc[:, -1] = xs[:, 0] + 1
deltas = cyc[:, 1:] - cyc[:, :-1]
epsilons = deltas.max(axis=1)


plt.plot(x_range, epsilons)
plt.ylim(0, 1)
plt.show()

plt.hist(epsilons, bins=100)
plt.show()
