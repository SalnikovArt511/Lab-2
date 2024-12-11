import numpy as np
import matplotlib.pyplot as plt

def av(X):
    return np.average(X)

def calib(X, Y):
    k = (av(Y * X) - av(Y) * av(X)) / (av(X * X) - av(X)**2)
    b = av(Y) - av(X) * k
    return k, b

plt.grid(which='major')
plt.grid(which='minor', linestyle=':')
plt.minorticks_on()

file = open('Mover_calib.txt')
X, Y = [], []
for l in file:
    X.append(float(l.split()[0]))
    Y.append(float(l.split()[1]))
file.close()

X, Y = np.array(X), np.array(Y)

print("X:", X)
print("Y:", Y)

k, b = calib(X, Y)

k_exp = k * 1e3
k_formatted = f"{k_exp:.2f} \\cdot 10^{{-3}}"

plt.scatter(X, Y, 20, marker='x', c='r', linewidths=1)
plt.xlabel('Шаг двигателя')
plt.ylabel('Пройденное трубкой Пито расстояние, см')
plt.title('График калибровки шагового двигателя')
plt.plot(X, k * X, 'black', linewidth=1, label='Аппроксимация $y=kx$\n' +
         f'$k={k_formatted}~\\text{{см/шаг}}$')
plt.legend()
plt.savefig('shag.png')
plt.show()
