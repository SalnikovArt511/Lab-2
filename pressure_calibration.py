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

file1 = open('pressure_calibration(0Pa).txt')
file2 = open('pressure_calibration(63,1Pa).txt')

x1 = [float(line.strip()) for line in file1]
x2 = [float(line.strip()) for line in file2]
file1.close()
file2.close()

X = np.array([av(x1), av(x2)])
Y = np.array([0.0, 63.1])

print("Усреднённые значения АЦП:", X)
print("Давление по барометру:", Y)

k, b = calib(X, Y)

plt.scatter(X, Y, 20, marker='x', c='r', linewidths=1, label='Данные измерений')
plt.title('Калибровка датчика давления')
plt.xlabel('Значение по шагам АЦП')
plt.ylabel('Давление по барометру, Па')
plt.plot(X, k * X + b, 'black', linewidth=1, label='Аппроксимация $y=kx+b$\n' +
         '$k={0:.2f} Па/шаг(АЦП)~~b={1:.2f} Па$'.format(k, b))
plt.legend()
plt.savefig('pressure_calib.png')
plt.show()
