from logging import fatal
from math import factorial
from sys import prefix

import pandas as pd
from fontTools.t1Lib import write
from scipy.optimize import curve_fit
import matplotlib.pyplot as plot
import numpy as np
import random as rnd
import math

from scipy.stats import expon

def factor(x):
    res = 1
    for i in range(2, x + 1):
        res *= i
    return res

# Аппромиксация функции методом наименьших квадратов
def approx(x, y):
    a = ( np.mean( np.multiply(x, y)) - np.mean(x) * np.mean(y) ) / ( np.mean(np.pow(x, 2)) - np.mean(x)**2 )
    b = np.mean(y) - a * np.mean(x)
    return a, b
def err_approx(x, y, r):
    err_k = 1 / ((x.__len__())**0.5) * ( (np.mean(np.pow(y, 2)) - np.mean(y)**2) / (np.mean(np.pow(x, 2) - np.mean(x)**2)) - r**2 )**0.5
    err_b = err_k * np.sqrt( np.mean(np.pow(x, 2)) - np.mean(x)**2 )
    return err_k, err_b
def err_approx_pro(x, y, err_y):
    length = x.__len__()
    err_max_k = ((y[-1] + err_y[-1]) - (y[0] - err_y[0])) / (x[-1] - x[1])
    err_min_k = ((y[-1] - err_y[-1]) - (y[0] + err_y[0])) / (x[-1] - x[1])
    return (err_max_k - err_min_k) / length**0.5

def parted_approx(x, y, step):
    parts_x = []
    parts_y = []
    min_parts_x = min(x)
    for j in range(0, int(len(x) / step)):
        parts_x.append([i for i in x if i >= min_parts_x and i < min_parts_x + step])
        parts_y.append([y[i] for i in range(0, len(y)) if
                               x[i] >= min_parts_x and x[i] < min_parts_x + step])
        min_parts_x += step
    a_appr = []
    b_appr = []
    for j in range(0, len(parts_x)):
        tmp_a, tmp_b = approx(parts_x[j], parts_y[j])
        a_appr.append(tmp_a)
        b_appr.append(tmp_b)
    return a_appr, b_appr


# Аппроксимация функции методом минимума хи-квадрат для функции типа y = kx + b
def xi_kvadro(x, y, err_x, err_y, k, b):
    length = x.__len__()
    delta_y = [ y[i] - k * x[i] + b for i in range(0, length) ]
    return np.sum([ (delta_y[i] / err_y[i])**2 for i in range(0, length) ])

def Maver(x, err_w): # Нахождение взвешенного среднего
    W = np.sum(err_w)
    return np.sum(np.multiply(err_w, x)) / W
def min_xi_kvadro(x, y, err_x, err_y): # Возвращение оптимальных k и b [[[ОСНОВА]]]
    err_w = [ 1 / i**2 for i in err_y ]
    k = ( Maver(np.multiply(x, y), err_w) - Maver(x, err_w) * Maver(y, err_w) ) / ( Maver(np.pow(x, 2), err_w) - Maver(x, err_w)**2 )
    b = Maver(y, err_w) - k * Maver(x, err_w)
    return k, b

def err_maker_xi_kvadro(x, y, err_x, err_y, k, b): # Нахождение погрешностей,
    num_of_sigm = 1 ### КОЛИЧЕСТВО СИГМ, рекомендуется 1
    diff = 10000
    err_opt_k = k
    err_opt_b = b
    xi = xi_kvadro(x, y, err_x, err_y, k, b)
    for i in range(0, 2000):
        tmp_err_k = k * i / 20000
        for j in range(0, 2000):
            tmp_err_b = b * j / 20000
            xi_delta = xi_kvadro(x, y, err_x, err_y, k + tmp_err_k, b + tmp_err_b) - xi
            if abs(xi_delta - num_of_sigm) < diff:
                diff = abs(xi_delta - num_of_sigm)
                err_opt_k = tmp_err_k
                err_opt_b = tmp_err_b
    return err_opt_k, err_opt_b

# АНТИ-ШУМ
def anti_noise(x, y, step):
    parts_x = []
    last_min_x = min(x)
    while True:
        if (last_min_x > max(x)):
            break
        parts_x.append([ i for i in x if i >= last_min_x and i < last_min_x + step ])
        if len(parts_x[len(parts_x) - 1]) == 0:
            parts_x.pop(len(parts_x) - 1)
            parts_x.append(parts_x[len(parts_x) - 1])
        last_min_x += step
    parts_indexes = [[ x.index(j) for j in i ] for i in parts_x]
    parts_y = [ [ y[j] for j in i ] for i in parts_indexes ]
    new_y = [ np.mean(i) for i in parts_y]
    new_x = [ np.mean(i) for i in parts_x]
    return new_x, new_y

# Определенныый интеграл с делением на partes частей
def integral(x, y, parts):
    step = len(x) / parts
    parts_x = []
    last_min_x = min(x)
    while True:
        if (last_min_x > max(x)):
            break
        parts_x.append([ i for i in x if i >= last_min_x and i <= last_min_x + step ])
        last_min_x += step
    parts_indexes = [[ x.index(j) for j in i ] for i in parts_x]
    parts_y = [ [ y[j] for j in i ] for i in parts_indexes ]
    meaned_y = [ np.mean(i) for i in parts_y ]
    length_parts_x = [ abs(max(i) - min(i)) for i in parts_x ]
    return np.sum([ meaned_y[i] * length_parts_x[i] for i in range(0, len(parts_x)) ])


# Погрешность среднего
def err_of_aver(x):
    len_x = len(x)
    if len_x <= 10:
        needed_sum = 0
        for i in range(0, len_x):
            needed_sum += (x[i] - np.mean(x))**2
        return (needed_sum / (len_x - 1))**0.5
    else:
        needed_sum = 0
        for i in range(0, len_x):
            needed_sum += (x[i] - np.mean(x))**2
        return (needed_sum / len_x)**0.5
    return x[0] / 0 # error maker ;)

# Среднее квадратичное отклонение
def aver_quadro_change(x, lenX):
    return (np.sum((x - np.mean(x))**2) / (lenX - 1))**0.5

# Вычисление распределения Пуассона
def w_n_maker(calced_aver_num_in_sec, n):
    return ( (calced_aver_num_in_sec**n) / factor(n) ) * np.e**(-1 * calced_aver_num_in_sec) * 100

# Среднеквадратичная погрешность определения среднего
def aver_quadro_error_of_aver(aver, num_of_data_in_time_group):
    return (aver / num_of_data_in_time_group)**0.5
# Относительная среднеквадратичная погрешность определения среднего
def relativ_aver_quadro_error_of_aver(num_of_data):
    return 1 / (num_of_data)**0.5

# Средняя интенсивность регистрируемых частиц в секунду
def aver_intens_of_particle_in_sec(aver, time_step):
    return aver / time_step
def maker_err_of_aver_intens(aver, err_of_aver, aver_intens):
    return aver_intens * (err_of_aver / aver)

# Распределение Гаусса
def Gauss(aver, aver_qu, x):
    return ( 1 / (aver_qu * (2 * np.pi)**0.5) ) * (np.exp( ((-1)/2) * ((x - aver)/aver_qu)**2 ))

fig, ax = plot.subplots(figsize=(18, 10), layout='constrained')

k_pressure_kalibr = 0.18
b_pressure_kalibr = -151.60
k_x_kalibr = 0.0000549

air_dencity = 1.2255
diametr_of_machine = 0.00761

step_of_noise_cancelling = 0.001
parts_for_integral = 242

Path = []
Path.append("C:/Users/artem/PycharmProjects/pythonProject4/data40.txt")
Path.append("C:/Users/artem/PycharmProjects/pythonProject4/data50.txt")
Path.append("C:/Users/artem/PycharmProjects/pythonProject4/data60.txt")
Path.append("C:/Users/artem/PycharmProjects/pythonProject4/data70.txt")

data = []
for j in range(0, len(Path)):
    data.append( open(Path[j]).read().split('\n') )

num_to_conv_to_x = []
num_to_conv_to_pressure = []
for j in range(0, len(Path)):
    num_to_conv_to_x.append([ i for i in range(0, int(len(data[j]) / 3)) ])
    num_to_conv_to_pressure.append([ ( int(data[j][i]) + int(data[j][i + 1]) + int(data[j][i + 2]) ) / 3 for i in range( 0, int(len(data[j])), 3 ) ])

x = []
pressure = []
for j in range(0, len(Path)):
    x.append([ ( k_x_kalibr * i ) for i in num_to_conv_to_x[j] ])
    pressure.append([ ( k_pressure_kalibr * i + b_pressure_kalibr ) for i in num_to_conv_to_pressure[j] ])

velocity = []
for j in range(0, len(Path)):
    tmp_velocity = []
    for i in pressure[j]:
        i = i - 13
        if ( (2 * i) / air_dencity) > 0:
            tmp_velocity.append( ((2 * i) / air_dencity)**0.5 )
        else:
            tmp_velocity.append(0)
    velocity.append(tmp_velocity)

### [#####--NOISE-CANCELLING--#####]
for j in range(0, len(Path)):
    x[j], velocity[j] = anti_noise(x[j], velocity[j], step_of_noise_cancelling)

### [#####--CENTRALIZATION--#####]
for j in range(0, len(Path)):
    x_for_max_velocity = x[j][ velocity[j].index(max(velocity[j])) ]
    x[j] = [ i - x_for_max_velocity for i in x[j] ]

### [#####--CALCULATING-FLOW-RATE--#####]
flow_rate = []
for j in range(0, len(Path)):
    tmp_r_V = [ abs(velocity[j][i] * x[j][i]) for i in range(0, len(x[j])) ]
    flow_rate.append(np.pi * air_dencity * integral(x[j], tmp_r_V, parts_for_integral))

k = 0
for j in range(0, len(Path)):
    label = "Q = " + "(" + str(k) + " мм) = " + str(flow_rate[j] * 1000)[:4] + " [г/с]"
    ax.plot(x[j], velocity[j], label=label)
    k += 10

print(flow_rate)

ax.set_title("Градиент скорости струи от расстоятния до сопла")
ax.set_ylabel('Скорость, м/c')
ax.set_xlabel("Расстояние, м")

ax.legend()
ax.legend(prop={'size':16})

plot.grid()
plot.show()
