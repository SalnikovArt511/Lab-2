import matplotlib.pyplot as plt
import numpy as np

# Данные
расстояние = np.array([0, 10, 20, 30, 40, 50, 60, 70])
расход = np.array([2.03, 1.81, 2.44, 2.82, 3.04, 3.48, 3.76, 3.95])

# Аппроксимация методом наименьших квадратов
coefficients = np.polyfit(расстояние, расход, 1)  # Степень полинома 1 для прямой
approximation = np.polyval(coefficients, расстояние)  # Вычисление значений по аппроксимации

# Создание графика
fig, ax = plt.subplots(figsize=(12, 6))

# Исходные данные
ax.plot(расстояние, расход, marker='^', linestyle='-', markersize=6, markerfacecolor='red', color='blue', linewidth=1.4, label='Экспериментальные точки')

# Линия аппроксимации
ax.plot(расстояние, approximation, color='green', linestyle='--', linewidth=1.4, label=f'Аппроксимация: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}')

# Настройка графика
ax.set_xlabel('Расстояние от сопла, мм')
ax.set_ylabel('Расход, г/с')
ax.set_title('Зависимость расхода от расстояния до сопла')

ax.minorticks_on()
ax.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.7)
ax.legend()

# Создание таблицы
data = np.column_stack((расстояние, расход))
table = ax.table(cellText=data, colLabels=['Расстояние (мм)', 'Расход (г/с)'], loc='upper left', cellLoc='center', bbox=[0.008, 0.6, 0.3, 0.3])

# Установка прозрачности границ ячеек
for cell in table._cells.values():
    cell.set_edgecolor('gray')
    cell.set_alpha(0.3)

plt.show()
