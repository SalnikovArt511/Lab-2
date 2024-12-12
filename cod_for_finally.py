import matplotlib.pyplot as plt
import numpy as np

расстояние = np.array([0, 10, 20, 30, 40, 50, 60, 70])
расход = np.array([2.03, 1.81, 2.44, 2.82, 3.04, 3.48, 3.76, 3.95])

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(расстояние, расход, marker='^', linestyle='-', markersize=6, markerfacecolor='red', color='blue', linewidth=1.4, label='Экспериментальные точки')
ax.set_xlabel('Расстояние от сопла, мм')
ax.set_ylabel('Расход, г/с')
ax.set_title('Зависимость расхода от расстояния до сопла')

ax.minorticks_on()
ax.grid(which='both', linestyle=':', linewidth=0.5, alpha=0.7)
ax.legend()

# Создаем таблицу с центрированием данных
data = np.column_stack((расстояние, расход))
table = ax.table(cellText=data, colLabels=['Расстояние (мм)', 'Расход (г/с)'], loc='upper left', cellLoc='center', bbox=[0.008, 0.6, 0.3, 0.3])


# Устанавливаем прозрачность границ ячеек
for cell in table._cells.values():
    cell.set_edgecolor('gray')
    cell.set_alpha(0.3)

plt.show()
