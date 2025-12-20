import matplotlib.pyplot as plt
import numpy as np

data = [
    [0.00, 1.09, 20.90, 40.25],
    [0.00, 0.00, 5.13, 20.29],
    [0.00, 0.10, 0.90, 6.40],
    [0.00, 0.00, 1.29, 6.01],
    [0.00, 0.00, 0.00, 0.80]
]
D_n = ['0.1', '0.25', '0.5', '0.75', '1']
fill = ['0.1', '0.15', '0.2', '0.25']

plt.figure(figsize=(10, 6))
for i, d in enumerate(D_n):
    plt.plot(fill, data[i], label=f'D_n = {d}', marker='o')

plt.xlabel('Заполненность контейнера')
plt.ylabel('F1 Score')
plt.title('Тестирование F1 Score по столбцам')
plt.legend()
plt.grid(True)
plt.show()
