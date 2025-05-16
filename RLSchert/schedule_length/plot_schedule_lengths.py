import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
data = np.loadtxt("episode_lengths(sl_50_nullver_episodes).txt")
last_lengths = data[-80:]
episode_numbers = range(41, 121)
# График
plt.figure(figsize=(12, 6))
plt.plot(episode_numbers, last_lengths, color="blue", linewidth=1, label="Длина эпизода")
plt.yticks(np.arange(0, 1001, 100))
plt.xlabel("Итерация")
plt.ylabel("Временные шаги")
plt.title("Длина расписания по эпизодам")
plt.savefig("schedule_length.png", dpi=300)
print("График сохранён в schedule_length.png")