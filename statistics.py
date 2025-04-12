import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import numpy as np

# Устанавливаем шрифт для matplotlib, поддерживающий кириллицу
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

class_names = {
    2: "Автомобили",
    3: "Автобусы",
    5: "Грузовики"
}


def save_statistics(statistics):
    from video_processing import speed_data
    # Сохранение статистики в файл
    with open("statistics.txt", "w", encoding="utf-8") as file:
        for class_id, count in statistics.items():
            file.write(f"{class_names.get(class_id, 'Неизвестный класс')}: {count} транспортных средств\n")

    # 1. Визуализация количества транспортных средств
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)  # Первый график (гистограмма)
    plt.bar(statistics.keys(), statistics.values(), color='skyblue')
    plt.xlabel("Тип транспортного средства")
    plt.ylabel("Количество")
    plt.title("Количество транспортных средств по типам")

    # Заменяем номера классов на названия
    class_labels = [class_names.get(class_id, 'Неизвестный класс') for class_id in statistics.keys()]
    plt.xticks(ticks=list(statistics.keys()), labels=class_labels, rotation=45, ha='right')

    # 2. График скорости автомобилей, автобусов и грузовиков
    plt.subplot(1, 2, 2)  # Второй график (линейный)
    plt.plot(speed_data["frames"], speed_data["cars"], label="Автомобили", color="blue")
    plt.plot(speed_data["frames"], speed_data["buses"], label="Автобусы", color="yellow")
    plt.plot(speed_data["frames"], speed_data["trucks"], label="Грузовики", color="red")

    plt.xlabel("Кадры")
    plt.ylabel("Скорость (пиксели/сек)")
    plt.title("Изменение скорости во времени")

    # # Зависимость скорости от Координата Y (глубина кадра)
    # # Расчет линейной регрессии для тренда
    # slope, intercept, r_value, p_value, std_err = stats.linregress(speed_data["y_coords"], speed_data["cars"])
    #
    # plt.figure(figsize=(6, 4))
    # plt.scatter(speed_data["y_coords"], speed_data["cars"], alpha=0.5, s=10, color='blue', label="Автомобили")
    # plt.scatter(speed_data["y_coords"], speed_data["buses"], alpha=0.5, s=10, color='yellow', label="Автобусы")
    # plt.scatter(speed_data["y_coords"], speed_data["trucks"], alpha=0.5, s=10, color='red', label="Грузовики")
    # # Линия тренда
    # plt.plot(speed_data["y_coords"], np.array(speed_data["y_coords"]) * slope + intercept, color='red', label=f"Тренд (r={r_value:.2f})",
    #          linewidth=2)
    #
    # plt.xlabel("Координата Y в кадре (глубина кадра)")
    # plt.ylabel("Скорость (пиксели/сек)")
    # plt.title("Зависимость скорости от положения в кадре (перспектива)")

    plt.legend()
    plt.grid()

    plt.tight_layout()  # Чтобы графики не налезали друг на друга
    plt.show()