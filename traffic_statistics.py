import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import numpy as np
import seaborn as sns
import cv2

# Устанавливаем шрифт для matplotlib, поддерживающий кириллицу
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

sns.set_style("whitegrid")  # стиль для красоты

class_names = {
    2: "Автомобили",
    3: "Автобусы",
    5: "Грузовики"
}


def save_statistics(statistics, heatmap_points, frame_shape):
    from video_processing import speed_data
    # Сохранение статистики в файл
    with open("statistics.txt", "w", encoding="utf-8") as file:
        for class_id, count in statistics.items():
            file.write(f"{class_names.get(class_id, 'Неизвестный класс')}: {count} транспортных средств\n")

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2)

    # 1. Визуализация количества транспортных средств
    ax1 = fig.add_subplot(gs[0, 0])
    class_labels = [class_names.get(cid, "Неизвестно") for cid in statistics.keys()]
    ax1.bar(class_labels, statistics.values(), color=sns.color_palette("pastel"))
    ax1.set_title("Количество транспортных средств по типам")
    ax1.set_ylabel("Количество")

    # 2. График скорости автомобилей, автобусов и грузовиков
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(speed_data["frames"], speed_data["cars"], label="Автомобили", color="blue")
    ax2.plot(speed_data["frames"], speed_data["buses"], label="Автобусы", color="orange")
    ax2.plot(speed_data["frames"], speed_data["trucks"], label="Грузовики", color="green")
    ax2.set_title("Изменение скорости по кадрам")
    ax2.set_xlabel("Кадры")
    ax2.set_ylabel("Скорость (пиксели/сек)")
    ax2.legend()

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

    # 3. Тепловая карта плотности по кадру

    # Обратите внимание: y сначала, потом x, чтобы совпало с изображением
    heatmap, yedges, xedges = np.histogram2d(
        [p[1] for p in heatmap_points],  # y
        [p[0] for p in heatmap_points],  # x
        bins=(50, 50),
        range=[[0, frame_shape[0]], [0, frame_shape[1]]]
    )

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        heatmap,
        cmap="hot",
        cbar=True,
        yticklabels=False,
        xticklabels=False
    )
    ax.set_title("Тепловая карта плотности транспорта")
    plt.xlabel("Ширина кадра")
    plt.ylabel("Высота кадра")

    plt.tight_layout()
    plt.show()