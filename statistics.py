import matplotlib.pyplot as plt
import matplotlib

# Устанавливаем шрифт для matplotlib, поддерживающий кириллицу
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

class_names = {
    2: "Автомобиль",
    3: "Автобус",
    5: "Грузовик"
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

    # 2. График скорости автомобилей и грузовиков
    plt.subplot(1, 2, 2)  # Второй график (линейный)
    plt.plot(speed_data["frames"], speed_data["cars"], label="Автомобили", color="blue")
    plt.plot(speed_data["frames"], speed_data["trucks"], label="Грузовики", color="red")

    plt.xlabel("Кадры")
    plt.ylabel("Скорость (пиксели/сек)")
    plt.title("Изменение скорости во времени")
    plt.legend()
    plt.grid()

    plt.tight_layout()  # Чтобы графики не налезали друг на друга
    plt.show()