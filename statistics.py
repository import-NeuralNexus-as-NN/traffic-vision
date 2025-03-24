import matplotlib.pyplot as plt
import matplotlib

# Устанавливаем шрифт для matplotlib, поддерживающий кириллицу
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

class_names = {
    2: "Автомобиль",
    3: "Автобус",
    5: "Грузовик",
    7: "Мотоцикл"
}


def save_statistics(statistics):
    # Сохранение статистики в файл
    with open("statistics.txt", "w", encoding="utf-8") as file:
        for class_id, count in statistics.items():
            file.write(f"{class_names.get(class_id, 'Неизвестный класс')}: {count} транспортных средств\n")

    # Визуализация статистики
    plt.bar(statistics.keys(), statistics.values())
    plt.xlabel("Тип транспортного средства")
    plt.ylabel("Количество")
    plt.title("Количество транспортных средств по типам")

    # Заменяем номера классов на названия
    class_labels = [class_names.get(class_id, 'Неизвестный класс') for class_id in statistics.keys()]
    plt.xticks(ticks=list(statistics.keys()), labels=class_labels, rotation=45, ha='right')

    plt.show()