import matplotlib.pyplot as plt


def save_statistics(statistics):
    # Сохранение статистики в файл
    with open("statistics.txt", "w") as file:
        for class_id, count in statistics.items():
            file.write(f"Class {class_id}: {count} vehicles\n")

    # Визуализация статистики
    plt.bar(statistics.keys(), statistics.values())
    plt.xlabel("Vehicle Class")
    plt.ylabel("Count")
    plt.title("Vehicle Count by Class")
    plt.show()