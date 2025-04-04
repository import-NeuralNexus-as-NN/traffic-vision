import numpy as np

# Словарь для хранения предыдущих координат объектов
prev_objects = {}  # {id: (x_center, y_center)}


# Рассчитывает скорость объекта по его смещению между кадрами.
def calculate_speed(object_id, x_center, y_center, fps):

    if object_id in prev_objects:
        x_prev, y_prev = prev_objects[object_id]
        dx = x_center - x_prev
        dy = y_center - y_prev
        speed = np.sqrt(dx ** 2 + dy ** 2) * fps  # Переводим в пиксели/сек
    else:
        speed = 0  # Если нет предыдущих данных, считаем скорость 0

    # Обновляем координаты объекта
    prev_objects[object_id] = (x_center, y_center)

    return speed