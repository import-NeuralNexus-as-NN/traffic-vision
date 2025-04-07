import numpy as np

# Словарь для хранения предыдущих координат объектов
prev_objects = {}  # {id: (x_center, y_center)}


# Класс для сглаживания скорости
class SpeedSmoothing:
    def __init__(self, window_size=5):
        self.window_size = window_size  # Размер окна для скользящего среднего
        self.speeds = {}  # Словарь для хранения скоростей по объектам

    def smooth(self, object_id, new_speed):
        """Добавляем новую скорость и возвращаем сглаженную скорость."""
        if object_id not in self.speeds:
            self.speeds[object_id] = []

        # Добавляем новую скорость в список
        self.speeds[object_id].append(new_speed)

        # Оставляем только последние window_size скоростей
        if len(self.speeds[object_id]) > self.window_size:
            self.speeds[object_id].pop(0)

        # Возвращаем среднее значение скоростей
        return np.mean(self.speeds[object_id]) if self.speeds[object_id] else new_speed


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