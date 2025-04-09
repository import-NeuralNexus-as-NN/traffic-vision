# Функция для расчета плотности потока.
def calculate_flow_density(tracks, width, height, scale_factor=1e6):
    num_objects = len(tracks)  # Количество отслеживаемых объектов
    area = width * height  # Площадь кадра

    # Проверка на корректность
    if num_objects == 0:
        return 0  # Если нет объектов, плотность будет 0

    if area == 0:
        return 0  # Если площадь кадра нулевая, возвращаем 0

    density = num_objects / area  # Плотность = количество объектов / площадь кадра

    # Нормализуем плотность, умножив на коэффициент для улучшения отображения
    density *= scale_factor

    return density