from ByteTrack.yolox.tracker.byte_tracker import BYTETracker


# Класс с аргументами для ByteTrack
class ByteTrackArgument:
    track_thresh = 0.5  # Порог для детекций
    match_thresh = 0.8  # Порог для соответствия
    mot20 = False  # Флаг для использования MOT20 (опционально)
    aspect_ratio_thresh = 1.6  # Порог соотношения сторон для фильтрации
    min_box_area = 10  # Минимальная площадь для фильтрации детекций
    track_buffer = 30  # Размер буфера для трекеров (можно менять)


# Инициализируем ByteTrack с аргументами
tracker = BYTETracker(ByteTrackArgument)