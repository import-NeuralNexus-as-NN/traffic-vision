import cv2
from ultralytics import YOLO
import logging
import os
from traffic_statistics import save_statistics
import numpy as np
from speed_tracker import calculate_speed, SpeedSmoothing
from tracker import tracker
from flow_density import calculate_flow_density
from visual_utils import draw_text_with_background

# Отключаем вывод логов
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Загрузка предобученной модели YOLOv8
model = YOLO("yolov8s.pt")  # Выбираем YOLOv8s (быстрая и точная)

# Список классов, которые оставляем (по COCO ID)
allowed_classes = {2, 3, 5}  # car, bus, truck
statistics = {class_id: 0 for class_id in allowed_classes}  # Словарь для хранения статистики

speed_data = {"cars": [], "buses": [], "trucks": [], "frames": [], "all_speeds": []}

track_classes = {}  # track_id: class_id

class_names = {
    2: "car",
    3: "bus",
    5: "truck"
}

# Глобальный список координат центров bbox
heatmap_points = []  # [(x1, y1), (x2, y2), ...]

# Порог уверенности
confidence_threshold = 0.5  # Установите нужный порог

speed_smoother = SpeedSmoothing()


def process_video(video_path, status_label):
    global speed_data, statistics, track_classes, speed_smoother, heatmap_points

    # Обновляем переменные
    speed_data = {"cars": [], "buses": [], "trucks": [], "frames": [], "all_speeds": []}
    statistics = {class_id: 0 for class_id in allowed_classes}
    track_classes = {}
    speed_smoother = SpeedSmoothing()
    heatmap_points.clear()

    if not video_path:
        status_label.configure(text="Сначала выберите видео!", text_color="red")
        return

        # Открытие видеопотока
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        status_label.configure(text="Ошибка: не удалось открыть видео!", text_color="red")
        return

    status_label.configure(text="Обработка видео...", text_color="yellow")

    # Получаем параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_shape = (height, width)

    # Определяем путь сохранения
    output_path = os.path.splitext(video_path)[0] + "_processed.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Переменные для графика
    frame_count = 0

    # Для оценки загруженности по количеству уникальных объектов за последние 5 секунд
    window_size = fps * 5  # Количество кадров в 5-секундном окне
    track_id_window = []

    # Множества для отслеживания уникальных объектов по track_id
    unique_cars = set()
    unique_buses = set()
    unique_trucks = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Выход из цикла, если видео закончилось

        # Получаем результаты детекции
        results = model(frame)

        total_speed_cars = []
        total_speed_buses = []
        total_speed_trucks = []
        detections = []

        # Обрабатываем каждый найденный объект
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)  # Класс объекта
                conf = float(box.conf)  # Уверенность модели
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты bbox

                # Пропускаем детекции с низкой уверенностью
                if conf < confidence_threshold:
                    continue

                # Добавляем информацию о классе в детекцию
                detections.append([x1, y1, x2, y2, conf, class_id])

        # Передаем данные в трекер (проверим, что они в нужном формате)
        if len(detections) > 0:
            detections_array = np.array(detections)

            # Обновляем трекер
            tracks = tracker.update(detections_array, frame.shape[:2], frame.shape[:2])

            # Создадим словарь для отслеживания классов и объектов
            track_info = {}

            # Добавляем текущие track_id в скользящее окно
            current_ids = [track.track_id for track in tracks]
            track_id_window.extend(current_ids)

            # Ограничиваем размер окна
            if len(track_id_window) > window_size:
                track_id_window = track_id_window[-window_size:]

            # Подсчет уникальных объектов в окне
            unique_objects_last_5_sec = len(set(track_id_window))

            # Пороговые значения (можно настроить)
            LOW_THRESHOLD = 10
            HIGH_THRESHOLD = 30

            if unique_objects_last_5_sec < LOW_THRESHOLD:
                congestion_level = "Low"
                color = (0, 255, 0)
            elif unique_objects_last_5_sec < HIGH_THRESHOLD:
                congestion_level = "Average"
                color = (0, 255, 255)
            else:
                congestion_level = "High"
                color = (0, 0, 255)

            # Визуализация загруженности
            draw_text_with_background(frame, f"Congestion level: {congestion_level}", (10, 60), bg_color=color)
            draw_text_with_background(frame, f"Objects for 5 sec: {unique_objects_last_5_sec}", (10, 100),
                                      bg_color=(50, 50, 50))

            for track in tracks:
                # Получаем координаты трека
                x1, y1, x2, y2 = map(int, track.tlbr)
                track_id = track.track_id

                # Попробуем извлечь class_id, если он уже сохранён
                class_id = track_classes.get(track_id)

                class_name = class_names.get(class_id, "unknown")

                # Визуализация от ByteTrack'а
                label = f"ID {track_id} | {class_name}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)  # Текст

                # Если не знаем class_id, попытаемся найти
                if class_id is None:
                    for detection in detections:
                        detection_x1, detection_y1, detection_x2, detection_y2, _, detection_class_id = detection
                        # Простое правило для нахождения соответствующей детекции:
                        if detection_x1 <= x1 <= detection_x2 and detection_y1 <= y1 <= detection_y2:
                            if detection_class_id in allowed_classes:
                                class_id = detection_class_id
                                track_classes[track_id] = class_id
                            break

                # Фильтрация по классу (например, только легковые автомобили и грузовики)
                if class_id not in allowed_classes:
                    continue  # Пропускаем объект, если он не в списке разрешенных классов

                x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2

                # Сохраняем координаты
                heatmap_points.append((x_center, y_center))

                speed = calculate_speed(track_id, x_center, y_center, fps)

                # Сглаживаем скорость
                smoothed_speed = speed_smoother.smooth(track_id, speed)

                # Подписываем ID трека и скорость
                label = f"ID {track_id} | {class_name} | {smoothed_speed:.1f} px/sec"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)  # Текст

                # Добавим информацию о треке в словарь
                track_info[track_id] = {'class_id': class_id, 'speed': smoothed_speed}

                # Добавляем скорость в статистику для классов
                if class_id == 2:  # car
                    total_speed_cars.append(smoothed_speed)
                elif class_id == 3:  # bus
                    total_speed_buses.append(smoothed_speed)
                elif class_id == 5:  # truck
                    total_speed_trucks.append(smoothed_speed)

                # Обновляем множества уникальных объектов для каждого класса
                if class_id == 2:  # car
                    unique_cars.add(track_id)
                elif class_id == 3:  # bus
                    unique_buses.add(track_id)
                elif class_id == 5:  # truck
                    unique_trucks.add(track_id)

                # Обновление статистики после подсчета уникальных объектов
                statistics[2] = len(unique_cars)  # количество уникальных легковых автомобилей
                statistics[3] = len(unique_buses)  # количество уникальных автобусов
                statistics[5] = len(unique_trucks)  # количество уникальных грузовиков

            # Расчет плотности потока
            density = calculate_flow_density(tracks, width, height, scale_factor=1e6)

            # # Отображаем плотность потока на кадре
            # cv2.putText(frame, f"Flow Density: {density:.5f} objects per million pixels",
            #             (10, 30),  # Позиция текста (верхний левый угол)
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8,  # Шрифт и размер текста
            #             (0, 255, 0),  # Цвет текста (зеленый)
            #             2)  # Толщина линии текста

        # Средняя скорость для графика
        avg_speed_cars = np.mean(total_speed_cars) if total_speed_cars else 0
        avg_speed_buses = np.mean(total_speed_buses) if total_speed_buses else 0
        avg_speed_trucks = np.mean(total_speed_trucks) if total_speed_trucks else 0
        speed_data["cars"].append(avg_speed_cars)
        speed_data["buses"].append(avg_speed_buses)
        speed_data["trucks"].append(avg_speed_trucks)
        # speed_data["y_coords"].append(y_center)  # Сохраняем координаты Y
        speed_data["frames"].append(frame_count)
        frame_count += 1

        # Отображаем видео с детекцией
        cv2.imshow('YOLOv8 + ByteTrack', frame)

        out.write(frame)  # Записываем кадр в выходное видео

        # Нажатие "q" для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Сохранение статистики
    save_statistics(statistics, heatmap_points, frame_shape)

    status_label.configure(text=f"Готово!Видео сохранено: {output_path}", text_color="green")