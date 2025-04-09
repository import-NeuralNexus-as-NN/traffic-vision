import cv2
from ultralytics import YOLO
import logging
import os
from statistics import save_statistics
import numpy as np
from speed_tracker import calculate_speed, SpeedSmoothing
from tracker import tracker

# Отключаем вывод логов
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Загрузка предобученной модели YOLOv8
model = YOLO("yolov8s.pt")  # Выбираем YOLOv8s (быстрая и точная)

# Список классов, которые оставляем (по COCO ID)
allowed_classes = {2, 3, 5}  # car, bus, truck
statistics = {class_id: 0 for class_id in allowed_classes}  # Словарь для хранения статистики

speed_data = {"cars": [], "trucks": [], "frames": []}

# Порог уверенности
confidence_threshold = 0.5  # Установите нужный порог

speed_smoother = SpeedSmoothing(window_size=5)


def process_video(video_path, status_label):
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

    # Определяем путь сохранения
    output_path = os.path.splitext(video_path)[0] + "_processed.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Переменные для графика
    frame_count = 0

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

            for track in tracks:
                # Получаем координаты трека
                x1, y1, x2, y2 = map(int, track.tlbr)
                track_id = track.track_id

                # Визуализация от ByteTrack'а
                label = f"ID {track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)  # Текст

                # Найдем детекцию, которая соответствует треку
                best_match = None
                for detection in detections:
                    detection_x1, detection_y1, detection_x2, detection_y2, _, detection_class_id = detection
                    # Простое правило для нахождения соответствующей детекции:
                    if detection_x1 <= x1 <= detection_x2 and detection_y1 <= y1 <= detection_y2:
                        best_match = detection
                        break

                if best_match is not None:
                    # Извлекаем class_id из детекции
                    class_id = best_match[5]

                    # Фильтрация по классу (например, только легковые автомобили и грузовики)
                    if class_id not in allowed_classes:
                        continue  # Пропускаем объект, если он не в списке разрешенных классов

                    x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
                    speed = calculate_speed(class_id, x_center, y_center, fps)

                    # Сглаживаем скорость
                    smoothed_speed = speed_smoother.smooth(class_id, speed)

                    # Подписываем ID трека и скорость
                    label = f"ID {track_id} | {speed:.1f} px/sec"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                2)  # Текст
                    # Добавим информацию о треке в словарь
                    track_info[track_id] = {'class_id': class_id, 'speed': smoothed_speed}

                # Обновляем множества уникальных объектов для каждого класса
                if class_id == 2:  # car
                    unique_cars.add(track_id)
                elif class_id == 3:  # bus
                    unique_buses.add(track_id)
                elif class_id == 5:  # truck
                    unique_trucks.add(track_id)

                # Добавим информацию о треке в словарь
                track_info[track_id] = {'class_id': class_id, 'speed': speed}

                # Обновление статистики после подсчета уникальных объектов
                statistics[2] = len(unique_cars)  # количество уникальных легковых автомобилей
                statistics[3] = len(unique_buses)  # количество уникальных автобусов
                statistics[5] = len(unique_trucks)  # количество уникальных грузовиков

        # Средняя скорость для графика
        avg_speed_cars = np.mean(total_speed_cars) if total_speed_cars else 0
        avg_speed_trucks = np.mean(total_speed_trucks) if total_speed_trucks else 0
        speed_data["cars"].append(avg_speed_cars)
        speed_data["trucks"].append(avg_speed_trucks)
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
    save_statistics(statistics)

    status_label.configure(text=f"Готово!Видео сохранено: {output_path}", text_color="green")