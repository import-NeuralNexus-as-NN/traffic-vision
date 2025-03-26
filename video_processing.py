import cv2
from ultralytics import YOLO
import logging
import os
from statistics import save_statistics
import numpy as np
from speed_tracker import calculate_speed


# Отключаем вывод логов
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Загрузка предобученной модели YOLOv8
model = YOLO("yolov8s.pt")  # Выбираем YOLOv8s (быстрая и точная)

# Список классов, которые оставляем (по COCO ID)
allowed_classes = {2, 3, 5, 7}  # car, bus, truck, motorcycle
statistics = {class_id: 0 for class_id in allowed_classes}  # Словарь для хранения статистики

speed_data = {"cars": [], "trucks": [], "frames": []}


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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Выход из цикла, если видео закончилось

        # Получаем результаты детекции
        results = model(frame)

        total_speed_cars = []
        total_speed_trucks = []

        # Обрабатываем каждый найденный объект
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)  # Класс объекта
                conf = float(box.conf)  # Уверенность модели
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты bbox
                x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2

                speed = calculate_speed(class_id, x_center, y_center, fps)

                # Сохраняем данные для графика
                if class_id == 2:  # Например, 2 - легковые авто
                    total_speed_cars.append(speed)
                elif class_id == 5:  # Например, 5 - грузовики
                    total_speed_trucks.append(speed)

                # Вывод скорости на видео
                label = f"{model.names[class_id]} - {speed:.1f} px/sec"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Фильтрация только наземного транспорта
                if class_id in allowed_classes:
                    label = f"{model.names[class_id]} {conf:.2f}"
                    statistics[class_id] += 1  # Увеличиваем счетчик для соответствующего класса
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Текст

        # Средняя скорость для графика
        avg_speed_cars = np.mean(total_speed_cars) if total_speed_cars else 0
        avg_speed_trucks = np.mean(total_speed_trucks) if total_speed_trucks else 0
        speed_data["cars"].append(avg_speed_cars)
        speed_data["trucks"].append(avg_speed_trucks)
        speed_data["frames"].append(frame_count)
        frame_count += 1

        # Отображаем видео с детекцией
        cv2.imshow('YOLOv8 Detection', frame)

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

