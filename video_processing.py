import cv2
from ultralytics import YOLO
import logging
import os
from statistics import save_statistics

# Отключаем вывод логов
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Загрузка предобученной модели YOLOv8
model = YOLO("yolov8s.pt")  # Выбираем YOLOv8s (быстрая и точная)

# Список классов, которые оставляем (по COCO ID)
allowed_classes = {2, 3, 5, 7}  # car, bus, truck, motorcycle
statistics = {class_id: 0 for class_id in allowed_classes}  # Словарь для хранения статистики


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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Выход из цикла, если видео закончилось

        # Получаем результаты детекции
        results = model(frame)

        # Обрабатываем каждый найденный объект
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)  # Класс объекта
                conf = float(box.conf)  # Уверенность модели
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты bbox

                # Фильтрация только наземного транспорта
                if class_id in allowed_classes:
                    label = f"{model.names[class_id]} {conf:.2f}"
                    statistics[class_id] += 1  # Увеличиваем счетчик для соответствующего класса
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Текст

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

