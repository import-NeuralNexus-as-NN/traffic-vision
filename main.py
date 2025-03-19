import cv2
from ultralytics import YOLO
import logging
import tkinter as tk
from tkinter import filedialog

# Отключаем вывод логов
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Загрузка предобученной модели YOLOv8
model = YOLO("yolov8s.pt")  # Выбираем YOLOv8s (быстрая и точная)

# Список классов, которые оставляем (по COCO ID)
allowed_classes = {2, 3, 5, 7}  # car, bus, truck, motorcycle


def process_video(video_path):
    # Открытие видеопотока
    cap = cv2.VideoCapture("source/1952-152220070_small.mp4")  # Укажите путь к вашему видео

    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео")
        return

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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Текст

        # Отображаем видео с детекцией
        cv2.imshow('YOLOv8 Detection', frame)

        # Нажатие "q" для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def select_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        process_video(file_path)


# Создание окна Tkinter
root = tk.Tk()
root.withdraw()  # Прячем основное окно
select_video()  # Запускаем выбор файла
