import cv2
from ultralytics import YOLO
import logging
from tkinter import filedialog
import threading
import gui

# Отключаем вывод логов
logging.getLogger("ultralytics").setLevel(logging.ERROR)


# Загрузка предобученной модели YOLOv8
model = YOLO("yolov8s.pt")  # Выбираем YOLOv8s (быстрая и точная)

# Список классов, которые оставляем (по COCO ID)
allowed_classes = {2, 3, 5, 7}  # car, bus, truck, motorcycle


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
    status_label.configure(text="Готово!", text_color="green")


def select_video(status_label):
    file_path = filedialog.askopenfilename(filetypes=[("Видеофайлы", "*.mp4;*.avi;*.mov")])
    if file_path:
        status_label.configure(text=f"Выбрано: {file_path.split('/')[-1]}", text_color="white")
        return file_path
    return None


def start_processing(video_path, status_label):
    # Запуск обработки видео в отдельном потоке, чтобы не зависал GUI
    if video_path:
        thread = threading.Thread(target=process_video, args=(video_path, status_label))
        thread.start()
    else:
        status_label.configure(text="Сначала выберите видео!", text_color="red")


if __name__ == "__main__":
    gui.app.mainloop()  # Запуск интерфейса