import customtkinter as ctk
from main import select_video, start_processing

# Настройки интерфейса
ctk.set_appearance_mode("dark")  # Темная тема
ctk.set_default_color_theme("blue")  # Синяя цветовая схема

# Создание GUI
app = ctk.CTk()
app.title("YOLOv8 Video Detector")
app.geometry("500x300")

# Заголовок
title_label = ctk.CTkLabel(app, text="Оценка транспортного потока на видео (YOLOv8)", font=("Arial", 18))
title_label.pack(pady=20)

# Кнопка выбора видео
video_path = None  # Путь к видео


def update_video_path():
    global video_path
    video_path = select_video(status_label)


select_button = ctk.CTkButton(app, text="Выбрать видео", command=update_video_path)
select_button.pack(pady=10)

# Кнопка запуска обработки
start_button = ctk.CTkButton(app, text="Запустить обработку", command=lambda: start_processing(video_path, status_label))
start_button.pack(pady=10)

# Статус
status_label = ctk.CTkLabel(app, text="Для начала выберите видео", text_color="gray")
status_label.pack(pady=20)

