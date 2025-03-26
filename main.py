from tkinter import filedialog
import threading
import gui
from video_processing import process_video
from speed_tracker import calculate_speed
from statistics import save_statistics


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