import cv2


def draw_text_with_background(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=1, text_color=(255, 255, 255),
                              bg_color=(0, 0, 0), thickness=2, alpha=0.5, padding=5):
    """
    Отрисовка текста с полупрозрачным фоном на изображении.

    :param image: Кадр (numpy array), на котором будет отрисован текст
    :param text: Строка текста
    :param position: Позиция (x, y) для верхнего левого угла текста
    :param font: Шрифт текста (по умолчанию cv2.FONT_HERSHEY_SIMPLEX)
    :param font_scale: Масштаб шрифта
    :param text_color: Цвет текста (BGR)
    :param bg_color: Цвет фона под текстом (BGR)
    :param thickness: Толщина текста
    :param alpha: Прозрачность фона (0.0 - полностью прозрачно, 1.0 - полностью непрозрачно)
    :param padding: Отступ вокруг текста
    """
    overlay = image.copy()

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    rect_x1 = x - padding
    rect_y1 = y - text_height - padding
    rect_x2 = x + text_width + padding
    rect_y2 = y + padding

    # Отрисовка прямоугольника с фоном
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)

    # Смешиваем с оригинальным изображением
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Наносим текст
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return image