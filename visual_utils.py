import cv2


# Отрисовка текста с полупрозрачным фоном на изображении
def draw_text_with_background(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=1, text_color=(255, 255, 255),
                              bg_color=(0, 0, 0), thickness=2, alpha=0.5, padding=5):
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