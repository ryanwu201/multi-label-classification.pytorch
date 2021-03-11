import cv2


def draw_bbox(image, coords, colors, texts=None):
    image = image.copy()
    for i, (x_min, y_min, x_max, y_max) in enumerate(coords):
        color = colors[i] if i < len(colors) else colors[0]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        if texts:
            text = texts[i] if i < len(texts) else texts[0]
            cv2.rectangle(image, (x_min, y_min), (x_min + len(text) * 8, y_min + 14), color, -1)
            cv2.putText(image, text, (x_min + 2, y_min + 11), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255),
                        thickness=1, )
    return image