from matplotlib import pyplot as plt

from .cams import *


def get_bbox_from_heatmap(heatmap, threshold, merge=True, label_name='unknown', probability=-1, color=(255, 0, 0)):
    gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    x1, y1, x2, y2 = 0, 0, 0, 0

    for item in range(len(contours)):
        cnt = contours[item]
        # x, y is the top left corner, and w, h are the width and height respectively
        x, y, w, h = cv2.boundingRect(cnt)
        _x2, _y2 = x + w, y + h

        if len(cnt) <= 40: continue
        if merge:
            if item > 0:
                x1 = x if x < x1 else x1
                y1 = y if y < y1 else y1
                x2 = _x2 if _x2 > x2 else x2
                y2 = _y2 if _y2 > y2 else y2
            else:
                x1, y1, x2, y2 = x, y, _x2, _y2
        else:
            x1, y1, x2, y2 = x, y, _x2, _y2
            bboxes.append(CanvasObject(x1, y1, x2, y2, label_name, probability, color))
    if merge:
        bboxes.append(CanvasObject(x1, y1, x2, y2, label_name, probability, color))
    return bboxes


def visualize(image, description, title, save_path=None):
    plt.rcParams['font.sans-serif'] = ['Courier New']
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(description)
    plt.title(title)
    if save_path:
        # save class activation map fig
        plt.savefig(save_path, bbox_inches='tight')
    return plt


class CanvasObject:

    def __init__(self, x_min, y_min, x_max, y_max, label='unknown', probability=-1, color=(255, 0, 0)):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.label = label
        self.color = color
        self.probability = probability

    @property
    def text(self):
        return '%s: %.3f' % (self.label, self.probability)
