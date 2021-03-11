from matplotlib import pyplot as plt

from .cams import *

def get_bbox_from_heatmap(heatmap, threshold):
    gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coords = []
    x1, y1, x2, y2 = 0, 0, 0, 0
    for item in range(len(contours)):
        cnt = contours[item]
        if len(cnt) > 20:
            # x, y is the top left corner, and w, h are the width and height respectively
            x, y, w, h = cv2.boundingRect(cnt)
            _x2, _y2 = x + w, y + h
            if item > 0:
                x1 = x if x < x1 else x1
                y1 = y if y < y1 else y1
                x2 = _x2 if _x2 > x2 else x2
                y2 = _y2 if _y2 > y2 else y2
            else:
                x1, y1, x2, y2 = x, y, _x2, _y2
            # coords.append((x1, y1, x2, y2))
        else:
            pass
            # print("contour error (too small)")
    coords.append((x1, y1, x2, y2))
    return tuple(coords)


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
