import cv2
import os


def draw_bbox(image, coords, colors, texts=None):
    image = image.copy()
    label_coords = []
    for i, (x_min, y_min, x_max, y_max) in enumerate(coords):
        color = colors[i] if i < len(colors) else colors[0]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        if texts:
            if y_min < 14:
                y_min = 14
            # if overlap
            if i != 0:
                for _, label_y in label_coords:
                    if y_min >= label_y - 14 and y_min <= label_y:
                        y_min = label_y + 14
                    elif y_min >= label_y and y_min <= label_y + 14:
                        y_min = label_y + 14
                    else:
                        break

            label_coords.append((x_min, y_min))
    draw_label(image, label_coords, colors, texts)
    return image


def draw_label(image, coords, colors, texts=None):
    if not texts:
        return image
    if len(coords) != len(texts):
        return image
    for i, (x_min, y_min) in enumerate(coords):
        text = texts[i]
        color = colors[i] if i < len(colors) else colors[0]
        cv2.rectangle(image, (x_min, y_min - 14), (x_min + len(text) * 7, y_min), color, -1)
        cv2.putText(image, text, (x_min + 2, y_min - 5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255),
                    thickness=1, )
    return image


def save_bbox_to_xml(image, coords, filename, labels, path):
    xml_file = open(os.path.join(path, filename + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder></folder>\n')
    xml_file.write('    <filename>' + filename.split('.')[0] + '.jpg</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(image.shape[1]) + '</width>\n')
    xml_file.write('        <height>' + str(image.shape[0]) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    # write the region of image on xml file
    for i, label in enumerate(labels):
        xmin, ymin, xmax, ymax = coords[i]
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + label + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(xmin) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(ymin) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(xmax) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(ymax) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
    xml_file.write('</annotation>')
    xml_file.close()