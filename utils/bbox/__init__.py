import cv2
import os
import copy


def draw_bbox(image, bboxes):
    image = image.copy()
    for i, bbox in enumerate(bboxes):
        cv2.rectangle(image, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), bbox.color, 2)
    draw_label(image, init_label_coords(bboxes))
    return image


def init_label_coords(bboxes):
    labels = []
    for i, bbox in enumerate(bboxes):
        label = copy.deepcopy(bbox)
        label.x_max, label.y_max = 0, 0
        if label.y_min < 14:
            label.y_min = 14
        # if overlap
        if i != 0:
            for label_save in labels:
                if label.y_min >= label_save.y_min - 14 and label.y_min <= label_save.y_min:
                    label.y_min = label_save.y_min + 14
                elif label.y_min >= label_save.y_min and label.y_min <= label_save.y_min + 14:
                    label.y_min = label_save.y_min + 14
                else:
                    break
        labels.append(label)
    return labels


def draw_label(image, labels):
    for label in labels:
        cv2.rectangle(image, (label.x_min, label.y_min - 14), (label.x_min + len(label.text) * 7, label.y_min),
                      label.color, -1)
        cv2.putText(image, label.text, (label.x_min + 2, label.y_min - 5), cv2.FONT_HERSHEY_COMPLEX, 0.4,
                    (255, 255, 255),
                    thickness=1, )
    return image


def save_bbox_to_xml(image, bboxes, filename, path):
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
    for i, bbox in enumerate(bboxes):
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + bbox.label + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(bbox.x_min) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(bbox.y_min) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(bbox.x_max) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(bbox.y_max) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
    xml_file.write('</annotation>')
    xml_file.close()
