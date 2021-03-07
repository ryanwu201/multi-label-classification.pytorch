# coding=utf-8

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# 请根据这篇博客，定义自己的dataset, 核心看这个FaceLandmarksDataset

import os
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

"""
这里举一个读取pasca voc数据的例子
pascal voc 数据介绍https://arleyzhang.github.io/articles/1dc20586/
其核心是由三个文件夹组成
|---root
|-------Annotation
|-------ImageSets
|----------------Main存放类标签
|-------JPEGImages 存放所有的图片

Main中存储形式
类名_train/val/test
分别是属于train，val和test的文件

第一列是文件名，第二列(-1, 1)表示该图片中不存在或者存在
"""


class MultiLabelDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, label_root_path, img_root_path, imageset_root_path=None, label_file_suffix='.txt',
                 train_type='trainval',
                 label_type='naive',
                 transform=None, requires_filename=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.suffix = '.jpg'
        self.label_file_suffix = label_file_suffix
        self.train_type = train_type
        self.label_type = label_type
        self.label_file_suffix = label_file_suffix
        if (self.label_type != 'native') and self.label_file_suffix == '.txt':
            self.label_file_suffix = '.xml'
        if (self.label_type == 'csv') and self.label_file_suffix == '.txt':
            self.label_file_suffix = '.csv'
        self.one_hot_map = dict()
        imageset_filenames = None
        if self.label_type == 'voc':
            imageset_filenames = self._get_imageset_filenames(imageset_root_path)
        self.data = self._read_file(label_root_path, imageset_filenames)
        self.img_root_path = img_root_path
        self.transform = transform
        self.requires_filename = requires_filename

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, multi_labels_embeding = self.data[idx]
        img = cv2.imread(os.path.join(self.img_root_path, filename + self.suffix))
        # img = io.imread(os.path.join(self.img_root_path, filename + self.suffix))
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        # 每次获得的图片样子形状可以在这里进行观察
        """
        调用matlabplot
        plt.figure()
        plt.imshow(img)
        plt.show()
        """
        if self.requires_filename:
            return img, torch.from_numpy(multi_labels_embeding).float(), filename + self.suffix
        else:
            return img, torch.from_numpy(multi_labels_embeding).float()

    def _read_file(self, path, imageset_filenames=None):
        self.get_one_hot_map()
        data = []
        if self.label_type == 'csv':
            labels_df = pd.read_csv(path)
            length = len(labels_df['image_name'])
            for index in range(length):
                filename = labels_df['image_name'][index]
                labels = labels_df['tags'][index].split('/')
                multi_labels_embeding = np.zeros(len(self.one_hot_map))
                for label in labels:
                    if label != '':
                        multi_labels_embeding += self.one_hot_map[label]
                data.append((filename, multi_labels_embeding))
            return data
        else:
            if imageset_filenames is None:
                imageset_filenames = os.listdir(path)
            for filename in imageset_filenames:
                if filename == '': continue
                if filename == 'voc': continue
                if filename.count('.') <= 0:
                    filename += self.label_file_suffix
                labels = self._read_multi_label_file(os.path.join(path, filename))

                multi_labels_embeding = np.zeros(len(self.one_hot_map))
                for label in labels:
                    if label != '':
                        multi_labels_embeding += self.one_hot_map[label]
                data.append((filename.split('.')[0], multi_labels_embeding))
        return data

    def _read_multi_label_file(self, path):
        with open(path, 'r') as f:
            labels = f.read()
            if self.label_type == 'voc':
                root = ET.fromstring(labels)
                # 可能有多个重复标签
                labels = set([neighbor.find('name').text for neighbor in root.iter('object')])
            else:
                labels = labels.split('\n')
            return labels

    def _get_imageset_filenames(self, path):
        with open(os.path.join(path, self.train_type + '.txt'), 'r') as f:
            filenames = f.read()
            filenames = set(filenames.split('\n'))
        return filenames

    def get_one_hot_map(self):
        if len(self.one_hot_map) == 0:
            if self.label_type == 'voc':
                categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                              'dog', 'horse', 'motorbike', 'pottedplant',
                              'sheep', 'sofa', 'train', 'tvmonitor', 'person']
            else:
                categories = ['flower', 'fruit', 'leaf']
            one_hot = np.eye(len(categories))
            for index, cls_name in enumerate(sorted(categories)):
                # 每一个类对应一个one hot 编码
                oh = one_hot[index, :]
                self.one_hot_map[cls_name] = oh

        return self.one_hot_map

    def get_multi_labels(self, multi_labels_embeding):
        labels = []
        indexs = np.argwhere(multi_labels_embeding[0].numpy())
        # indexs = np.argwhere(multi_labels_embeding)
        for i in indexs:
            for key, values in self.one_hot_map.items():
                if np.argmax(values) == i:
                    labels.append(key)
        return labels


if __name__ == '__main__':
    # example 使用dataloader
    # data = PascalVOCType('/Users/sober/Downloads/pine_voc/ImageSets/Main', '/Users/sober/Downloads/pine_voc/JPEGImages')

    # data = MSIDataset('/home/khtt/code/pytorch-classification/tgmake/201810')
    # print('hello')
    # data = TP_Dataset('D:/Project/nor_255data/samples')    #ftype = 'fmsi'
    # for index, (img, label) in enumerate(data):
    #     print(index, img.shape, label, np.max(img), np.min(img))

    # face_dataset = MultiLabelDataset(txt_root_path='D:\Datasets\strawberry\label\disease_label\\test',
    #                                  img_root_path='D:\Datasets\strawberry\image\disease\\test\JPEGImages')
    face_dataset = MultiLabelDataset(txt_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\Annotations',
                                     img_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\JPEGImages',
                                     dataset_type='voc')

    fig = plt.figure()
    for i in range(600, len(face_dataset)):
        input, target = face_dataset[i]

        print(i, target)
        print(np.argwhere(target))
        # print(i, input.shape, target.shape)

        ax = plt.subplot(4, 3, i + 1 - 600)
        plt.tight_layout()
        ax.set_title('{}{}'.format(i, face_dataset.get_multi_labels(target)))
        ax.axis('off')
        img_2 = np.array(input)[:, :, [2, 1, 0]]
        plt.imshow(img_2)
        if i == 611:
            plt.show()
            break
