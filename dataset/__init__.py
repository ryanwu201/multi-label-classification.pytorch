import torchvision.transforms as transforms
from torchvision import datasets

from .BaseDataset import *


def prepare_dataset(dataset_name, pretrained=False, on_linux=False):
    normalize = transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    if pretrained:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    if dataset_name == 'cifar10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.Resize((419, 419)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize((419, 419)),
        transforms.ToTensor(),
        normalize,
    ])
    if dataset_name == 'cifar10':
        trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
        valset = testset
    elif dataset_name == 'voc2007':
        root_path = 'D:\Datasets\VOC\VOCdevkit'
        if on_linux:
            root_path = '/home/ryan/dataset/'
        trainset = MultiLabelDataset(label_root_path=root_path + '/VOC2007/Annotations',
                                     img_root_path=root_path + '/VOC2007/JPEGImages',
                                     imageset_root_path=root_path + '/VOC2007/ImageSets/Main',
                                     label_type='voc', transform=transform_train)
        testset = MultiLabelDataset(label_root_path=root_path + '/VOC2007/Annotations',
                                    img_root_path=root_path + '/VOC2007/JPEGImages',
                                    imageset_root_path=root_path + '/VOC2007/ImageSets/Main',
                                    train_type='test',
                                    label_type='voc',
                                    transform=transform_test)
        valset = testset
    elif dataset_name == 'strawberry_disease':
        trainset = MultiLabelDataset(label_root_path='D:\Datasets\strawberry\label\disease_label\\train',
                                     img_root_path='D:\Datasets\strawberry\image\disease\\train\JPEGImages',
                                     transform=transform_train)
        testset = MultiLabelDataset(label_root_path='D:\Datasets\strawberry\label\disease_label\\test',
                                    img_root_path='D:\Datasets\strawberry\image\disease\\test\JPEGImages',
                                    transform=transform_test)
        valset = testset
    elif dataset_name == 'strawberry_normal':
        trainset = MultiLabelDataset(label_root_path='D:\Datasets\strawberry\label\\normal_label\\train',
                                     img_root_path='D:\Datasets\strawberry\image\\normal_dataset',
                                     transform=transform_train)
        testset = MultiLabelDataset(label_root_path='D:\Datasets\strawberry\label\\normal_label\\test',
                                    img_root_path='D:\Datasets\strawberry\image\\normal_dataset',
                                    transform=transform_test)
        valset = testset
    elif dataset_name == 'strawberry_new':
        trainset = MultiLabelDataset(label_root_path='D:\Datasets\multi-label-classification(strawberry)\\train.csv',
                                     img_root_path='D:\Datasets\multi-label-classification(strawberry)\images',
                                     label_type='csv',
                                     transform=transform_train)
        valset = MultiLabelDataset(label_root_path='D:\Datasets\multi-label-classification(strawberry)\\val.csv',
                                   img_root_path='D:\Datasets\multi-label-classification(strawberry)\images',
                                   label_type='csv',
                                   transform=transform_test)
        testset = MultiLabelDataset(label_root_path='D:\Datasets\multi-label-classification(strawberry)\\test.csv',
                                    label_type='csv',
                                    img_root_path='D:\Datasets\multi-label-classification(strawberry)\images',
                                    transform=transform_test)
    elif dataset_name == 'strawberry_03_05':
        root_path = 'D:\Datasets\multi-label-classification(strawberry)\\now\split'
        if on_linux:
            root_path = '/home/ryan/dataset/multi-label-classification(strawberry)/'
        trainset = MultiLabelDataset(
            label_root_path=root_path + '/train/annotations',
            img_root_path=root_path + '/train/images',
            label_type='naive',
            transform=transform_train)
        valset = MultiLabelDataset(
            label_root_path=root_path + '/test/annotations',
            img_root_path=root_path + '/test/images',
            label_type='naive',
            transform=transform_test)
        testset = MultiLabelDataset(
            label_root_path=root_path + '/test/annotations',
            label_type='naive',
            img_root_path=root_path + '/test/images',
            transform=transform_test)
    else:
        trainset, valset, testset = None, None, None

    return trainset, valset, testset
