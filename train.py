import argparse
import logging
import os
import random
import shutil
import sys
import time
import warnings
from datetime import datetime
from io import TextIOBase

import numpy as np
import shortuuid
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from prettytable import PrettyTable
from sklearn.metrics import fbeta_score
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import datasets

import models as models
from dataset.BaseDataset import MultiLabelDataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('-data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-dataset', type=str, default='strawberry_disease', help='dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=0, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-t-b', '--test-batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--num-classes', default=7, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('-pretrain', '--pretrain', dest='pretrain', action='store_true', default=True,
                    help='pretrain')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[81, 122],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-c', '--checkpoint', default='../result', type=str,
                    metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-resume', '--resume', dest='resume', action='store_true', default=True,
                    help='loding by latest checkpoint')
parser.add_argument('--resume-path',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-infer', '--inference', dest='inference', action='store_true', default=False,
                    help='inference model on validation set')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False,
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

args = parser.parse_args()
best_acc1 = 0

logger = logging.getLogger(parser.description)
uuid = shortuuid.uuid()
# checkpoint = os.path.join(args.checkpoint, args.arch + '_' + str(args.depth) + '_' + uuid)
pretrain_str = '_pretrain' if args.pretrain else ''
checkpoint_name = args.arch + '_' + str(args.dataset) + pretrain_str + '_' + uuid
checkpoint = os.path.join(args.checkpoint, checkpoint_name)
if args.resume and args.resume_path:
    roots = args.resume_path.split(os.sep)
    uuid = (roots[-2]).split('_')[-1]
    checkpoint_name = roots[-2]
    checkpoint = os.sep.join(roots[:-1])
writer = SummaryWriter('runs/' + checkpoint_name)


def main():
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.cuda.empty_cache()
    if not os.path.isdir(checkpoint):
        '''make dir if not exist'''
        os.makedirs(checkpoint)
    # log
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(os.path.join(checkpoint, "log.txt"))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)

    logger.addHandler(console)
    logger.addHandler(handler)
    sys.stdout = _LoggerFileWrapper(handler.baseFilename)
    # view args
    arg_table = PrettyTable()
    arg_table.field_names = ["Argument name", "Value"]
    for key in args.__dict__:
        arg_table.add_row([key, args.__dict__[key]])
    print('\n' + str(arg_table))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, checkpoint, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, checkpoint, args)


def main_worker(gpu, ngpus_per_node, checkpoint_path, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
            cardinality=args.cardinality,
            num_classes=args.num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            dropRate=args.drop,
        )
    elif args.arch.endswith('densenet'):
        model = models.__dict__[args.arch](
            num_classes=args.num_classes,
            depth=args.depth,
            growthRate=args.growthRate,
            compressionRate=args.compressionRate,
            dropRate=args.drop,
        )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
            num_classes=args.num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            dropRate=args.drop,
        )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
            num_classes=args.num_classes,
            # depth=args.depth,
            # block_name=args.block_name,
        )
    else:
        model = models.__dict__[args.arch](num_classes=args.num_classes)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    # view the visualization of the model
    summary(model, input_size=(3, 224, 224))
    # define loss function (criterion) and optimizer
    criterion = nn.BCELoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                              weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume and args.start_epoch != 0:
        # optionally resume from a checkpoint of specific epoch
        args.resume_path = os.sep.join(args.resume_path.split(os.sep)[:-1]) + '/checkpoint_epoch' + str(
            args.start_epoch - 1) + '.pth.tar'
    if args.resume and args.resume_path:
        if os.path.isfile(args.resume_path):
            print("=> loading checkpoint '{}'".format(args.resume_path))
            if args.gpu is None:
                checkpoint = torch.load(args.resume_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume_path, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume_path))

    cudnn.benchmark = True

    # Data loading code
    print('==> Preparing dataset')
    normalize = transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    if args.pretrain:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    if args.dataset == 'cifar10':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    if args.dataset == 'cifar10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        valset = testset
    elif args.dataset == 'voc2007':
        trainset = MultiLabelDataset(label_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\Annotations',
                                     img_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\JPEGImages',
                                     imageset_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\ImageSets\Main',
                                     label_type='voc', transform=transform_train)
        testset = MultiLabelDataset(label_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\Annotations',
                                    img_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\JPEGImages',
                                    imageset_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\ImageSets\Main',
                                    train_type='test',
                                    label_type='voc',
                                    transform=transform_test)
        valset = testset
    elif args.dataset == 'strawberry_disease':
        trainset = MultiLabelDataset(label_root_path='D:\Datasets\strawberry\label\disease_label\\train',
                                     img_root_path='D:\Datasets\strawberry\image\disease\\train\JPEGImages',
                                     transform=transform_train)
        testset = MultiLabelDataset(label_root_path='D:\Datasets\strawberry\label\disease_label\\test',
                                    img_root_path='D:\Datasets\strawberry\image\disease\\test\JPEGImages',
                                    transform=transform_test)
        valset = testset
    elif args.dataset == 'strawberry_normal':
        trainset = MultiLabelDataset(label_root_path='D:\Datasets\strawberry\label\\normal_label\\train',
                                     img_root_path='D:\Datasets\strawberry\image\\normal_dataset',
                                     transform=transform_train)
        testset = MultiLabelDataset(label_root_path='D:\Datasets\strawberry\label\\normal_label\\test',
                                    img_root_path='D:\Datasets\strawberry\image\\normal_dataset',
                                    transform=transform_test)
        valset = testset
    else:
        trainset, valset, testset = None, None, None

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None

    if args.inference:
        inference(transform_test, model, criterion, args)
        return
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    threshold = AverageMeter('threshold', '', 'list')
    threshold.update(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    if args.evaluate:
        validate(val_loader, model, criterion, args, threshold)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss, train_metric = train(train_loader, model, criterion, optimizer, args, epoch)
        # evaluate on validation set
        val_loss, val_metric = validate(val_loader, model, criterion, args, threshold, epoch)

        # remember best acc@1 and save checkpoint
        is_best = val_metric > best_acc1
        best_acc1 = max(val_metric, best_acc1)
        if is_best: print('Epoch ' + str(epoch) + ', this is the best yet.')
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint_path)
        writer.add_scalars('epoch_loss', {'train': train_loss, 'test': val_loss}, epoch)
        writer.add_scalars('epoch_f2_score', {'train': train_metric, 'test': val_metric}, epoch)


def train(train_loader, model, criterion, optimizer, args, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data loading time', ':6.3f')
    losses = AverageMeter('Loss', ':f')
    f2 = AverageMeter('f2', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, f2],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    lenth = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        f2_val = f2_score(output > 0.5, target)
        losses.update(loss.item(), images.size(0))
        f2.update(f2_val, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalars('iteration_loss', {'train': loss.item() / images.size(0)}, epoch * lenth + i)
        if i % args.print_freq == 0 or i == (lenth - 1):
            progress.display(i)
    writer.add_histogram('last_layer', list(model.parameters())[0], epoch)
    print()
    return losses.avg, f2.avg


def validate(val_loader, model, criterion, args, threshold, epoch=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':f')
    f2 = AverageMeter('f2', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, f2, threshold],
        prefix='Val: ')

    # switch to evaluate mode
    model.eval()

    lenth = len(val_loader)
    with torch.no_grad():
        end = time.time()
        # optimal threshold
        if args.evaluate is not True:
            for i, (image, target) in enumerate(val_loader):
                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)
                # compute output
                output = model(image)
                threshold_val = get_optimal_threshold(target.cpu().numpy(), output.cpu().numpy())
                threshold.update(np.array(threshold_val))
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            f2_val = f2_score(output > torch.tensor(threshold.avg).cuda(args.gpu), target)
            losses.update(loss.item(), images.size(0))
            f2.update(f2_val, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if epoch is not None:
                writer.add_scalars('iteration_loss', {'test': loss.item() / images.size(0)},
                                   epoch * lenth + i)
            if i % args.print_freq == 0 or i == (lenth - 1):
                progress.display(i)
        print()
        # TODO: this should also be done with the ProgressMeter
        print(' * F2 {f2.avg:.3f}'
              .format(f2=f2))

    return losses.avg, f2.avg


def inference(transform_test, model, criterion, args):
    if args.dataset == 'voc2007':
        testset = MultiLabelDataset(label_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\Annotations',
                                    img_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\JPEGImages',
                                    imageset_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\ImageSets\Main',
                                    train_type='train',
                                    label_type='voc',
                                    transform=transform_test)
        inferenceset = MultiLabelDataset(label_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\Annotations',
                                         img_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\JPEGImages',
                                         imageset_root_path='D:\Datasets\VOC\VOCdevkit\VOC2007\ImageSets\Main',
                                         train_type='train',
                                         label_type='voc',
                                         transform=None)
    elif args.dataset == 'strawberry_disease':
        testset = MultiLabelDataset(label_root_path='D:\Datasets\strawberry\label\disease_label\\test',
                                    img_root_path='D:\Datasets\strawberry\image\disease\\test\JPEGImages',
                                    transform=transform_test, requires_filename=True)
        inferenceset = MultiLabelDataset(label_root_path='D:\Datasets\strawberry\label\disease_label\\test',
                                         img_root_path='D:\Datasets\strawberry\image\disease\\test\JPEGImages',
                                         transform=None, requires_filename=True)
    elif args.dataset == 'strawberry_normal':
        testset = MultiLabelDataset(label_root_path='D:\Datasets\strawberry\label\\normal_label\\test',
                                    img_root_path='D:\Datasets\strawberry\image\\normal_dataset',
                                    transform=transform_test, requires_filename=True)

        inferenceset = MultiLabelDataset(label_root_path='D:\Datasets\strawberry\label\\normal_label\\test',
                                         img_root_path='D:\Datasets\strawberry\image\\normal_dataset',
                                         transform=None, requires_filename=True)
    else:
        testset, inferenceset = None, None

    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':f')
    f2 = AverageMeter('f2', ':6.2f')
    threshold = AverageMeter('threshold', '', 'list')
    threshold.update(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    progress = ProgressMeter(
        len(testset),
        [batch_time, losses, f2, threshold],
        prefix='inference: ')

    one_hot_map = inferenceset.get_one_hot_map()
    data = open(os.path.join(checkpoint, 'inference.txt'), 'a', encoding="utf-8")
    lenth = len(val_loader)
    with torch.no_grad():
        end = time.time()
        tp, tn, fn, fp = dict(), dict(), dict(), dict()
        print('threshold:', threshold.avg, file=data)

        for i, (image, target, filename) in enumerate(testset):
            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            if len(image.shape) < 4:
                image = image.unsqueeze(0)
                target = target.unsqueeze(0)
            output = model(image)
            loss = criterion(output, target)

            # measure accuracy and record loss
            f2_val = f2_score(output > torch.tensor(threshold.avg).cuda(args.gpu), target)
            for label, one_hot in one_hot_map.items():
                if label not in tp:
                    tp[label], tn[label], fn[label], fp[label] = 0, 0, 0, 0
                tp_, tn_, fn_, fp_, _, _, _, _ = compute_evaluation_metric(
                    output > torch.tensor(threshold.avg).cuda(args.gpu), target,
                    mask=one_hot)
                tp[label] += tp_
                tn[label] += tn_
                fn[label] += fn_
                fp[label] += fp_
            losses.update(loss.item(), image.size(0))
            f2.update(f2_val, image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == (lenth - 1):
                progress.display(i)
        result_table = PrettyTable()
        result_table.field_names = ["label", "tp", 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'f2']
        f2s = dict()
        for label, _ in one_hot_map.items():
            acc, p, r, f2__ = compute_evaluation_metric2(tp[label], tn[label], fn[label], fp[label],
                                                         {'a', 'p', 'r', 'f2'})
            result_table.add_row(
                [label, tp[label].cpu().numpy(), tn[label].cpu().numpy(), fp[label].cpu().numpy(),
                 fn[label].cpu().numpy(), acc.cpu().numpy(), p.cpu().numpy(), r.cpu().numpy(),
                 f2__.cpu().numpy()])
            f2s[label] = f2__.cpu().numpy()
        print('\n' + str(result_table))
        print('\n' + str(result_table), file=data)
    return f2.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename=None, filename_best=None):
    if filename is None:
        filename = 'checkpoint_epoch' + str(state['epoch'] - 1) + '.pth.tar'
    # if filename_best is None:
    #     filename_best = 'model_best_epoch' + str(state['epoch'] - 1) + '.pth.tar'
    filepath = os.path.join(checkpoint, 'checkpoint.pth.tar')
    torch.save(state, filepath)
    shutil.copyfile(filepath, os.path.join(checkpoint, filename))
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        # shutil.copyfile(filepath, os.path.join(checkpoint, filename_best))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', type='number'):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.type = type

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.name == 'threshold':
            self.avg = 0.5

    def update(self, val, n=1):
        # if self.name == 'threshold':
        #     return
        if self.count == 0 and self.type == 'list':
            self.val = np.zeros(val.shape)
            self.avg = np.zeros(val.shape)
            self.sum = np.zeros(val.shape)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\r' + '\t'.join(entries), end='', flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    # Learning rate has changed before resume, so cannot update it with args.lr
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.gamma


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # topk index
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # 펼쳐서 expand 비교하기 广播
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            # 有对的就行
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def f2_score(output, target, mask=None):
    if mask is not None:
        if len(output.shape) == 2:
            mask = torch.ByteTensor([mask, ] * output.shape[0])
        output = output[mask]
        target = target[mask]
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    # TP    predict 和 label 同时为1
    tp += ((output == 1) & (target.data == 1)).sum()
    # TN    predict 和 label 同时为0
    tn += ((output == 0) & (target.data == 0)).sum()
    # FN    predict 0 label 1
    fn += ((output == 0) & (target.data == 1)).sum()
    # FP    predict 1 label 0
    fp += ((output == 1) & (target.data == 0)).sum()

    acc = (tp + tn) / (tp + tn + fn + fp)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    # F1 = 2 * r * p / (r + p)c
    lamba_ = 2
    f2 = (1 + lamba_ ** 2) * (p * r / (lamba_ ** 2 * p + r))
    return f2


def compute_evaluation_metric(output, target, metrics=None, mask=None):
    if metrics is None:
        metrics = {'a'}
    if mask is not None:
        if len(output.shape) == 2:
            mask = torch.ByteTensor([mask, ] * output.shape[0])
        output = output[mask]
        target = target[mask]
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    # TP    predict 和 label 同时为1
    tp += ((output == 1) & (target.data == 1)).sum()
    # TN    predict 和 label 同时为0
    tn += ((output == 0) & (target.data == 0)).sum()
    # FN    predict 0 label 1
    fn += ((output == 0) & (target.data == 1)).sum()
    # FP    predict 1 label 0
    fp += ((output == 1) & (target.data == 0)).sum()

    acc = None
    p = None
    r = None
    f2 = None
    if 'a' in metrics:
        acc = (tp + tn) / (tp + tn + fn + fp)
    if 'p' in metrics or 'f2' in metrics:
        # if tp == 0:
        #     p = 0
        # else:
        p = tp / (tp + fp)
    if 'r' in metrics or 'f2' in metrics:
        # if tp == 0:
        #     r = 0
        # else:
        r = tp / (tp + fn)
    # F1 = 2 * r * p / (r + p)

    if 'f2' in metrics:
        lamba_ = 2
        f2 = (1 + lamba_ ** 2) * (p * r / (lamba_ ** 2 * p + r))
    return tp, tn, fn, fp, acc, p, r, f2


def compute_evaluation_metric2(tp, tn, fn, fp, metrics=None):
    if metrics is None:
        metrics = {'a'}
    acc = None
    p = None
    r = None
    f2 = None
    if 'a' in metrics:
        acc = (tp + tn) / (tp + tn + fn + fp)
    if 'p' in metrics or 'f2' in metrics:
        # if tp == 0:
        #     p = 0
        # else:
        p = tp / (tp + fp)
    if 'r' in metrics or 'f2' in metrics:
        # if tp == 0:
        #     r = 0
        # else:
        r = tp / (tp + fn)
    # F1 = 2 * r * p / (r + p)

    if 'f2' in metrics:
        lamba_ = 2
        f2 = (1 + lamba_ ** 2) * (p * r / (lamba_ ** 2 * p + r))
    return acc, p, r, f2


def fbeta(true_label, prediction):
    return fbeta_score(true_label, prediction, beta=2, average='samples')


def get_optimal_threshold(true_label, prediction, iterations=100, metric=fbeta):
    best_threshold = [0.2] * args.num_classes
    for t in range(args.num_classes):
        best_metric = 0
        temp_threshold = [0.2] * args.num_classes
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshold[t] = temp_value
            temp_metric = metric(true_label, prediction > temp_threshold)
            if temp_metric > best_metric:
                best_metric = temp_metric
                best_threshold[t] = temp_value
    return best_threshold


class _LoggerFileWrapper(TextIOBase):
    """
    sys.stdout = _LoggerFileWrapper(logger_file_path)
    Log with PRINT Imported from NNI
    """

    def __init__(self, logger_file_path):
        self.terminal = sys.stdout
        logger_file = open(logger_file_path, 'a')
        self.file = logger_file

    def write(self, s):
        self.terminal.write(s)
        if s != '\n':
            _time_format = '%m/%d/%Y, %I:%M:%S %p'
            cur_time = datetime.now().strftime(_time_format)
            self.file.write('[{}] PRINT '.format(cur_time) + s + '\n')
            self.file.flush()
        return len(s)


if __name__ == '__main__':
    main()
    # test the function of accuracy
    # output = torch.tensor([[1, 2, 3],
    #                        [6, 5, 4]])
    # target = torch.tensor([[1],
    #                        [0]]).squeeze()
    # accuracy(output, target, (1, 2))
    # threshold = AverageMeter('threshold', ':6.f', 'list')
    # for i in range(10):
    # x = np.array([1, 2, 3, 4, 5, 6])
    # threshold.update(x)
    # x = np.array([2, 2, 3, 4, 5, 6])
    # threshold.update(x)
    # x = np.array([3, 2, 1, 4, 5, 6])
    # threshold.update(x)
    # print(threshold.avg)
