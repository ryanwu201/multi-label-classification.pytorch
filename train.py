import argparse
import logging
import os
import random
import sys
import time
import warnings

import cv2
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
from prettytable import PrettyTable
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import models as models
import utils.metrics as metrics
from dataset import prepare_dataset
from utils import AverageMeter, ProgressMeter, LoggerFileWrapper, save_checkpoint, CAMGenerator, adjust_learning_rate, \
    get_optimal_threshold, pass_threshold, get_bbox_from_heatmap, draw_bbox, save_bbox_to_xml, \
    show_confusion_matrix, show_multi_label_confusion_matrix, multi_label_score_confusion_matrix

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Classification Training')
parser.add_argument('-data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-dataset', type=str, default='strawberry_03_05', help='dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='se_resnet152',
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
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-t-b', '--test-batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--num-classes', default=3, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('-pretrain', '--pretrain', dest='pretrain', action='store_true', default=True,
                    help='pretrain')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[-1],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-c', '--checkpoint', default='../' + sys.path[0].split(os.sep)[-1] + '_result', type=str,
                    metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-tensorboard-path', '-tensorboard-path', default='../' + sys.path[0].split(os.sep)[-1] + '_runs',
                    type=str,
                    metavar='PATH',
                    help='path to save tensorboard output')
parser.add_argument('-resume', '--resume', dest='resume', action='store_true', default=False,
                    help='loding by latest checkpoint')
parser.add_argument('--resume-path',
                    default='../' + sys.path[0].split(os.sep)[
                        -1] + '_result' + os.sep + 'se_resnet152_strawberry_03_05_pretrain_AF6qk9ZZQ8Aq48atcVfsum' + os.sep + 'model_best' + '.pth.tar',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-visualize', '--visualize', dest='visualize', action='store_true', default=False,
                    help='visualize model on validation set when inference time')
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
parser.add_argument('--on-linux', action='store_true', default=os.name == 'posix',
                    help='System platform.')

best_acc1 = 0
writer = None
checkpoint = None


def main(logger, args):
    torch.set_default_tensor_type(torch.FloatTensor)
    torch.cuda.empty_cache()
    if not os.path.isdir(checkpoint):
        '''make dir if not exist'''
        os.makedirs(checkpoint)
    # log
    logger.setLevel(level=logging.DEBUG)
    log_file_name = "log.txt"
    log_file_name = 'inference.txt' if args.inference else log_file_name
    handler = logging.FileHandler(os.path.join(checkpoint, log_file_name))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)

    logger.addHandler(console)
    logger.addHandler(handler)
    sys.stdout = LoggerFileWrapper(handler.baseFilename)
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
    summary(model, input_size=(3, 419, 419))
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
                best_acc1 = torch.tensor(best_acc1)
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
    trainset, valset, testset = prepare_dataset(args.dataset, args.pretrain, args.on_linux)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    test_loader_args = {'dataset': testset, 'batch_size': args.test_batch_size, 'shuffle': False,
                        'num_workers': args.workers,
                        'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(**test_loader_args)
    if args.inference:
        inference(testset, test_loader_args, model, criterion, args)
        return
    threshold = AverageMeter('threshold', '', 'list')
    threshold.update(np.array([0.5] * args.num_classes))
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

        f2_val = metrics.f2_score(pass_threshold(output, 0.5), target)
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
                threshold_val = get_optimal_threshold(target.cpu().numpy(), output.cpu().numpy(),
                                                      num_classes=args.num_classes)
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
            f2_val = metrics.f2_score(pass_threshold(output, threshold.avg), target)
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


def inference(testset, test_loader_args, model, criterion, args):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    testset.requires_filename = True
    testset.requires_origin = True
    test_loader_args['dataset'] = testset
    test_loader = torch.utils.data.DataLoader(**test_loader_args)

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':f')
    f2 = AverageMeter('f2', ':6.2f')
    threshold = AverageMeter('threshold', '', 'list')
    threshold.update(np.array([0.5] * args.num_classes))
    progress = ProgressMeter(
        len(testset),
        [batch_time, losses, f2, threshold],
        prefix='inference: ')
    _CAMGenerator = CAMGenerator('layer4', model)

    one_hot_map = testset.get_one_hot_map()
    set_labels = testset.get_labels()
    connect_labels = testset.get_connect_labels()
    confusion_matrix = np.zeros((len(set_labels), 2, 2), np.int32)
    score_confusion_matrix = np.zeros((len(connect_labels), len(connect_labels)), np.int32)
    lenth = len(test_loader)
    with torch.no_grad():
        end = time.time()
        tp, tn, fn, fp = dict(), dict(), dict(), dict()
        print('threshold:', threshold.avg)

        for i, (image, target, filename, origin_image) in enumerate(testset):
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
            f2_val = metrics.f2_score(pass_threshold(output, threshold.avg), target)

            # confusion_matrix
            confusion_matrix += multilabel_confusion_matrix(pass_threshold(output, threshold.avg).cpu().int(),
                                                            target.cpu().int())

            score_confusion_matrix += multi_label_score_confusion_matrix(
                pass_threshold(output, threshold.avg).cpu().int().numpy(),
                target.cpu().int().numpy(), len(connect_labels))

            # get origin image
            origin_image = np.array(origin_image)

            width, height = origin_image.shape[1], origin_image.shape[0]

            inference_path = os.path.join(checkpoint, 'inference')
            if not os.path.isdir(inference_path):
                '''make dir if not exist'''
                os.makedirs(inference_path)
            bboxes = []
            for label, one_hot in one_hot_map.items():
                if label not in tp:
                    tp[label], tn[label], fn[label], fp[label] = 0, 0, 0, 0
                tp_, tn_, fn_, fp_, _, _, _, _ = metrics.compute_evaluation_metric(
                    pass_threshold(output, threshold.avg), target,
                    mask=one_hot)
                tp[label] += tp_
                tn[label] += tn_
                fn[label] += fn_
                fp[label] += fp_
                if not args.visualize: continue
                if (tp_ == 0 and fp_ == 0 and fn_ == 0) and tn_ > 0: continue
                # result text per image
                title_table = PrettyTable([''] + sorted(one_hot_map.keys()))
                title_table.add_row(['after sigmoid'] + [str(i) for i in output.cpu().numpy()[0]])
                title_table.add_row(['prediction'] + [str(i) for i in (
                    pass_threshold(output, threshold.avg)).cpu().int().numpy()[0]])
                title_table.add_row(['target'] + [str(i) for i in target.cpu().int().numpy()[0]])

                # generate class activation mapping
                class_idx = [np.argmax(one_hot)]
                CAMs = _CAMGenerator.generate(i, class_idx, width, height)

                type_ = 'unknown'
                type_ = 'tp' if tp_ > 0 else type_
                type_ = 'tn' if tn_ > 0 else type_
                type_ = 'fp' if fp_ > 0 else type_
                type_ = 'fn' if fn_ > 0 else type_
                save_path = os.path.join(inference_path, label + os.sep + type_)
                cam_save_path = os.path.join(save_path, 'cams')
                bbox_save_path = os.path.join(save_path, 'bbox')
                if not os.path.isdir(save_path):
                    '''make dir if not exist'''
                    os.makedirs(save_path)
                if not os.path.isdir(cam_save_path):
                    '''make dir if not exist'''
                    os.makedirs(cam_save_path)
                if not os.path.isdir(bbox_save_path):
                    '''make dir if not exist'''
                    os.makedirs(bbox_save_path)

                result = torch.tensor(origin_image)
                for j in range(len(class_idx)):
                    heatmap = CAMs[j]
                    colors = (255, 0, 0), (0, 165, 255), (0, 0, 255)
                    bboxes_per_label = get_bbox_from_heatmap(heatmap, 110, merge=label == 'leaf', label_name=label,
                                                             probability=output.cpu().numpy()[0][class_idx[j]],
                                                             color=colors[class_idx[j]])
                    bboxes.extend(bboxes_per_label)

                    origin_image_with_bbox = draw_bbox(origin_image, bboxes_per_label)
                    cv2.imwrite(os.path.join(bbox_save_path, filename), origin_image_with_bbox)
                    if j == 0:
                        result = torch.tensor(origin_image_with_bbox)
                    else:
                        result = torch.cat([result, torch.tensor(origin_image_with_bbox)], 1)
                    cv2.imwrite(os.path.join(cam_save_path, filename), heatmap)
                    heatmap_ = cv2.addWeighted(origin_image, 0.3, heatmap, 0.7, 0)
                    result = torch.cat([result, torch.tensor(heatmap_)], 1)

                # plot class activation map and save
                cv2.imwrite(os.path.join(save_path, filename), np.array(result))

            if args.visualize:
                # save image with bbox per image
                image_with_bbox_per_image = draw_bbox(origin_image, bboxes)
                cv2.imwrite(os.path.join(inference_path, filename), image_with_bbox_per_image)

                # save bbox to xml
                annotation_save_path = os.path.join(inference_path, 'annotations')
                if not os.path.isdir(annotation_save_path):
                    '''make dir if not exist'''
                    os.makedirs(annotation_save_path)
                save_bbox_to_xml(origin_image, bboxes, filename.split('.')[0], annotation_save_path)
            losses.update(loss.item(), image.size(0))
            f2.update(f2_val, image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == (lenth - 1):
                progress.display(i)

        # result per label
        result_table = PrettyTable()
        result_table.field_names = ["label", "tp", 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall', 'f2']
        f2s = dict()
        for label, _ in one_hot_map.items():
            acc, p, r, f2__ = metrics.compute_evaluation_metric2(tp[label], tn[label], fn[label], fp[label],
                                                                 {'a', 'p', 'r', 'f2'})
            result_table.add_row(
                [label, tp[label].cpu().numpy(), tn[label].cpu().numpy(), fp[label].cpu().numpy(),
                 fn[label].cpu().numpy(), acc.cpu().numpy(), p.cpu().numpy(), r.cpu().numpy(),
                 f2__.cpu().numpy()])
            f2s[label] = f2__.cpu().numpy()
        print('\n' + str(result_table))
        # result
        f2_table = PrettyTable(['', 'flower', 'fruit', 'leaf', 'mean'])
        mean = (f2s['flower'] + f2s['fruit'] + f2s['leaf']) / args.num_classes
        f2_table.add_row(['f2', f2s['flower'], f2s['fruit'], f2s['leaf'], mean])
        print('\n' + str(f2_table))

        str_confusion_matrix = show_multi_label_confusion_matrix(confusion_matrix, set_labels)
        str_score_confusion_matrix = show_confusion_matrix(score_confusion_matrix, connect_labels)
        print('\n' + str_confusion_matrix)
        print('\n' + str_score_confusion_matrix)

    return f2.avg


if __name__ == '__main__':
    args = parser.parse_args()
    best_acc1 = 0

    logger = logging.getLogger(parser.description)
    uuid = shortuuid.uuid()
    pretrain_str = '_pretrain' if args.pretrain else ''
    checkpoint_name = args.arch + '_' + str(args.dataset) + pretrain_str + '_' + uuid
    checkpoint = os.path.join(args.checkpoint, checkpoint_name)
    if args.resume and args.resume_path:
        roots = args.resume_path.split(os.sep)
        uuid = (roots[-2]).split('_')[-1]
        checkpoint_name = roots[-2]
        checkpoint = os.sep.join(roots[:-1])
    writer = SummaryWriter(os.path.join(args.tensorboard_path, checkpoint_name))

    main(logger, args)
