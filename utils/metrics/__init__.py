import torch
from prettytable import PrettyTable
import numpy as np
from sklearn.metrics import fbeta_score, confusion_matrix, multilabel_confusion_matrix


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


def pass_threshold(output, threshold_val):
    return output > torch.tensor(threshold_val).cuda()


def f2_score(output, target, mask=None):
    if mask is not None:
        if len(output.shape) == 2:
            mask = torch.BoolTensor([mask, ] * output.shape[0])
        output = output[mask]
        target = target[mask]
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    # TP    predict 和 label 同时为1
    tp += ((output == 1) & (target.data == 1)).sum().float()
    # TN    predict 和 label 同时为0
    tn += ((output == 0) & (target.data == 0)).sum().float()
    # FN    predict 0 label 1
    fn += ((output == 0) & (target.data == 1)).sum().float()
    # FP    predict 1 label 0
    fp += ((output == 1) & (target.data == 0)).sum().float()

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
            mask = torch.BoolTensor([mask, ] * output.shape[0])
        output = output[mask]
        target = target[mask]
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    # TP    predict 和 label 同时为1
    tp += ((output == 1) & (target.data == 1)).sum().float()
    # TN    predict 和 label 同时为0
    tn += ((output == 0) & (target.data == 0)).sum().float()
    # FN    predict 0 label 1
    fn += ((output == 0) & (target.data == 1)).sum().float()
    # FP    predict 1 label 0
    fp += ((output == 1) & (target.data == 0)).sum().float()

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


def multi_label_confusion_matrix(y_true, y_pred, labels=None):
    matrix = multilabel_confusion_matrix(y_true, y_pred)
    if not labels:
        return matrix
    return show_multi_label_confusion_matrix(matrix, labels)


def multi_label_score_confusion_matrix(y_true, y_pred, num_labels, labels=None):
    scores_true, scores_pred = multi_label_score(y_true), multi_label_score(y_pred)
    matrix = confusion_matrix(scores_true, scores_pred, labels=range(num_labels))
    if not labels:
        return matrix
    return show_confusion_matrix(matrix, labels)


def show_multi_label_confusion_matrix(matrix, labels):
    result = ''
    for i, label in enumerate(labels):
        table = PrettyTable(['', 'not ' + label, label])
        table.add_row(['not ' + label, *[str(value) for value in matrix[i][0]]])
        table.add_row([label, *[str(value) for value in matrix[i][1]]])
        result += '\n' + str(table)
    return result


def show_confusion_matrix(matrix, labels):
    table = PrettyTable()
    indexes = []

    # find the indexes of  rows required
    for i in range(len(labels)):
        rows, columns = matrix[:, i], matrix[i, :]
        if any(columns) or any(rows):
            indexes.append(i)
            continue
    # remove the rows useless
    matrix = np.delete(matrix, [i for i in range(len(labels)) if i not in indexes], 0)

    # add the name of row
    table_columns = [('', [labels[i] for i in indexes])]
    # add the values
    for i in indexes:
        columns = matrix[:, i]
        table_columns.append((labels[i], columns))
    # show
    for label, columns in table_columns:
        table.add_column(label, columns)
    return str(table)


def multi_label_score(y):
    scores = None
    for y_true_one in y:
        indexes = np.argwhere(y_true_one)
        indexes_sum = indexes.sum()
        if indexes.shape[0] > 1:
            indexes_sum += indexes.shape[0]
        score = np.expand_dims(indexes_sum, 0)
        if scores is not None:
            scores = np.concatenate((scores, score), axis=0)
        else:
            scores = score
    return scores
