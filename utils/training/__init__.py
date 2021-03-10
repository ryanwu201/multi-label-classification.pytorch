from ..metrics import fbeta


def adjust_learning_rate(optimizer, epoch, args):
    # Learning rate has changed before resume, so cannot update it with args.lr
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.gamma


def get_optimal_threshold(true_label, prediction, iterations=100, num_classes=None, metric=fbeta):
    best_threshold = [0.2] * num_classes
    for t in range(num_classes):
        best_metric = 0
        temp_threshold = [0.2] * num_classes
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshold[t] = temp_value
            temp_metric = metric(true_label, prediction > temp_threshold)
            if temp_metric > best_metric:
                best_metric = temp_metric
                best_threshold[t] = temp_value
    return best_threshold
