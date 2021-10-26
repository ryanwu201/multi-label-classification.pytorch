import os
import shutil

import torch


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename=None, filename_best=None):
    if filename is None:
        filename = 'checkpoint_epoch' + str(state['epoch'] - 1) + '.pth.tar'
    # if filename_best is None:
    #     filename_best = 'model_best_epoch' + str(state['epoch'] - 1) + '.pth.tar'
    filepath = os.path.join(checkpoint, 'checkpoint.pth.tar')
    torch.save(state, filepath)
    # shutil.copyfile(filepath, os.path.join(checkpoint, filename))
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        # shutil.copyfile(filepath, os.path.join(checkpoint, filename_best))
