import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def weighted_cross_entropy(out,label,weight=None):
    if weight is not None:
        cross_entropy = F.cross_entropy(out,label,reduction='none')
        return torch.sum(weight*cross_entropy)/(torch.sum(weight)+1e-5)
    else:
        return F.cross_entropy(out,label)

def weighted_smooth_cross_entropy(out,label,weight=None):
    if weight is not None:
        cross_entropy = CrossEntropyLabelSmooth(num_classes=out.size(1),reduction=False)(out,label)
        return torch.sum(weight*cross_entropy)/(torch.sum(weight)+1e-5)
    else:
        return CrossEntropyLabelSmooth(num_classes=out.size(1),reduction=True)(out,label)


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss