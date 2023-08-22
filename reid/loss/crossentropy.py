import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon=0.2):
    super(WeightedCrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

  def forward(self, inputs, targets, distmat):
    """
    Args:
      inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
      targets: ground truth labels with shape (num_classes)
    """
    final_distmat = torch.zeros((len(distmat), inputs.shape[1])).cuda()
    final_distmat[:, :distmat.shape[1]] = distmat
    final_distmat = final_distmat / final_distmat.sum()
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon * final_distmat
    #targets /= targets.sum()
    loss = (- targets * log_probs).mean(0).sum()
    return loss


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon=0.1):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

  def forward(self, inputs, targets):
    """
    Args:
      inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
      targets: ground truth labels with shape (num_classes)
    """
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (- targets * log_probs).mean(0).sum()
    return loss


class SoftEntropy(nn.Module):
  def __init__(self):
    super(SoftEntropy, self).__init__()
    self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    loss = (- F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
    return loss
