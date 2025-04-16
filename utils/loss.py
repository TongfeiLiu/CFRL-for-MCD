import torch.nn as nn
import torch
# import torch.nn.functional as F

import numpy as np

def JS_loss(p, q, get_softmax=True):
    softmax2d = nn.Softmax2d()
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p = softmax2d(p)
        q = softmax2d(q)
        # p = F.softmax(p)
        # q = F.softmax(q)
    leg_mean = ((p + q) / 2).log()
    return (KLDivLoss(leg_mean, p) + KLDivLoss(leg_mean, q)) / 2


def Cosine(x, y):
    xx = torch.sum(x ** 2, dim=1) ** 0.5
    x = x / xx[:, np.newaxis]

    yy = torch.sum(y ** 2, dim=1) ** 0.5
    y = y / torch.unsqueeze(yy, dim=1)

    dist = 1-torch.dot(x, y.transpose())
    return dist

def l2_distance(x, y):
    l2_dis = torch.sum(((x - y) ** 2), dim=1)
    return l2_dis
