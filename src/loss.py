# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : loss.py
# Time       ：27/3/2024 2:49 pm
# Author     ：Chuang Zhao
# version    ：python 
# Description：一些特殊的loss
"""
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

#
# from torch.autograd import Variable
#
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int,torch.long)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)
#
#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)
#
#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()

def focal_loss(y_pred, y_true, weight=None, alpha=0.25, gamma=2):
    sigmoid_p = nn.Sigmoid(y_pred)
    zeros = torch.zeros_like(sigmoid_p)
    pos_p_sub = torch.where(y_true > zeros,y_true - sigmoid_p,zeros)
    neg_p_sub = torch.where(y_true > zeros,zeros,sigmoid_p)
    per_entry_cross_ent = -alpha * (pos_p_sub ** gamma) * torch.log(torch.clamp(sigmoid_p,1e-8,1.0))-(1-alpha)*(neg_p_sub ** gamma)*torch.log(torch.clamp(1.0-sigmoid_p,1e-8,1.0))
    return per_entry_cross_ent.mean()

def softmax_loss(y_pred,y_true):
    y_pred = (1 - 2*y_true)*y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[...,:1])
    y_pred_neg = torch.cat((y_pred_neg,zeros),dim=-1)
    y_pred_pos = torch.cat((y_pred_pos,zeros),dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg,dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos,dim=-1)
    return torch.mean(neg_loss + pos_loss)

class FocalLossMultiLabel(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLossMultiLabel, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        sigmoid_inputs = torch.sigmoid(inputs)
        pos_loss = -targets * (1 - sigmoid_inputs)**self.gamma * torch.log(sigmoid_inputs + 1e-10)
        neg_loss = -(1 - targets) * sigmoid_inputs**self.gamma * torch.log(1 - sigmoid_inputs + 1e-10)
        loss = pos_loss + neg_loss
        return loss.mean()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # 分别计算正负例的概率
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # 非对称裁剪
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)  # 给 self.xs_neg 加上 clip 值

        # 先进行基本交叉熵计算
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            # 以下 4 行相当于做了个并行操作
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.mean()





class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        # print(p_s.device, p_t.device)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


loss_asy = AsymmetricLossOptimized()
loss_focal = FocalLossMultiLabel()




# def focal_binary_cross_entropy(logits, targets, gamma=2):
#     l = logits.reshape(-1)
#     t = targets.reshape(-1)
#     p = torch.sigmoid(l)
#     p = torch.where(t >= 0.5, p, 1-p)
#     logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
#     loss = logp*((1-p)**gamma)
#     loss = logits.shape[1]*loss.mean()
#     return loss
