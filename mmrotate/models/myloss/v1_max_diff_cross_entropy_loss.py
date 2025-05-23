# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from ..builder import ROTATED_LOSSES


@ROTATED_LOSSES.register_module()
class CrossEntropyLossWithMaxDiffAndEntropy(CrossEntropyLoss):
    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean', 
                 class_weight=None, ignore_index=None, loss_weight=1.0, 
                 avg_non_ignore=False):
        super().__init__(use_sigmoid, use_mask, reduction, class_weight, 
                         ignore_index, loss_weight, avg_non_ignore)

        # 设置更稳定的初始权重
        self.lambda_ce = torch.tensor(0.6) # 交叉熵损失的权重
        self.lambda_max_diff = torch.tensor(0.4)  # 最大差异损失的权重
        #self.lambda_entropy = nn.Parameter(torch.tensor(0.1))  # 熵损失的权重

    def forward(self, cls_score, label, weight=None, avg_factor=None, 
                reduction_override=None, ignore_index=None, **kwargs):
        """
        Forward function. 计算损失，并加入最大差异损失和熵损失。
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        # 计算交叉熵损失
        loss_cls = super().forward(
            cls_score, label, weight, avg_factor, reduction_override, 
            ignore_index, **kwargs)

        # 计算最大差异损失
        max_diff_loss = self._max_diff_loss(cls_score)

        # 计算熵损失（加入数值稳定性）
        #entropy_loss = self._entropy_loss(cls_score)

        # 最终损失 = λ_ce * 交叉熵损失 + λ_max_diff * 最大差异损失 + λ_entropy * 熵损失
        total_loss = (
            self.lambda_ce * loss_cls + 
            self.lambda_max_diff * max_diff_loss 
        )


        return total_loss

    def _max_diff_loss(self, cls_score):
        """
        计算最大概率与第二大概率之间的差异
        """
        # Softmax激活，获得每个类别的预测概率
        probs = F.softmax(cls_score, dim=-1)

        # 获取每个样本的最大预测概率和第二大预测概率
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        max_probs = sorted_probs[:, 0]
        second_max_probs = sorted_probs[:, 1]

        # 计算最大概率和次大概率之间的差异（确保不出现负数）
        max_diff = torch.abs(max_probs - second_max_probs)

        # 如果 max_diff 接近零，可以加上一个小常数，避免数值不稳定
        max_diff = torch.clamp(max_diff, min=1e-6)

        # 返回损失，平均所有样本的最大差异
        return max_diff.mean()

    # def _entropy_loss(self, cls_score):
    #     """
    #     计算熵损失，鼓励减少预测的不确定性
    #     """
    #     # Softmax激活，获得每个类别的预测概率
    #     probs = F.softmax(cls_score, dim=-1)

    #     # 计算每个样本的熵（避免 log(0)）
    #     entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)

    #     # 返回熵的平均损失
    #     return entropy.mean()
