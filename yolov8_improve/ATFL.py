import torch
import torch.nn as nn


class AdaptiveThresholdFocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(AdaptiveThresholdFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        # self.reduction = loss_fcn.reduction
        # self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)  # 得出预测概率
        p_t = torch.Tensor(p_t)  # 将张量转化为pytorch张量，使其在pytorch中可以进行张量运算

        mean_pt = p_t.mean()
        p_t_list = []
        p_t_list.append(mean_pt)
        p_t_old = sum(p_t_list) / len(p_t_list)
        p_t_new = 0.05 * p_t_old + 0.95 * mean_pt
        # gamma =2
        gamma = -torch.log(p_t_new)
        # 处理大于0.5的元素
        p_t_high = torch.where(p_t > 0.5, (1.000001 - p_t) ** gamma, torch.zeros_like(p_t))

        # 处理小于0.5的元素
        p_t_low = torch.where(p_t <= 0.5, (1.5 - p_t) ** (-torch.log(p_t)), torch.zeros_like(p_t))  # # 将两部分结果相加
        modulating_factor = p_t_high + p_t_low
        loss *= modulating_factor
        # if self.reduction == 'mean':
        #     return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # else:  # 'none'
        return loss


# # Focal loss
# g = 1  # focal loss gamma
# if g > 0:
#     self.bce = AdaptiveThresholdFocalLoss(self.bce, g)