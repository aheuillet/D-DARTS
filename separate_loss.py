import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ConvSeparateLoss', 'TriSeparateLoss', 'ConvAblationLoss']
class ConvSeparateLoss(nn.modules.loss._Loss):
    """Separate the weight value between each operations using L2"""
    def __init__(self, weight=0.1, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(ConvSeparateLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, input1, target1, input2, cell_mc, mean_mc):
        loss1 = F.cross_entropy(input1, target1)
        loss2 = -F.mse_loss(input2, torch.tensor(0.5, requires_grad=False).cuda())
        return loss1 + self.weight*loss2, loss1.detach().item(), loss2.detach().item()


class TriSeparateLoss(nn.modules.loss._Loss):
    """Separate the weight value between each operations using L1"""
    def __init__(self, weight=0.1, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(TriSeparateLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, input1, target1, input2, cell_mc, mean_mc):
        loss1 = F.cross_entropy(input1, target1)
        loss2 = -F.l1_loss(input2, torch.tensor(0.5, requires_grad=False).cuda())
        return loss1 + self.weight*loss2, loss1.detach().item(), loss2.detach().item()

class ConvAblationLoss(nn.modules.loss._Loss):
    """Separate the weight value between each operations using L2 and a cell-specific ablation study."""
    def __init__(self, weight=0.1, abl_weight=0.9, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(ConvAblationLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.weight = weight
        self.abl_weight = abl_weight
    
    def compute_ablation_loss(self, cell_mc, mean_mc):
        """
        Compute the ablation loss for a given cell from the marginal contributions.
        """
        indice = (cell_mc - mean_mc) / mean_mc if mean_mc != 0 else 0 #prevent division by zero
        return torch.tensor(indice).cuda()
        

    def forward(self, logits, target1, input2, cell_mc, mean_mc):
        loss1 = F.cross_entropy(logits, target1)
        loss2 = -F.mse_loss(input2, torch.tensor(0.5, requires_grad=False).cuda())
        loss3 = -self.compute_ablation_loss(cell_mc, mean_mc)
        return loss1 + self.weight*loss2 + self.abl_weight*loss3, loss1.detach().item(), loss2.detach().item(), loss3.detach().item()
