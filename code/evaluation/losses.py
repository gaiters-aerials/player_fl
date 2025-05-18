from helper import *
from configs import *

class MulticlassFocalLoss(nn.Module):
    def __init__(self, num_classes, alpha = 1, gamma=3, reduction="mean", device = DEVICE):
        super(MulticlassFocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma).to(device)
        self.reduction = reduction
        if isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha).to(device)
        else:  
            self.alpha = torch.full((num_classes,), alpha, dtype=torch.float32).to(device)
      
    def forward(self, inputs, targets):
        p = F.softmax(inputs, dim=-1) #probabilities
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1))
        ce_loss = F.cross_entropy(inputs, targets, reduction='none').unsqueeze(-1)
        p_t = torch.where(targets_one_hot == 1, p, 1-p)
        modulated_loss = (1. - p_t) ** self.gamma * ce_loss
        alpha_t = torch.where(targets_one_hot == 1, self.alpha, 1-self.alpha)
        loss = alpha_t * modulated_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss*10 #helps with numerical stabality