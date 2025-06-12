import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, num_classes=20):
        super().__init__()

        self.weight = torch.ones(num_classes)
        self.weight[0] = 2.5959737
        self.weight[1] = 6.741505
        self.weight[2] = 3.5353868
        self.weight[3] = 9.866315
        self.weight[4] = 9.690922
        self.weight[5] = 9.369371
        self.weight[6] = 10.289124 
        self.weight[7] = 9.953209
        self.weight[8] = 4.3098087
        self.weight[9] = 9.490392
        self.weight[10] = 7.674411
        self.weight[11] = 9.396925	
        self.weight[12] = 10.347794 	
        self.weight[13] = 6.3928986
        self.weight[14] = 10.226673 	
        self.weight[15] = 10.241072	
        self.weight[16] = 10.28059
        self.weight[17] = 10.396977
        self.weight[18] = 10.05567	
        self.weight[19] = 0
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.loss = torch.nn.NLLLoss(self.weight.to(device))

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs['logits'], dim=1), targets)
    

class DiceFocalLoss(nn.Module):
    def __init__(self, num_classes, weight=None, dice_smooth=1.0, gamma=2.0, alpha=0.25, ignore_index=None):
        super(DiceFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.dice_smooth = dice_smooth
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: (B, C, H, W) - raw logits
        targets: (B, H, W) - class index map
        """
        # Compute softmax over classes
        probs = F.softmax(inputs, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            probs = probs * mask.unsqueeze(1)
            targets_one_hot = targets_one_hot * mask.unsqueeze(1)

        # -------------------
        # Dice Loss
        # -------------------
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice_loss = 1.0 - ((2. * intersection + self.dice_smooth) / (cardinality + self.dice_smooth))
        dice_loss = dice_loss.mean()

        # -------------------
        # Focal Loss
        # -------------------
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = prob of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        focal_loss = focal_loss.mean()

        # Combine
        total_loss = dice_loss + focal_loss
        return total_loss


class DiceFocalLoss_updated(nn.Module):
    def __init__(self, num_classes, weight=None, dice_smooth=1.0, gamma=2.0, alpha=0.5,
                 ignore_index=None, dice_weight=0.7, focal_weight=None):
        super(DiceFocalLoss_updated, self).__init__()
        self.num_classes = num_classes
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if weight is not None:
            self.weight = weight.to(device)
        else:
            self.weight = None
        self.dice_smooth = dice_smooth
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.focal_weight = 1.0 - dice_weight if focal_weight is None else focal_weight

    def forward(self, inputs, targets):
        # Dice loss 계산용 softmax probs
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        dims = (2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice_loss = 1.0 - ((2. * intersection + self.dice_smooth) / (cardinality + self.dice_smooth))
        dice_loss = dice_loss.mean()

        # Focal loss 계산용 log_softmax + nll_loss
        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(log_probs, targets, weight=self.weight,
                             ignore_index=self.ignore_index, reduction='none')

        pt = torch.exp(-ce_loss)  # 확률
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        focal_loss = focal_loss.mean()

        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        return total_loss