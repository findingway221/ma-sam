import torch
from torch import nn

class CEDCLoss(nn.Module):
    def __init__(self, class_num, lambda_ce=1, lambda_dice=1):
        super().__init__()
        self.class_num = class_num
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice

        self.CEloss = nn.CrossEntropyLoss()
        self.DCloss = DiceLoss(self.class_num)

    def forward(self, pred, target):
        b, _, h, w = target.shape
        celoss = self.CEloss(pred, target.view(b, h, w).long())
        diceloss = self.DCloss(pred, target.view(b, h, w).long())

        loss = self.lambda_ce * celoss + self.lambda_dice * diceloss

        return loss


class DiceLoss(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num

    def binary_soft_dice_loss(self, output, target, eps=1e-8):
        target = target.float()
        b = target.shape[0]
        output = output.reshape(b,-1)
        target = target.reshape(b, -1)
        intersection = 2*torch.sum(output*target)+eps
        union = torch.sum(output)+torch.sum(target)+eps
        loss = 1-intersection/union
        return loss

    def forward(self, pred, gt):
        '''
        The dice loss for using softmax activation function
        :param pred: (b, num_class, x1, x2...)
        :param target: (b, x1, x2...)
        :return: multi-class softmax soft dice loss
        '''
        assert self.class_num == pred.shape[1], 'In dice loss: predict.shape[1] != num_class'

        pred = torch.softmax(pred, dim=1)

        lossList=[]
        for i in range (1, self.class_num):
            binary_loss = self.binary_soft_dice_loss(pred[:, i, ...], (gt == i).float())
            lossList.append(binary_loss)

        losses = torch.tensor(lossList)
        loss_mean = torch.mean(losses)

        return loss_mean