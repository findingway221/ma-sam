import torch

def multiclass_dice_coeff(pred, target, class_num, reduction='Mean', need_softmax=False, need_argmax=False):
    if need_softmax:
        predSoftmax = torch.softmax(pred, 1)
    else:
        predSoftmax = pred

    if need_argmax:
        pred = torch.argmax(predSoftmax, dim=1, keepdim=True)
    else:
        pred = predSoftmax

    pred = pred.to(torch.int32).contiguous()
    target = target.to(torch.int32).contiguous()

    assert (pred.shape == target.shape)
    diceList = []

    for c in range(class_num):
        overlap = ((pred == c) * (target == c)).sum()
        union = (pred == c).sum() + (target == c).sum()
        diceList.append(((2 * overlap + 1e-5) / (union + 1e-5)).item())

    if reduction == 'Mean':
        return sum(diceList[1:]) / len(diceList[1:])
    elif reduction == 'None':
        return diceList