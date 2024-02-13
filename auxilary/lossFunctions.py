
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp
from typing import Optional


def label_to_one_hot_label(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    ignore_index=255,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([
                [[0, 1], 
                [2, 0]]
            ])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    shape = labels.shape
    # one hot : (B, C=ignore_index+1, H, W)
    one_hot = torch.zeros((shape[0], ignore_index+1) + shape[1:], device=device, dtype=dtype)
    
    # labels : (B, H, W)
    # labels.unsqueeze(1) : (B, C=1, H, W)
    # one_hot : (B, C=ignore_index+1, H, W)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    
    # ret : (B, C=num_classes, H, W)
    ret = torch.split(one_hot, [num_classes, ignore_index+1-num_classes], dim=1)[0]
    
    return ret


# https://github.com/zhezh/focalloss/blob/master/focalloss.py
def focal_loss(input, target, alpha, gamma, reduction, eps, ignore_index):
    
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    # input : (B, C, H, W)
    n = input.size(0) # B
    
    # out_sie : (B, H, W)
    out_size = (n,) + input.size()[2:]
    
    # input : (B, C, H, W)
    # target : (B, H, W)
    # convert target to a single channel by taking the argmax along the channel dimension
    target = torch.argmax(target, dim=1)

    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")
    
    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)       
        

    # compute softmax over the classes axis
    # input_soft : (B, C, H, W)
    input_soft = F.softmax(input, dim=1) + eps
    
    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W)
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device, dtype=input.dtype, ignore_index=ignore_index)

    # compute the actual focal loss
    weight = torch.pow(1.0 - input_soft, gamma)
    
    # alpha, weight, input_soft : (B, C, H, W)
    # focal : (B, C, H, W)
    focal = -alpha * weight * torch.log(input_soft)
    
    # loss_tmp : (B, H, W)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        # loss : (B, H, W)
        loss = loss_tmp
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss

class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math:

        FL(p_t) = -alpha_t(1 - p_t)^{gamma}, log(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha, gamma = 2.0, reduction = 'mean', eps = 1e-8, ignore_index=30):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps, self.ignore_index)


# Jaccard Loss
def jaccard_loss(logits, true, class_weights=None, eps=1e-7):
    """
    Computes the Jaccard loss, a.k.a the IoU loss, with optional class weights.

    Args:
        logits: Tensor of shape [B, C, H, W] containing the raw output or logits of the model.
        true: Tensor of shape [B, H, W] containing the ground truth labels.
        class_weights: List or Tensor of shape [C] containing the weight for each class.
        eps: Epsilon value added to the denominator for numerical stability.

    Returns:
        jacc_loss: The Jaccard loss.
    """
    num_classes = logits.shape[1]

    # Apply class weights if provided
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=logits.dtype, device=logits.device, requires_grad=True)
        class_weights = class_weights.view(1, num_classes, 1, 1)

    true_1_hot = F.one_hot(true.squeeze(1), num_classes).permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())

    intersection = torch.sum(probas * true_1_hot, dim=(0, 2, 3))
    cardinality = torch.sum(probas + true_1_hot, dim=(0, 2, 3))
    union = cardinality - intersection

    # Apply class weights to the intersection and union
    if class_weights is not None:
        intersection = intersection * class_weights
        union = union * class_weights

    jacc_loss = (intersection / (union + eps)).mean()

    return (1 - jacc_loss)

class jaccLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights

    def setWeights(self, class_weights):
        self.class_weights = class_weights
    
    def forward(self, input, target):
        return jaccard_loss(input, target, class_weights=self.class_weights)
    

# Pixel Wise Cross Entropy Loss
def pixelwise_cross_entropy_loss(predictions, targets, class_weights):
    # Flatten the predictions and targets tensors
    predictions_flat = predictions.view(-1, predictions.size(1))
    targets_flat = targets.view(-1)
    # Compute the cross-entropy loss with class weights
    loss = F.cross_entropy(predictions_flat, targets_flat, weight=class_weights)
    return loss

class pwcel(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights    

    def setWeights(self, class_weights):
        self.class_weights = class_weights

    def forward(self, input, target):
        return pixelwise_cross_entropy_loss(input, target, self.class_weights)

    
# Weighted Dice Loss    
class weightedDiceLoss(nn.Module):
    def __init__(self, smooth=1., weights=None):
        super(weightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.weights = weights

    def setWeights(self, weights):
        self.weights = weights

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()

        # Calculate intersection and weighted union
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        union = (pred + target).sum(dim=2).sum(dim=2)
        
        # Calculate Dice coefficient for each class
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Dice Loss
        dice_loss = 1 - dice
        if self.weights is not None:
            dice_loss = dice_loss * self.weights

        return dice_loss.mean()


# Dice Loss
# Dice Loss
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

class diceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, target):
        return dice_loss(input, target)



# Modified Jaccard Loss
def mod_jaccard_loss(logits, true, class_weights=None, eps=1e-7):
    num_classes = logits.shape[1]

    # Apply class weights if provided
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=logits.dtype, device=logits.device)
        class_weights = class_weights.view(1, num_classes, 1, 1)

    true_1_hot = F.one_hot(true.squeeze(1), num_classes).permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())

    intersection = torch.sum(probas * true_1_hot, dim=(0, 2, 3))
    cardinality = torch.sum(probas + true_1_hot, dim=(0, 2, 3))
    union = cardinality - intersection

    # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
    TP = intersection
    FP = cardinality - intersection
    FN = torch.sum(true_1_hot, dim=(0, 2, 3)) - intersection
    TN = torch.sum(probas == 0, dim=(0, 2, 3))

    # Apply class weights to TP, FP, FN, and TN
    if class_weights is not None:
        TP = TP * class_weights
        FP = FP * class_weights
        FN = FN * class_weights
        TN = TN * class_weights

    # Calculate xi based on TP and FP rates
    tp_sum = torch.sum(TP)
    fp_sum = torch.sum(FP)
    if tp_sum.item() == 0 or fp_sum.item() == 0:
        xi = 1.0  # Set a default value if there are no TP or FP counts
    else:
        xi = fp_sum / tp_sum

    # Adjust the calculation of the loss to give more importance to wrongly classified classes
    weighted_intersection = TP / (TP + (torch.abs(xi) * (FP + FN)) + eps)
    jacc_loss = (weighted_intersection * union).sum() / (union.sum() + eps)

    return (1 - jacc_loss)

class modJaccLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights

    def setWeights(self, class_weights):
        self.class_weights = class_weights
    
    def forward(self, input, target):
        return mod_jaccard_loss(input, target, class_weights=self.class_weights)


# Custom Loss fuction that merges Jaccard loss and Focal loss
def jaccard_focal_loss(logits, true, class_weights=None, eps=1e-7, alpha=0.5, gamma=2.0):
    num_classes = logits.shape[1]
    # Apply class weights if provided
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=logits.dtype, device=logits.device, requires_grad=True)
        class_weights = class_weights.view(1, num_classes, 1, 1)
    true_1_hot = F.one_hot(true.squeeze(1), num_classes).permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    intersection = torch.sum(probas * true_1_hot, dim=(0, 2, 3))
    cardinality = torch.sum(probas + true_1_hot, dim=(0, 2, 3))
    union = cardinality - intersection
    # Apply class weights to the intersection and union
    if class_weights is not None:
        intersection = intersection * class_weights
        union = union * class_weights
    jacc_loss = (intersection / (union + eps)).mean()
    focal_loss = - alpha * (torch.pow((1. - intersection / cardinality), gamma)) * torch.log(
        intersection / cardinality)
    return (1 - jacc_loss) + focal_loss.mean() 


# Binary Cross Entropy Loss 2
def bce_loss2(logits, true):
    """
    Computes the binary cross-entropy loss.

    Args:
        logits: Tensor of shape [B, C, H, W] containing the raw output or logits of the model.
        true: Tensor of shape [B, H, W] containing the ground truth labels.
        class_weights: List or Tensor of shape [C] containing the weight for each class.
        eps: Epsilon value added to the denominator for numerical stability.

    Returns:
        bce_loss: The binary cross-entropy loss.
    """
    num_classes = logits.shape[1]

    true_1_hot = F.one_hot(true.squeeze(1), num_classes).permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())

    bce_loss = nn.BCELoss()(probas[:, 1, :, :], true_1_hot[:, 1, :, :])  # Use BCELoss on the positive class

    return 1 - bce_loss


# Binary Cross Entropy Loss 
def BCE_loss(pred,label):
    #bce_loss = nn.BCELoss(size_average=True)
    bce_loss = nn.BCELoss(reduction='mean')
    #pred = pred.squeeze(1).float()
    bce_out = bce_loss(pred, label.float())
    #print("bce_loss:", bce_out.data.cpu().numpy())
    return bce_out


# IOU Loss
def _iou(pred, target, size_average = True):

    # print pred and target shape
    #print(pred.shape)
    #print(target.shape)

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)

def IOU_loss(pred,label):
    iou_loss = IOU(size_average=True)
    iou_out = iou_loss(pred, label)
    return iou_out


# MSSSIM Loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).cuda()
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        #print("sim",sim)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

# MSSSIM - Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1 ):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible,
        # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)
    

# Loss used with Unet+3 -  Combining BCE, IOU and MSSSIM Losses
class unet_3Loss(torch.nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights

    def setWeights(self, class_weights):
        self.class_weights = class_weights

    def forward(self, input, target):
        iou_loss = IOU_loss(input, target)
        bce_loss = BCE_loss(input, target)
        #print("input",input.shape)
        #print("target",target.shape)
        msssim_loss =  1 -  msssim(input.float(), target.float(), window_size=11, size_average=True,
                                    normalize=True)

        cumul_loss = iou_loss + bce_loss + msssim_loss
        return cumul_loss


class improvedLoss(torch.nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights

    def setWeights(self, class_weights):
        self.class_weights = class_weights

    def forward(self, input, target):
        # combine Jaccard, Focal and msssim loss
        jacc_loss = jaccard_loss(input, target, class_weights=self.class_weights)
        foc_loss = 1 - focal_loss(input, target, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-8, ignore_index=30)
        msssim_loss =  1 -  msssim(input.float(), target.float(), window_size=11, size_average=True, normalize=True)

        cumul_loss = jacc_loss + foc_loss + msssim_loss
        return cumul_loss


'''class ClassRatioLoss(torch.nn.Module):
    def __init__(self):
        super(ClassRatioLoss, self).__init__()

    def forward(self, prediction, target):
        # Calculate the class ratio by summing along the class dimension
        class_counts = target.sum(dim=(0, 2, 3))
        total_counts = torch.sum(class_counts).float()
        class_ratio = class_counts.float() / total_counts

        # Invert the class ratio to assign higher weight to the minority class
        class_weights = 1 / (class_ratio + 1e-6)
        class_weights = class_weights / torch.sum(class_weights)

        # Compute the weighted cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(prediction, target.float(), weight=class_weights[None, :, None, None])

        return loss'''


class ClassRatioLoss(torch.nn.Module):
    def __init__(self):
        super(ClassRatioLoss, self).__init__()

    def forward(self, prediction, target):
        # Compute the class ratio for the target
        class_counts = target.sum(dim=(0, 2, 3))
        total_counts = torch.sum(class_counts).float()
        target_class_ratio = class_counts / total_counts

        # Compute the class ratio for the prediction (using softmax to ensure sum to 1)
        predicted_class_ratio = F.softmax(prediction.mean(dim=(0, 2, 3)), dim=0)

        # Compute the mean-squared error between the predicted and target class ratios
        loss = F.mse_loss(predicted_class_ratio, target_class_ratio)

        return loss


class ClassRatioLossPlus(torch.nn.Module):
    def __init__(self, num_classes=2, class_weights=[1.0, 1.0], segmentation_weight=0.5, temporal_decay=0.05):
        super(ClassRatioLossPlus, self).__init__()
        self.class_weights = torch.tensor(class_weights)
        self.segmentation_weight = segmentation_weight
        self.temporal_weights = None
        self.temporal_decay = temporal_decay

    def setWeights(self, class_weights):
        self.class_weights = class_weights

    def forward(self, prediction, target):
        # Compute the class ratio for the target
        class_counts = target.sum(dim=(0, 2, 3))
        total_counts = torch.sum(class_counts).float()
        target_class_ratio = class_counts / total_counts

        # Compute the class ratio for the prediction (using softmax to ensure sum to 1)
        predicted_class_ratio = F.softmax(prediction.mean(dim=(0, 2, 3)), dim=0)

        # Compute MSE and MAE between predicted and target class ratios
        mse_loss = F.mse_loss(predicted_class_ratio, target_class_ratio)
        mae_loss = F.l1_loss(predicted_class_ratio, target_class_ratio)
        class_ratio_loss = self.class_weights[0] * mse_loss + self.class_weights[1] * mae_loss

        # Temporal adjustment
        if self.temporal_weights is None:
            self.temporal_weights = target_class_ratio
        else:
            self.temporal_weights = (1 - self.temporal_decay) * self.temporal_weights + self.temporal_decay * target_class_ratio

        # Compute segmentation loss (e.g., cross-entropy)
        segmentation_loss = F.cross_entropy(prediction, target)

        # Combine the class ratio loss and segmentation loss
        loss = (1 - self.segmentation_weight) * class_ratio_loss + self.segmentation_weight * segmentation_loss

        return loss


class RBAF(torch.nn.Module):
    '''
    Ratio-Boundary-Aware Focal Loss (RBAF Loss)
    "Ratio" for the class ratio component.
    "Boundary-Aware" for the Jaccard loss, which is sensitive to the object boundaries.
    "Focal" for the Focal loss component.

    The hyperparameters alpha and beta control the balance between the three loss components. 

    '''
    def __init__(self, alpha=0.3, beta=0.3, gamma=2.0):
        super(RBAF, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, prediction, target):
        # Compute the class ratio for the target
        class_counts = target.sum(dim=(0, 2, 3))
        total_counts = torch.sum(class_counts).float()
        target_class_ratio = class_counts / total_counts

        # Compute the class ratio for the prediction
        predicted_class_ratio = F.softmax(prediction.mean(dim=(0, 2, 3)), dim=0)
        class_ratio_loss = F.mse_loss(predicted_class_ratio, target_class_ratio)

        # Jaccard Loss (IoU Loss)
        intersection = (prediction * target).sum(dim=(0, 2, 3))
        union = prediction.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) - intersection
        jaccard_loss = 1 - (intersection + 1e-6) / (union + 1e-6)

        # Focal Loss
        probs = torch.sigmoid(prediction)
        focal_loss = -target * torch.pow(1 - probs, self.gamma) * torch.log(probs + 1e-6)
        focal_loss = focal_loss.sum(dim=(0, 2, 3))

        # Combine the losses with weights alpha and beta
        loss = self.alpha * class_ratio_loss + self.beta * jaccard_loss.mean() + (1 - self.alpha - self.beta) * focal_loss.mean()

        return loss

class focalDiceLoss(nn.Module):
    def __init__(self, class_weights=None):
        super(focalDiceLoss, self).__init__()
        self.class_weights = class_weights

        # Initialize the weights for each loss as learnable parameters
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Weight for Dice Loss
        self.beta = nn.Parameter(torch.tensor(0.5))   # Weight for Focal Loss

    def setWeights(self, class_weights):
        self.class_weights = class_weights

    def forward(self, prediction, target):

        # Focal Loss
        pt = prediction * target + (1 - prediction) * (1 - target)
        focal_loss = -((1 - pt) ** 2) * torch.log(pt + 1e-8)

        if self.class_weights is not None:
            class_weights = self.class_weights.view(1, -1, 1, 1)
            focal_loss = focal_loss * class_weights
        focal_loss = focal_loss.mean()

        # Dice Loss
        intersection = (prediction * target).sum(dim=(0, 2, 3))
        union = prediction.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) - intersection
        dice_loss = 1 - (intersection + 1e-6) / (union + 1e-6)

        if self.class_weights is not None:
            dice_loss = dice_loss * self.class_weights

        dice_loss = dice_loss.mean()

        # Combine the losses with learnable weights alpha and beta
        loss = self.alpha * dice_loss + self.beta * focal_loss

        return loss

class WassersteinLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(WassersteinLoss, self).__init__()
        self.epsilon = epsilon

    def get_stats(self, segment):
        coords = torch.nonzero(segment)
        if coords.shape[0] == 0:
            return torch.Tensor([0]).to(segment.device), torch.Tensor([[0]]).to(segment.device)
        mean = torch.mean(coords.float(), dim=0)
        cov = self.torch_cov(coords.float())
        return mean, cov

    def torch_cov(self, m, rowvar=False):
        if m.size(0) == 1:
            return torch.Tensor([[self.epsilon]]).to(m.device)
        else:
            fact = 1.0 / (m.size(1) - 1)
            m -= torch.mean(m, dim=1, keepdim=True)
            mt = m.t()
            return fact * m.matmul(mt).squeeze() + self.epsilon * torch.eye(m.size(0)).to(m.device)

    def wasserstein_distance(self, mean1, cov1, mean2, cov2):
        term_1 = torch.norm(mean1 - mean2, p=2) ** 2
        term_2 = torch.trace(cov1 + cov2 - 2 * (cov2.sqrt().mm(cov1.mm(cov2.sqrt())).sqrt() + self.epsilon))
        return torch.sqrt(term_1 + term_2 + self.epsilon)

    def forward(self, pred, target):
        # Placeholder for loss
        total_loss = torch.zeros(1, requires_grad=True).to(pred.device)

        # Get the statistics for each nucleus in the prediction and target
        for i in range(pred.size(0)):  # assuming pred is of shape (batch_size, num_classes, height, width)
            for c in range(pred.size(1)):
                mean_pred, cov_pred = self.get_stats(pred[i, c])
                mean_target, cov_target = self.get_stats(target[i, c])

                # Calculate Wasserstein distance
                distance = self.wasserstein_distance(mean_pred, cov_pred, mean_target, cov_target)

                # Add to total loss
                total_loss += distance

        return total_loss / (pred.size(0) * pred.size(1))


if __name__ == '__main__':
    # sample usage to check if loss function is working with dummy data
    input = torch.randn(1, 1, 256, 256)
    target = torch.randn(1, 1, 256, 256)
    #print(input.shape)
    #print(target.shape)
    loss = unet_3Loss()
    print(loss(input, target))