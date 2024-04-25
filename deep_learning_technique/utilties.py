from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from deep_learning_technique.config import *
from sklearn.metrics import jaccard_score

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

def l1_loss(input, target):
    """ L1 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(input - target))

##
def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)

def cos_loss(input, target, size_average=True):
    """ cosine Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: cosine distance between input and output
    """
    input = input.view(input.shape[0],-1)
    target = target.view(target.shape[0],-1)
    
    if size_average:
        return torch.mean(1-F.cosine_similarity(input, target))
    else:
        return 1-F.cosine_similarity(input, target)
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1
    
    def forward(self, predict, target):
#         target = target.unsqueeze(1)
#         print(predict.shape,target.shape)
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        
        pre = torch.sigmoid(predict).view(num, -1)
#         pre = predict.view(num, -1)
        tar = target.view(num, -1)
        
        intersection = (pre * tar).sum(-1).sum()  #利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()
        
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        
        return score
    
def get_errors(err_d, err_g):
    """ Get netD and netG errors.

    Returns:
        [OrderedDict]: Dictionary containing errors.
    """
    errors = OrderedDict([
        ('err_d', err_d.item()),
        ('err_g', err_g.item())])

    return errors

def f1_score(score, lb):
    lb = lb.cpu().numpy()
    score = np.array(score.detach().squeeze(0).cpu())
    threshold =THRESHOLD
    score[score>threshold] = 1.0
    score[score<=threshold] = 0.0 
    lb[lb>threshold] = 1.0
    lb[lb<=threshold] = 0.0 
    lb = lb[0,:,:]
    lb = np.round(lb)
    
    tp = np.sum(lb*score)
    fn = lb-score
    fn[fn<0]=0
    fn = np.sum(fn)
    tn = lb+score
    tn[tn>0]=-1
    tn[tn>=0]=1
    tn[tn<0]=0
    tn = np.sum(tn)
    fp = score - lb
    fp[fp<0] = 0
    fp = np.sum(fp)
    
    return tp, fp, tn, fn
    

def compute_jaccard_index(prediction, ground_truth):
    # This function remains the same    
    jaccard_indices = []
    ground_truth=ground_truth.cpu().numpy()
    prediction=prediction.cpu().numpy()
    for result_img,label_img in zip(prediction,ground_truth):
      # Compute Jaccard Index
      img_true=np.array(label_img).ravel()
      img_pred=np.array(result_img).ravel()
      img_pred=np.where(img_pred>THRESHOLD,255,0)
      img_true=np.where(img_true>THRESHOLD,255,0)
      jaccard_index = jaccard_score(img_true, img_pred,pos_label=255,zero_division=1)
      jaccard_indices.append(jaccard_index)
    # mean_jaccard_index = np.mean(jaccard_indices)
    # print("Mean Jaccard Index:", mean_jaccard_index)
    return jaccard_indices