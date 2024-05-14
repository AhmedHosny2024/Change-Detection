import os
import time
import numpy as np
import torchvision.utils as vutils
from collections import OrderedDict
import torch
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from deep_learning_technique.config import *
from sklearn.metrics import jaccard_score
import cv2
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

def L1_loss(input, target):
    diff = torch.abs(input - target)
    weighted_diff = L1_WEIGHT * diff * target + L0_WEIGHT * diff * (1 - target)
    loss = torch.mean(weighted_diff)
    return loss


# def L1_loss(input, target):
#     loss=0
#     # loop on all pixels get number of pixels where imput is 1 and target is 0
#     for i in range(input.shape[0]):
#         for j in range(input.shape[1]):
#             if input[i][j]==1 and target[i][j]==0:
#                 loss+=G_WEIGHT
#             if input[i][j]==0 and target[i][j]==1:
#                 loss+=1
            
#     loss= np.mean(loss)
#     return loss

#======================================display================================================#

def plot_current_errors(epoch, counter_ratio, errors,vis):
        """Plot current errros.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            errors (OrderedDict): Error for the current epoch.
        """
        
        
#         plot_data = None
#         plot_res = None
#         if not hasattr('plot_data') or plot_data is None:
        plot_data = {}
        plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        plot_data['X'].append(epoch + counter_ratio)
        plot_data['Y'].append([errors[k] for k in plot_data['legend']])
        
        vis.line(win='wire train loss', update='append',
            X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
            Y=np.array(plot_data['Y']),
            opts={
                'title': 'CSA-CDGAN' + ' loss over time',
                'legend': plot_data['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Loss'
            })

        
def normalize(inp):
        """Normalize the tensor

        Args:
            inp ([FloatTensor]): Input tensor

        Returns:
            [FloatTensor]: Normalized tensor.
        """
        return (inp - inp.min()) / (inp.max() - inp.min() + 1e-5)
            
def display_current_images(reals, fakes, vis):
        """ Display current images.

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        reals = normalize(reals.cpu().numpy())
        fakes = normalize(fakes.cpu().numpy())
#         fixed = normalize(fixed.cpu().numpy())

        vis.images(reals, win=1, opts={'title': 'Reals'})
        vis.images(fakes, win=2, opts={'title': 'Fakes'})
#         vis.images(fixed, win=3, opts={'title': 'fixed'})
        
def get_errors(err_d, err_g):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """
        errors = OrderedDict([
            ('err_d', err_d.item()),
            ('err_g', err_g.item())])

        return errors
    
def save_current_images(reals, fakes,save_dir,name,k,label_names):
        """ Save images for epoch i.

        Args:
            epoch ([int])        : Current epoch
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        save_path = os.path.join(save_dir,name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # make fakes to cpu
        fakes = fakes.detach().cpu().numpy()
        reals = reals.detach().cpu().numpy()
        # make fakes binary
        for i in range(fakes.shape[0]):
            fakes[i][fakes[i]>0.5] = 255
            fakes[i][fakes[i]<=0.5] = 0
            reals[i][reals[i]>0.5] = 255
            reals[i][reals[i]<=0.5] = 0
            cv2.imwrite(save_path+"/pre_"+label_names[i+k*BATCH_SIZE],fakes[i][0])
            cv2.imwrite(save_path+"/gth"+label_names[i+k*BATCH_SIZE],reals[i][0])

def save_test_images(fakes,save_dir,name,k):
        """ Save images for epoch i.

        Args:
            epoch ([int])        : Current epoch
            reals ([FloatTensor]): Real Image
            fakes ([FloatTensor]): Fake Image
            fixed ([FloatTensor]): Fixed Fake Image
        """
        save_path = os.path.join(save_dir,name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # make fakes to cpu
        fakes = fakes.detach().cpu().numpy()
        reals = reals.detach().cpu().numpy()
        # make fakes binary
        for i in range(fakes.shape[0]):
            fakes[i][fakes[i]>0.5] = 255
            fakes[i][fakes[i]<=0.5] = 0
            reals[i][reals[i]>0.5] = 255
            reals[i][reals[i]<=0.5] = 0
            cv2.imwrite(save_path+"/"+str(i+k*BATCH_SIZE),fakes[i][0])

def save_weights(epoch,net,optimizer,save_path, model_name):
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learning_rate': optimizer.state_dict()['param_groups'][0]['lr'],
        }
    torch.save(checkpoint,os.path.join(save_path,'current_%s.pth'%(model_name)))
    if epoch % 20 == 0:
        torch.save(checkpoint,os.path.join(save_path,'%d_%s.pth'%(epoch,model_name)))
  
  
def plot_performance( epoch, performance, vis):
        """ Plot performance

        Args:
            epoch (int): Current epoch
            counter_ratio (float): Ratio to plot the range between two epoch.
            performance (OrderedDict): Performance for the current epoch.
        """
        plot_res = []
        plot_res = {'X': [], 'Y': [], 'legend': list(performance.keys())}
        plot_res['X'].append(epoch)
        plot_res['Y'].append([performance[k] for k in plot_res['legend']])
        vis.line(win='AUC', update='append',
            X=np.stack([np.array(plot_res['X'])] * len(plot_res['legend']), 1),
            Y=np.array(plot_res['Y']),
            opts={
                'title': 'Testing ' + 'Performance Metrics',
                'legend': plot_res['legend'],
                'xlabel': 'Epoch',
                'ylabel': 'Stats'
            },
        )  
  
    
