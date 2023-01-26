import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .polar_transform import polar_transform
from .parallel_transform import parallel_transform


def focal_loss(labels, preds, 
               alpha : float = 0.25, gamma: float = 2.0, 
               cutoff: float = 0.5, epsilon: float=1e-6):
    """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

    Args
        y_true: Tensor of target data from the generator with shape (B, N, num_classes).
        y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.
        cutoff: Positive prediction cutoff for soft targets

    Returns
        The focal loss of y_pred w.r.t. y_true.
    """
    assert labels.shape[-1]==preds.shape[-1]
    preds        = torch.clamp(preds, epsilon, 1-epsilon)

    # compute the focal loss
    alpha_factor = torch.ones_like(labels) * alpha
    alpha_factor = torch.where(labels>cutoff, alpha_factor, 1-alpha_factor)
    focal_weight = torch.where(labels>cutoff, 1-preds, preds)
    focal_weight = alpha_factor * focal_weight ** gamma

    bce = -labels*torch.log(preds)-(1-labels)*torch.log(1-preds)
    cls_loss = focal_weight * bce

    # compute the normalizer: the number of positive anchors
    normalizer = torch.sum(torch.sum(labels, dim=1)>0)
    normalizer = torch.max(torch.tensor(1).to(normalizer.device), normalizer)
    
    return torch.sum(cls_loss) / normalizer


def smooth_l1_loss(target, preds, weight, sigma: float = 3, size_average: bool = True):
    """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

    Args
        y_true: Tensor from the generator of shape (N, 4). 
        y_pred: Tensor from the network of shape (N, 4).
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        The smooth L1 loss of y_pred w.r.t. y_true.
    """
    assert target.shape[-1]==preds.shape[-1]
    sigma_squared = sigma ** 2
        
    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    diff = torch.abs(preds - target)
    regression_loss = torch.where(diff<1.0/sigma_squared,
                                  0.5*sigma_squared*torch.pow(diff, 2),
                                  diff-0.5/sigma_squared
                                  )
    if weight is None:
        regression_loss = regression_loss.sum()
        normalizer = max(1, target.shape[0])
    else:
        weight = weight.reshape(-1, 1)
        regression_loss = (regression_loss * weight).sum()
        normalizer = weight.sum() + 1e-6

    if size_average:
        return regression_loss/normalizer
    else:
        return regression_loss


def balanced_binary_loss(ytrue, ypred, glaucoma_custom_weight: int = 0, dilation_diameter=[11,11],
                         sampling_prob: float = 1.0, epsilon: float = 1e-6):

    assert len(ytrue.shape)==4
    if isinstance(dilation_diameter,int):
        dilation_diameter = [dilation_diameter]*ytrue.shape[1]
    ypred = torch.clamp(ypred, epsilon, 1-epsilon)

    # balanced loss
    loss_pos = -torch.sum(ytrue*torch.log(ypred), dim=(0,2,3))
    loss_pos = loss_pos/(torch.sum(ytrue,dim=(0,2,3))+epsilon)

    weight = 1 - ytrue # weights for negative samples
    weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
    # w0 = weight.sum(dim=(0,2,3))
    # print('----------')
    if glaucoma_custom_weight==1:
        rim = (ytrue[:,1,:,:]-ytrue[:,0,:,:])
        weight[:,0,:,:] = weight[:,0,:,:]+rim
    if glaucoma_custom_weight==2:
        for k in range(ytrue.shape[1]):
            filters = torch.ones((1,1,dilation_diameter[k],dilation_diameter[k]))/dilation_diameter[k]**2
            filters = filters.to(ytrue.device)
            padding = (dilation_diameter[k]-1)//2
            outputs = F.conv2d(ytrue[:,k,:,:].unsqueeze(1), filters, padding=padding)>0
            outputs = outputs.type(ytrue.dtype)
            # diff = outputs[:,0,:,:]-ytrue[:,k,:,:]
            # print(k,diff.sum()/ytrue[:,k,:,:].sum())
            weight[:,k,:,:] = weight[:,k,:,:]+outputs[:,0,:,:]-ytrue[:,k,:,:]
    # w1 = weight.sum(dim=(0,2,3))
    # wb = ytrue.sum(dim=(0,2,3))
    # print(w0/wb,w1/wb,(w1-w0)/w0)

    loss_neg = -torch.sum(weight*(1-ytrue)*torch.log(1-ypred), dim=(0,2,3))
    loss_neg = loss_neg/(torch.sum((1-ytrue)*weight,dim=(0,2,3))+epsilon)

    return (loss_pos+loss_neg)/2


def sigmoid_focal_loss(ytrue, ypred, flag_weight: int = 0, dilation_diameter=11,
                       sampling_prob: float = 1.0, 
                       alpha : float = 0.25, gamma: float = 2.0, 
                       cutoff: float = 0.5, epsilon: float=1e-6):
    """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

    Args
        ytrue: Tensor of target data from the generator with shape (B, N, num_classes).
        y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.
        cutoff: Positive prediction cutoff for soft targets

    Returns
        The focal loss of ypred w.r.t. ytrue.
    """
    assert len(ytrue.shape)==4
    ypred = torch.clamp(ypred, epsilon, 1-epsilon)

    # compute the focal loss
    alpha_factor = torch.ones_like(ytrue) * alpha
    alpha_factor = torch.where(ytrue>cutoff, alpha_factor, 1-alpha_factor)
    focal_weight = torch.where(ytrue>cutoff, 1-ypred, ypred)
    focal_weight = alpha_factor * focal_weight ** gamma

    if flag_weight==0:
        bce = -ytrue*torch.log(ypred)-(1-ytrue)*torch.log(1-ypred)
    else:
        weight = 1 - ytrue # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        if flag_weight==1:
            rim = (ytrue[:,1,:,:]-ytrue[:,0,:,:])
            weight[:,0,:,:] = weight[:,0,:,:]+rim
        if flag_weight==2:
            filters = torch.ones((1,1,dilation_diameter,dilation_diameter))/dilation_diameter**2
            filters = filters.to(ytrue.device)
            padding = (dilation_diameter-1)//2
            for k in range(ytrue.shape[1]):
                outputs = F.conv2d(ytrue[:,k,:,:].unsqueeze(1), filters, padding=padding)>0
                outputs = outputs.type(ytrue.dtype)
                weight[:,k,:,:] = weight[:,k,:,:]+outputs[:,0,:,:]-ytrue[:,k,:,:]

        bce = -ytrue*torch.log(ypred)-weight*(1-ytrue)*torch.log(1-ypred)

    cls_loss = focal_weight * bce
    cls_loss = torch.sum(cls_loss, dim=(0,2,3))

    # compute the normalizer: the number of positive anchors
    normalizer = torch.sum(ytrue, dim=(0,2,3))
    normalizer = torch.clamp(normalizer, 1)
    
    return cls_loss / normalizer


def mil_unary_sigmoid_loss(ypred, mask, gt_boxes, mode='all', 
                           focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
                           epsilon=1e-6):
    """ Compute the mil unary loss.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes: Tensor of boxes with (N, 6), where N is the number of bouding boxes in the batch,
                    the 6 elements of each row are [nb_img, class, x1, y1, x2, y2]
    Returns
        unary loss for each category (C,) if mode='balance'
        otherwise, the average unary loss (1,) if mode='all'
    """
    assert (mode=='all')|(mode=='balance')|(mode=='focal')|(mode=='mil_focal')
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes.shape[0]):
        nb_img = gt_boxes[nb_ob,0]
        c      = gt_boxes[nb_ob,1].item()
        box    = gt_boxes[nb_ob,2:]
        pred   = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
        # print('***',c,box, pred.shape)
        if pred.numel() == 0:
            continue
        ypred_pos[c].append(torch.max(pred, dim=0)[0])
        ypred_pos[c].append(torch.max(pred, dim=1)[0])
    
    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        v1 = torch.max(ypred*(1-mask), dim=2)[0]
        v2 = torch.max(ypred*(1-mask), dim=3)[0]
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        ## for negative class
        v1 = torch.max(ypred*(1-mask), dim=2)[0]
        v2 = torch.max(ypred*(1-mask), dim=3)[0]
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                bce_pos = -torch.log(torch.cat(ypred_pos[c], dim=0))
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    return losses


def mil_unary_approx_sigmoid_loss(ypred, mask, gt_boxes, mode='all', method='gm', gpower=4, 
                                  focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, epsilon=1e-6):
    assert (mode=='all')|(mode=='balance')|(mode=='focal')|(mode=='mil_focal')
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    if method=='gm':
        ypred_g = ypred**gpower
    elif method=='expsumr': #alpha-softmax function
        ypred_g = torch.exp(gpower*ypred)
    elif method=='explogs': #alpha-quasimax function
        ypred_g = torch.exp(gpower*ypred)
    num_classes = ypred.shape[1]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes.shape[0]):
        nb_img = gt_boxes[nb_ob,0]
        c      = gt_boxes[nb_ob,1].item()
        box    = gt_boxes[nb_ob,2:]
        pred   = ypred_g[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
        if method=='gm':
            prob0 = torch.mean(pred, dim=0)**(1.0/gpower)
            prob1 = torch.mean(pred, dim=1)**(1.0/gpower)
        elif method=='expsumr':
            pd_org = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
            prob0 = torch.sum(pd_org*pred,dim=0)/torch.sum(pred,dim=0)
            prob1 = torch.sum(pd_org*pred,dim=1)/torch.sum(pred,dim=1)
        elif method=='explogs':
            msk = mask[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
            prob0 = torch.log(torch.sum(pred,dim=0))/gpower - torch.log(torch.sum(msk,dim=0))/gpower
            prob1 = torch.log(torch.sum(pred,dim=1))/gpower - torch.log(torch.sum(msk,dim=1))/gpower
        ypred_pos[c].append(prob0)
        ypred_pos[c].append(prob1)
    
    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))
        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            neg = torch.clamp(1-ypred_neg[c], epsilon, 1-epsilon)
            bce_neg = -torch.log(neg)
            if len(ypred_pos[c])>0:
                pos = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pos)
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    return losses


def mil_unary_softmax_loss(ypred, mask, gt_boxes, mode='balance', epsilon=1e-6):
    """ Compute the mil unary loss.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        gt_boxes: Tensor of boxes with (N, 6), where N is the number of bouding boxes in the batch,
                    the 6 elements of each row are [nb_img, class, x1, y1, x2, y2]
    Returns
        unary loss for each category (C,) if mode='balance'
        otherwise, the average unary loss (1,) if mode='all'
    """
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(gt_boxes.shape[0]):
        nb_img = gt_boxes[nb_ob,0]
        c      = gt_boxes[nb_ob,1].item()
        box    = gt_boxes[nb_ob,2:]
        pred   = ypred[nb_img,c,box[1]:box[3]+1,box[0]:box[2]+1]
        ypred_pos[c].append(torch.max(pred, dim=0)[0])
        ypred_pos[c].append(torch.max(pred, dim=1)[0])

    ## for background class
    v1 = torch.max(ypred[:,-1,:,:]*mask[:,-1,:,:], dim=1)[0].flatten()
    v2 = torch.max(ypred[:,-1,:,:]*mask[:,-1,:,:], dim=2)[0].flatten()
    ypred_pos[num_classes-1].append(v1)
    ypred_pos[num_classes-1].append(v2)

    losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
    nb_sps = torch.zeros((num_classes,), device=ypred.device)
    for c in range(num_classes):
        bce = -torch.log(torch.cat(ypred_pos[c],dim=-1))
        losses[c] = bce.sum()
        nb_sps[c] = len(bce)

    if mode=='all':
        loss = torch.sum(losses)/torch.sum(nb_sps)
    elif mode=='balance':
        loss = losses/nb_sps
    return loss


def mil_polar_unary_sigmoid_loss(ypred, mask, gt_boxes, mode='all', alpha=0.25, gamma=2.0, 
                                 sampling_prob=1.0, epsilon=1e-6):
    """ Compute the mil unary loss from polar transformation.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        crop_boxes: Tensor of boxes with (N, 5), where N is the number of bouding boxes in the batch,
                    the 5 elements of each row are [nb_img, class, center_x, center_r, radius]
    Returns
        polar unary loss for each category (C,) if mode='balance'
        otherwise, the average polar unary loss (1,) if mode='all'
    """
    assert (mode=='all')|(mode=='balance')|(mode=='focal')
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index   = gt_boxes[:,0].type(torch.int32)
    ob_class_index = gt_boxes[:,1].type(torch.int32)
    ob_gt_boxes  = gt_boxes[:,2:]
    ob_centers   = torch.round((gt_boxes[:,2:4]+gt_boxes[:,4:])/2).type(torch.int32)
    ob_radius    = torch.sqrt((gt_boxes[:,4]-gt_boxes[:,2])**2+(gt_boxes[:,5]-gt_boxes[:,3])**2)/2
    ob_min_len   = torch.min(gt_boxes[:,4]-gt_boxes[:,2], gt_boxes[:,5]-gt_boxes[:,3])/2
    ypred_pos = {c:[] for c in range(num_classes)}
    extra = 5
    for nb_ob in range(gt_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c      = ob_class_index[nb_ob]
        r      = ob_radius[nb_ob].type(torch.int32) + extra
        radius = ob_radius[nb_ob].cpu().numpy() + extra
        cx,cy  = ob_centers[nb_ob,:] 
        xmin   = torch.clamp(cx-r,0)
        ymin   = torch.clamp(cy-r,0)
        pred   = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        msk    = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        center = torch.tensor((cy-ymin,cx-xmin), dtype=pred.dtype, device=pred.device)

        pred_polar = polar_transform(pred, center=center, radius=radius+extra, scaling='linear')
        msk_polar  = polar_transform(msk, center=center, radius=radius+extra, scaling='linear')>0.5
        prob = torch.max(pred_polar*msk_polar, dim=1)[0]
        valid = torch.sum(msk_polar, dim=1)>ob_min_len[nb_ob]/3
        ypred_pos[c.item()].append(prob[valid])

    if mode=='focal':
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        ## for negative class
        v1 = torch.max(ypred*(1-mask), dim=2)[0]
        v2 = torch.max(ypred*(1-mask), dim=3)[0]
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    
    return losses


def mil_polar_approx_sigmoid_loss(ypred, mask, gt_boxes, mode='all', method='gm', gpower=4, epsilon=1e-6):
    assert (mode=='all')|(mode=='balance')|(mode=='focal')
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    if method=='gm':
        ypred_g = ypred**gpower
    elif method=='expsumr': #alpha-softmax function
        ypred_g = torch.exp(gpower*ypred)
    elif method=='explogs': #alpha-quasimax function
        ypred_g = torch.exp(gpower*ypred)
    num_classes = ypred.shape[1]
    ob_img_index   = gt_boxes[:,0].type(torch.int32)
    ob_class_index = gt_boxes[:,1].type(torch.int32)
    ob_gt_boxes  = gt_boxes[:,2:]
    ob_centers   = torch.round((gt_boxes[:,2:4]+gt_boxes[:,4:])/2).type(torch.int32)
    ob_radius    = torch.sqrt((gt_boxes[:,4]-gt_boxes[:,2])**2+(gt_boxes[:,5]-gt_boxes[:,3])**2)/2
    ob_min_len   = torch.min(gt_boxes[:,4]-gt_boxes[:,2], gt_boxes[:,5]-gt_boxes[:,3])/2
    ypred_pos = {c:[] for c in range(num_classes)}
    extra = 5
    for nb_ob in range(gt_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c      = ob_class_index[nb_ob]
        r      = ob_radius[nb_ob].type(torch.int32) + extra
        radius = ob_radius[nb_ob].cpu().numpy() + extra
        cx,cy  = ob_centers[nb_ob,:] 
        xmin   = torch.clamp(cx-r,0)
        ymin   = torch.clamp(cy-r,0)
        pred   = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        msk    = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        center = torch.tensor((cy-ymin,cx-xmin), dtype=pred.dtype)

        pred_polar = polar_transform(pred, center=center, radius=radius, scaling='linear')
        msk_polar  = polar_transform(msk, center=center, radius=radius, scaling='linear')>0.5
        valid      = torch.sum(msk_polar, dim=1)>ob_min_len[nb_ob]/3
        pred_polar = pred_polar[valid,:]
        msk_polar  = msk_polar[valid,:]
        if method=='gm':
            prob = torch.sum((pred_polar**gpower)*msk_polar, dim=1)/torch.sum(msk_polar, dim=1)
            prob = prob**(1.0/gpower)
        elif method=='expsumr':
            w = torch.exp(gpower*pred_polar)
            prob = torch.sum(pred_polar*w*msk_polar,dim=1)/torch.sum(w*msk_polar,dim=1)
        elif method=='explogs':
            w = torch.exp(gpower*pred_polar)
            prob = torch.log(torch.sum(w*msk_polar,dim=1))/gpower - torch.log(torch.sum(msk_polar,dim=1).float())/gpower
        ypred_pos[c.item()].append(prob)

        # plt.figure()
        # plt.imshow(msk.cpu().numpy())
        # plt.plot(center[0].numpy(),center[1].numpy(),'r+')
        # plt.title(str(center.numpy()))
        # plt.savefig(str(nb_ob)+'.jpg')
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(msk_polar.cpu().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow(msk_polar[valid,:].cpu().numpy())
        # plt.title(str(ob_min_len[nb_ob].numpy()/4))
        # plt.savefig(str(nb_ob)+'_polar.png')

    ## for negative class
    if method=='gm':
        v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
        v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
    elif method=='expsumr':
        v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
        v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
    elif method=='explogs':
        v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
        v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
    ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
    ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))
    losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
    for c in range(num_classes):
        bce_neg = -torch.log(1-ypred_neg[c])
        if len(ypred_pos[c])>0:
            pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
            bce_pos = -torch.log(pred)
            if mode=='all':
                loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
            elif mode=='balance':
                loss = (bce_pos.mean()+bce_neg.mean())/2
        else:
            loss = bce_neg.mean()
        losses[c] = loss    
    return losses


def mil_parallel_unary_sigmoid_loss(ypred, mask, crop_boxes, angle_params=(0,45,5), mode='all', 
                                    focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
                                    obj_size=0, epsilon=1e-6):
    """ Compute the mil unary loss from parallel transformation.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
        crop_boxes: Tensor of boxes with (N, 5), where N is the number of bouding boxes in the batch,
                    the 5 elements of each row are [nb_img, class, center_x, center_r, radius]
    Returns
        polar unary loss for each category (C,) if mode='balance'
        otherwise, the average polar unary loss (1,) if mode='all'
    """
    assert (mode=='all')|(mode=='balance')|(mode=='focal')|(mode=='mil_focal')
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index   = crop_boxes[:,0].type(torch.int32)
    ob_class_index = crop_boxes[:,1].type(torch.int32)
    ob_crop_boxes  = crop_boxes[:,2:]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(crop_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c      = ob_class_index[nb_ob]
        radius = ob_crop_boxes[nb_ob,-1]

        extra = 5
        cx,cy,r = ob_crop_boxes[nb_ob,:].type(torch.int32)
        r = r + extra
        xmin = torch.clamp(cx-r,0)
        ymin = torch.clamp(cy-r,0)
        pred = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]
        msk  = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]

        index = torch.nonzero(msk[0]>0.5, as_tuple=True)
        y0,y1 = index[0].min(), index[0].max()
        x0,x1 = index[1].min(), index[1].max()
        box_h = y1-y0+1
        box_w = x1-x0+1
        # print('-----',box_h,box_w, y1,y0)

        if min(box_h, box_w) <= obj_size:
            parallel_angle_params = [0]
        else:
            parallel_angle_params = list(range(angle_params[0],angle_params[1],angle_params[2]))
        # print("#angles = {}".format(len(parallel_angle_params)))

        for angle in parallel_angle_params:
            pred_parallel = parallel_transform(pred, box_h, box_w, angle, is_mask=False)
            msk0, msk1  = parallel_transform(msk, box_h, box_w, angle, is_mask=True)
            pred_parallel0 = pred_parallel*msk0
            pred_parallel1 = pred_parallel*msk1
            flag0 = torch.sum(msk0[0], dim=0)>0.5
            prob0 = torch.max(pred_parallel0[0], dim=0)[0]
            prob0 = prob0[flag0]
            flag1 = torch.sum(msk1[0], dim=1)>0.5
            prob1 = torch.max(pred_parallel1[0], dim=1)[0]
            prob1 = prob1[flag1]
            if len(prob0)>0:
                ypred_pos[c.item()].append(prob0)
            if len(prob1)>0:
                ypred_pos[c.item()].append(prob1)
            # print(nb_ob,angle,len(prob0),len(prob1))
            # print(torch.unique(torch.sum(msk0[0], dim=0)))
            # print(torch.unique(torch.sum(msk1[0], dim=1)))
        #     plt.figure()
        #     plt.subplot(1,2,1)
        #     plt.imshow(msk0[0].cpu().numpy())
        #     plt.subplot(1,2,2)
        #     plt.imshow(msk1[0].cpu().numpy())
        #     plt.savefig('mask_'+str(angle)+'.png')
        # import sys
        # sys.exit()

    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        v1 = torch.max(ypred*(1-mask), dim=2)[0]
        v2 = torch.max(ypred*(1-mask), dim=3)[0]
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        ## for negative class
        v1 = torch.max(ypred*(1-mask), dim=2)[0]
        v2 = torch.max(ypred*(1-mask), dim=3)[0]
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss

    return losses


def mil_parallel_approx_sigmoid_loss(ypred, mask, crop_boxes, angle_params=(0,45,5), mode='all', 
                                     method='gm', gpower=4, 
                                     focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
                                     obj_size=0, epsilon=1e-6):
    assert (mode=='all')|(mode=='balance')|(mode=='focal')|(mode=='mil_focal')
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index   = crop_boxes[:,0].type(torch.int32)
    ob_class_index = crop_boxes[:,1].type(torch.int32)
    ob_crop_boxes  = crop_boxes[:,2:]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(crop_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c      = ob_class_index[nb_ob]
        radius = ob_crop_boxes[nb_ob,-1]

        extra = 5
        cx,cy,r = ob_crop_boxes[nb_ob,:].type(torch.int32)
        r = r + extra
        xmin = torch.clamp(cx-r,0)
        ymin = torch.clamp(cy-r,0)
        pred = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]
        msk  = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1][None,:,:]

        index = torch.nonzero(msk[0]>0.5, as_tuple=True)
        y0,y1 = index[0].min(), index[0].max()
        x0,x1 = index[1].min(), index[1].max()
        box_h = y1-y0+1
        box_w = x1-x0+1

        if min(box_h, box_w) <= obj_size:
            parallel_angle_params = [0]
        else:
            parallel_angle_params = list(range(angle_params[0],angle_params[1],angle_params[2]))
        
        for angle in parallel_angle_params:
            pred_parallel = parallel_transform(pred, box_h, box_w, angle, is_mask=False)
            msk0, msk1  = parallel_transform(msk, box_h, box_w, angle, is_mask=True)
            # plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(msk0[0].cpu().numpy())
            # plt.subplot(1,2,2)
            # plt.imshow(msk1[0].cpu().numpy())
            # plt.savefig('mask_'+str(angle)+'.png')
            pred_parallel = pred_parallel[0]
            msk0 = msk0[0]>0.5
            msk1 = msk1[0]>0.5
            flag0 = torch.sum(msk0, dim=0)>0.5
            flag1 = torch.sum(msk1, dim=1)>0.5
            pred_parallel0 = pred_parallel[:,flag0]
            pred_parallel1 = pred_parallel[flag1,:]
            msk0 = msk0[:,flag0]
            msk1 = msk1[flag1,:]
            # plt.figure()
            # if torch.sum(flag0)>0.5:
            #     plt.subplot(1,2,1)
            #     plt.imshow(msk0.cpu().numpy())
            # if torch.sum(flag1)>0.5:
            #     plt.subplot(1,2,2)
            #     plt.imshow(msk1.cpu().numpy())
            # plt.savefig('mask_'+str(angle)+'_crop.png')
            
            if torch.sum(flag0)>0.5:
                if method=='gm':
                    w = pred_parallel0**gpower
                    prob0 = torch.sum(w*msk0, dim=0)/torch.sum(msk0, dim=0)
                    prob0 = prob0**(1.0/gpower)
                elif method=='expsumr':
                    w = torch.exp(gpower*pred_parallel0)
                    prob0 = torch.sum(pred_parallel0*w*msk0,dim=0)/torch.sum(w*msk0,dim=0)
                elif method=='explogs':
                    w = torch.exp(gpower*pred_parallel0)
                    prob0 = torch.log(torch.sum(w*msk0,dim=0))/gpower - torch.log(torch.sum(msk0, dim=0))/gpower
                ypred_pos[c.item()].append(prob0)
            if torch.sum(flag1)>0.5:
                if method=='gm':
                    w = pred_parallel1**gpower
                    prob1 = torch.sum(w*msk1, dim=1)/torch.sum(msk1, dim=1)
                    prob1 = prob1**(1.0/gpower)
                elif method=='expsumr':
                    w = torch.exp(gpower*pred_parallel1)
                    prob1 = torch.sum(pred_parallel1*w*msk1,dim=1)/torch.sum(w*msk1,dim=1)
                elif method=='explogs':
                    w = torch.exp(gpower*pred_parallel1)
                    prob1 = torch.log(torch.sum(w*msk1,dim=1))/gpower - torch.log(torch.sum(msk1,dim=1))/gpower
                ypred_pos[c.item()].append(prob1)
            # print(nb_ob,angle,len(prob0),len(prob1))
        # import sys
        # sys.exit()

    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss

    return losses


def mil_polar_unary_softmax_loss(ypred, gt_points, bkg_gt, weights=None,
        pt_params={"valid_radius": 25, "extra": 5, "output_shape": [90, 30], "scaling": "linear"}, 
        mode='focal_all', focal_params={'gamma':2.0}, epsilon=1e-6):
    """ Compute the mil unary loss from polar transformation.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
    Returns
        polar unary loss
    """
    assert mode in ['focal_all', 'focal_foreground', 'ce_all', 'ce_balance']
    pt_params_used = copy.deepcopy(pt_params)
    dtype, device = ypred.dtype, ypred.device
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index   = gt_points[:,0].type(torch.int32)
    ob_class_index = gt_points[:,1].type(torch.int32)
    ob_gt_points  = gt_points[:,2:].round().type(torch.int32)
    ypred_pos = {c:[] for c in range(num_classes)}
    extra = pt_params_used.pop("extra")
    radius = pt_params_used.pop("valid_radius")
    r = radius + extra
    x_cord = torch.arange(2*radius + 1) - radius
    x_grid = x_cord.repeat(2*radius + 1).view(2*radius + 1, 2*radius + 1)
    distance = x_grid**2 + x_grid.t()**2
    mask = (distance <= distance[0, radius])
    base_mask = torch.full((2*r + 1, 2*r + 1), 0, dtype=dtype, device=device)
    base_mask[extra:extra+2*radius+1, extra:extra+2*radius+1] = mask
    m_center = torch.tensor((r, r), dtype=ypred.dtype, device=ypred.device)
    base_mask_polar = polar_transform(base_mask, center=m_center, radius=r, **pt_params_used)
    # base_mask_polar = polar_transform(base_mask, center=m_center, radius=r, scaling='linear')
    for nb_ob in range(gt_points.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c = ob_class_index[nb_ob]
        cx,cy = ob_gt_points[nb_ob,:] 
        xmin = torch.clamp(cx - r, 0)
        ymin = torch.clamp(cy - r, 0)
        xmax = torch.clamp(cx + r + 1, max=ypred.shape[-1])
        ymax = torch.clamp(cy + r + 1, max=ypred.shape[-2])
        pred = ypred[nb_img, c, ymin:ymax, xmin:xmax]
        p_center = torch.tensor((cy-ymin, cx-xmin), dtype=dtype, device=device)
        pred_polar = polar_transform(pred, center=p_center, radius=r, **pt_params_used)
        # pred_polar = polar_transform(pred, center=p_center, radius=r, scaling='linear')

        mask = base_mask
        msk_polar = base_mask_polar
        xmin = torch.clamp(r - cx, 0)
        ymin = torch.clamp(r - cy, 0)
        xmax = torch.clamp(cx + r - ypred.shape[-1], 0)
        ymax = torch.clamp(cy + r - ypred.shape[-2], 0)
        m_center = torch.tensor((r-ymin, r-xmin), dtype=dtype, device=device)
        if (xmin > 0) | (ymin > 0) | (xmax > 0) | (ymax > 0):
            mask = mask[ymin:, xmin:]
            if ymax > 0:
                mask = mask[:-ymax-1, :]
            if xmax > 0:
                mask = mask[:, :-xmax-1]
            msk_polar = polar_transform(mask, center=m_center, radius=r, **pt_params_used)
        msk_polar = msk_polar * (msk_polar > 0.5)
        flag = torch.sum(msk_polar,dim=1) > 0.5
        msk_polar = msk_polar[flag, :]
        pred_polar = pred_polar[flag, :]

        prob = torch.max(pred_polar*msk_polar, dim=1)[0]
        ypred_pos[c.item()].append(prob)

        # plt.figure()
        # plt.subplot(2,2,1)
        # plt.imshow(pred.detach().cpu().numpy())
        # plt.title(p_center.cpu().numpy())
        # plt.subplot(2,2,2)
        # plt.imshow(mask.cpu().numpy())
        # plt.title(m_center.cpu().numpy())
        # plt.subplot(2,2,3)
        # plt.imshow(pred_polar.T.detach().cpu().numpy())
        # plt.subplot(2,2,4)
        # plt.imshow(msk_polar.T.cpu().numpy())
        # plt.savefig(f"{nb_ob}_yx.jpg")
    # print(ypred[:,-1].shape, bkg_gt.shape)
    ypred_pos[num_classes-1].append(ypred[:, -1][bkg_gt==1])
    losses = []
    nb_samples = []
    for key, values in ypred_pos.items():
        if len(values) == 0:
            continue
        pred = torch.clamp(torch.cat(values), epsilon, 1-epsilon)
        nb_samples.append(len(pred))
        if 'focal' in mode:
            loss = -(1-pred).pow(focal_params["gamma"])*torch.log(pred)
        else:
            loss = -torch.log(pred)
        if weights is not None:
            loss = loss * weights[k]
        losses.append(loss.sum())
    losses = torch.stack(losses)
    nb_samples = torch.as_tensor(nb_samples, dtype=dtype, device=device)

    # print("MIL",losses.sum()/nb_samples.sum(), losses.sum(), nb_samples[-1], nb_samples[:-1].sum())
    # print("MIL", losses, nb_samples)

    if (mode == 'focal_all') | (mode == 'ce_all'):
        losses = losses.sum()/nb_samples.sum()
    elif mode == 'focal_foreground':
        losses = losses.sum()/nb_samples[:-1].sum()
    elif mode == 'ce_balance':
        losses = torch.mean(losses/torch.clamp(nb_samples, epsilon))
    
    return [losses]


def mil_polar_approx_softmax_loss(ypred, gt_points, bkg_gt, weights=None, method='gm', gpower=4, 
        pt_params={"valid_radius": 25, "extra": 5, "output_shape": [90, 30], "scaling": "linear"}, 
        mode='focal_all', focal_params={'gamma':2.0}, epsilon=1e-6):
    """ Compute the mil unary loss from polar transformation.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
    Returns
        polar unary loss
    """
    assert mode in ['focal_all', 'focal_foreground', 'ce_all', 'ce_balance']
    assert method in ['gm', 'expsumr', 'explogs']
    pt_params_used = copy.deepcopy(pt_params)
    dtype, device = ypred.dtype, ypred.device
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index   = gt_points[:,0].type(torch.int32)
    ob_class_index = gt_points[:,1].type(torch.int32)
    ob_gt_points  = gt_points[:,2:].round().type(torch.int32)
    ypred_pos = {c:[] for c in range(num_classes)}
    extra = pt_params_used.pop("extra")
    radius = pt_params_used.pop("valid_radius")
    r = radius + extra
    x_cord = torch.arange(2*radius + 1) - radius
    x_grid = x_cord.repeat(2*radius + 1).view(2*radius + 1, 2*radius + 1)
    distance = x_grid**2 + x_grid.t()**2
    mask = (distance <= distance[0, radius])
    base_mask = torch.full((2*r + 1, 2*r + 1), 0, dtype=dtype, device=device)
    base_mask[extra:extra+2*radius+1, extra:extra+2*radius+1] = mask
    m_center = torch.tensor((r, r), dtype=ypred.dtype, device=ypred.device)
    base_mask_polar = polar_transform(base_mask, center=m_center, radius=r, **pt_params_used)
    for nb_ob in range(gt_points.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c = ob_class_index[nb_ob]
        cx,cy = ob_gt_points[nb_ob,:] 
        xmin = torch.clamp(cx - r, 0)
        ymin = torch.clamp(cy - r, 0)
        xmax = torch.clamp(cx + r + 1, max=ypred.shape[-1])
        ymax = torch.clamp(cy + r + 1, max=ypred.shape[-2])
        pred = ypred[nb_img, c, ymin:ymax, xmin:xmax]
        p_center = torch.tensor((cy-ymin, cx-xmin), dtype=dtype, device=device)
        pred_polar = polar_transform(pred, center=p_center, radius=r, **pt_params_used)

        mask = base_mask
        msk_polar = base_mask_polar
        xmin = torch.clamp(r - cx, 0)
        ymin = torch.clamp(r - cy, 0)
        xmax = torch.clamp(cx + r - ypred.shape[-1], 0)
        ymax = torch.clamp(cy + r - ypred.shape[-2], 0)
        m_center = torch.tensor((r-ymin, r-xmin), dtype=dtype, device=device)
        if (xmin > 0) | (ymin > 0) | (xmax > 0) | (ymax > 0):
            mask = mask[ymin:, xmin:]
            if ymax > 0:
                mask = mask[:-ymax-1, :]
            if xmax > 0:
                mask = mask[:, :-xmax-1]
            msk_polar = polar_transform(mask, center=m_center, radius=r, **pt_params_used)
        msk_polar = msk_polar * (msk_polar > 0.5)
        flag = torch.sum(msk_polar,dim=1) > 0.5
        msk_polar = msk_polar[flag, :]
        pred_polar = pred_polar[flag, :]
        
        if method=='gm':
            w = pred_polar**gpower
            prob = torch.sum(w*msk_polar, dim=1)/torch.sum(msk_polar, dim=1)
            prob = prob**(1.0/gpower)
        elif method=='expsumr':
            w = torch.exp(gpower*pred_polar)
            prob = torch.sum(pred_polar*w*msk_polar,dim=1)/torch.sum(w*msk_polar,dim=1)
        elif method=='explogs':
            w = torch.exp(gpower*pred_polar)
            prob = torch.log(torch.sum(w*msk_polar,dim=1))/gpower - torch.log(torch.sum(msk_polar, dim=1))/gpower
        ypred_pos[c.item()].append(prob)

        # a = torch.sum(msk_polar,dim=1)
        # if (a==0).sum() > 0:
        #     print("-------unique")
        #     print(torch.unique(msk_polar))
        #     print("-------------sum")
        #     print(a)
        #     flag = torch.sum(msk_polar,dim=1) > 0.5
        #     print("-------------sum pos")
        #     print(a[flag])
        #     print(a)
        #     plt.figure()
        #     plt.subplot(2,2,1)
        #     plt.imshow(pred.detach().cpu().numpy())
        #     plt.title(p_center.cpu().numpy())
        #     plt.subplot(2,2,2)
        #     plt.imshow(mask.cpu().numpy())
        #     plt.title(m_center.cpu().numpy())
        #     plt.subplot(2,2,3)
        #     plt.imshow(pred_polar.T.detach().cpu().numpy())
        #     plt.subplot(2,2,4)
        #     plt.imshow(msk_polar.T.cpu().numpy())
        #     plt.savefig(f"{nb_ob}_yx.jpg")
    
    ypred_pos[num_classes-1].append(ypred[:, -1][bkg_gt==1])
    losses = []
    nb_samples = []
    for key, values in ypred_pos.items():
        if len(values) == 0:
            continue
        pred = torch.clamp(torch.cat(values), epsilon, 1-epsilon)
        nb_samples.append(len(pred))
        if 'focal' in mode:
            loss = -(1-pred).pow(focal_params["gamma"])*torch.log(pred)
        else:
            loss = -torch.log(pred)
        if weights is not None:
            loss = loss * weights[k]
        # print("pred: ", pred.max(), pred.min())
        # print("loss: ", loss.max(), loss.min())
        # print("isnan: ", torch.isnan(loss).sum())
        losses.append(loss.sum())
    losses = torch.stack(losses)
    nb_samples = torch.as_tensor(nb_samples, dtype=dtype, device=device)

    if (mode == 'focal_all') | (mode == 'ce_all'):
        losses = losses.sum()/nb_samples.sum()
    elif mode == 'focal_foreground':
        losses = losses.sum()/nb_samples[:-1].sum()
    elif mode == 'ce_balance':
        losses = torch.mean(losses/torch.clamp(nb_samples, epsilon))
    
    return [losses]


def mil_polar_approx_sigmoid_loss(ypred, mask, crop_boxes, mode='all', 
        center_mode='fixed', method='gm', gpower=4, 
        focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
        pt_params={"output_shape": [90, 30], "scaling": "linear"}, 
        smoothness_weight=-1, epsilon=1e-6):
    """ Compute the mil unary loss from polar transformation.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
    Returns
        polar unary loss
    """
    assert mode in ['all', 'balance', 'focal', 'mil_focal']
    assert method in ['gm', 'expsumr', 'explogs']
    assert center_mode in ['fixed', 'estimated']
    pt_params_used = copy.deepcopy(pt_params)
    ph, pw = pt_params['output_shape']
    assert (ph > 0) & ((pw == -1) | (pw > 0))
    dtype, device = ypred.dtype, ypred.device
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index   = crop_boxes[:,0].type(torch.int32)
    ob_class_index = crop_boxes[:,1].type(torch.int32)
    ob_crop_boxes  = crop_boxes[:,2:]
    if smoothness_weight > 0:
        center_weight = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        pairwise_weights_list = [
            torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]]),
            ]
        smooth_loss = []
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(crop_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c      = ob_class_index[nb_ob]
        cx,cy,r = ob_crop_boxes[nb_ob,:].type(torch.int32)
        r = r + 5
        xmin = torch.clamp(cx-r,0)
        ymin = torch.clamp(cy-r,0)
        pred = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        msk  = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        if pw == -1:
            pt_params_used['output_shape'] = [ph, r.item()]

        if center_mode == 'fixed':
            p_center = torch.tensor((cy-ymin, cx-xmin), dtype=dtype, device=device)
        else:
            pred_msk = pred * (msk >= 0.5)
            cands = (pred_msk == torch.max(pred_msk)).nonzero()
            rand = torch.randperm(cands.shape[0])
            p_center = cands[rand[0], :]

        pred_polar = polar_transform(pred, center=p_center, radius=r, **pt_params_used)
        msk_polar = polar_transform(msk, center=p_center, radius=r, **pt_params_used)
        
        msk_polar = msk_polar * (msk_polar > 0.5)
        flag = torch.sum(msk_polar,dim=1) > 0.5
        msk_polar = msk_polar[flag, :]
        pred_polar = pred_polar[flag, :]
        # print(pred_polar.shape, pred_polar.shape)
        
        if method=='gm':
            w = pred_polar**gpower
            prob = torch.sum(w*msk_polar, dim=1)/torch.sum(msk_polar, dim=1)
            prob = prob**(1.0/gpower)
        elif method=='expsumr':
            w = torch.exp(gpower*pred_polar)
            prob = torch.sum(pred_polar*w*msk_polar,dim=1)/torch.sum(w*msk_polar,dim=1)
        elif method=='explogs':
            w = torch.exp(gpower*pred_polar)
            prob = torch.log(torch.sum(w*msk_polar,dim=1))/gpower - torch.log(torch.sum(msk_polar, dim=1))/gpower
        ypred_pos[c.item()].append(prob)

        if smoothness_weight > 0:
            for w in pairwise_weights_list:
                weights = center_weight - w
                weights = weights.view(1, 1, 3, 3).to(device)
                aff_map = F.conv2d(pred_polar[None, None], weights, padding=1)
                cur_loss = aff_map**2
                cur_loss = torch.sum(cur_loss[:,:,1:-1,1:-1]*msk_polar[None, None, 1:-1, 1:-1])/(torch.sum(msk_polar[1:-1, 1:-1]+epsilon))
                smooth_loss.append(cur_loss)
    
    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    if smoothness_weight > 0:
        smooth_loss = torch.stack(smooth_loss).mean() * smoothness_weight
        losses = torch.cat([losses, smooth_loss.reshape(-1)])
    return losses


def mil_polar_approx_pseudolabel_sigmoid_loss(ypred, mask, crop_boxes, mode='all', 
        center_mode='fixed', method='gm', gpower=4, 
        focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
        pt_params={"output_shape": [90, 30], "scaling": "linear"}, 
        smoothness_weight=-1, pseudolabel=True, pseudolabel_threshold=0.8, epsilon=1e-6):
    """ Compute the mil unary loss from polar transformation.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
    Returns
        polar unary loss
    """
    assert mode in ['all', 'balance', 'focal', 'mil_focal']
    assert method in ['gm', 'expsumr', 'explogs']
    assert center_mode in ['fixed', 'estimated']
    pt_params_used = copy.deepcopy(pt_params)
    ph, pw = pt_params['output_shape']
    assert (ph > 0) & ((pw == -1) | (pw > 0))
    dtype, device = ypred.dtype, ypred.device
    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index   = crop_boxes[:,0].type(torch.int32)
    ob_class_index = crop_boxes[:,1].type(torch.int32)
    ob_crop_boxes  = crop_boxes[:,2:]
    if smoothness_weight > 0:
        center_weight = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        pairwise_weights_list = [
            torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]]),
            ]
        smooth_loss = []
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(crop_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c      = ob_class_index[nb_ob]
        cx,cy,r = ob_crop_boxes[nb_ob,:].type(torch.int32)
        r = r + 5
        xmin = torch.clamp(cx-r,0)
        ymin = torch.clamp(cy-r,0)
        pred = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        msk  = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        if pw == -1:
            pt_params_used['output_shape'] = [ph, r.item()]

        if center_mode == 'fixed':
            p_center = torch.tensor((cy-ymin, cx-xmin), dtype=dtype, device=device)
        else:
            pred_msk = pred * (msk >= 0.5)
            cands = (pred_msk == torch.max(pred_msk)).nonzero()
            rand = torch.randperm(cands.shape[0])
            p_center = cands[rand[0], :]

        pred_polar = polar_transform(pred, center=p_center, radius=r, **pt_params_used)
        msk_polar = polar_transform(msk, center=p_center, radius=r, **pt_params_used)
        
        pseudo_mask = pred_polar > pseudolabel_threshold
        msk_polar = msk_polar * (msk_polar > 0.5)
        if pseudolabel:
            ind = torch.nonzero(msk_polar*pseudo_mask, as_tuple=True)
            prob_pseudo = pred_polar[ind]
            ypred_pos[c.item()].append(prob_pseudo)

        msk_polar = msk_polar * torch.logical_not(pseudo_mask)
        flag = torch.sum(msk_polar,dim=1) > 0.5
        msk_polar = msk_polar[flag, :]
        pred_polar = pred_polar[flag, :]
        
        if method=='gm':
            w = pred_polar**gpower
            prob = torch.sum(w*msk_polar, dim=1)/torch.sum(msk_polar, dim=1)
            prob = prob**(1.0/gpower)
        elif method=='expsumr':
            w = torch.exp(gpower*pred_polar)
            prob = torch.sum(pred_polar*w*msk_polar,dim=1)/torch.sum(w*msk_polar,dim=1)
        elif method=='explogs':
            w = torch.exp(gpower*pred_polar)
            prob = torch.log(torch.sum(w*msk_polar,dim=1))/gpower - torch.log(torch.sum(msk_polar, dim=1))/gpower
        ypred_pos[c.item()].append(prob)
        # print(prob_pseudo.shape, prob.shape)

        if pseudolabel & (smoothness_weight > 0):
            for w in pairwise_weights_list:
                weights = center_weight - w
                weights = weights.view(1, 1, 3, 3).to(device)
                aff_map = F.conv2d(pred_polar[None, None], weights, padding=1)
                cur_loss = aff_map**2
                cur_loss = torch.sum(cur_loss[:,:,1:-1,1:-1]*msk_polar[None, None, 1:-1, 1:-1])/(torch.sum(msk_polar[1:-1, 1:-1]+epsilon))
                smooth_loss.append(cur_loss)
    
    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    if pseudolabel & (smoothness_weight > 0):
        smooth_loss = torch.stack(smooth_loss).mean() * smoothness_weight
        losses = torch.cat([losses, smooth_loss.reshape(-1)])
    return losses


def mil_polar_approx_weighted_sigmoid_loss(ypred, mask, crop_boxes, mode='all', 
        weight_min=0.5, center_mode='fixed', method='gm', gpower=4, 
        focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
        pt_params={"output_shape": [90, 30], "scaling": "linear"}, 
        smoothness_weight=-1, epsilon=1e-6):
    """ Compute the mil unary loss from polar transformation.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
    Returns
        polar unary loss
    """
    assert mode in ['all', 'balance', 'focal', 'mil_focal']
    assert method in ['gm', 'expsumr', 'explogs']
    assert center_mode in ['fixed', 'estimated']
    pt_params_used = copy.deepcopy(pt_params)
    ph, pw = pt_params['output_shape']
    assert (ph > 0) & ((pw == -1) | (pw > 0))
    dtype, device = ypred.dtype, ypred.device
    if pw > 0:
        d = pt_params["output_shape"][1]
        sigma2 = (d-1)**2 / (-2*math.log(weight_min))
        polar_weights = torch.exp(-torch.arange(d, dtype=dtype, device=device)**2/(2*sigma2))
        polar_weights = polar_weights.repeat(pt_params["output_shape"][0], 1)

    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index   = crop_boxes[:,0].type(torch.int32)
    ob_class_index = crop_boxes[:,1].type(torch.int32)
    ob_crop_boxes  = crop_boxes[:,2:]
    if smoothness_weight > 0:
        center_weight = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        pairwise_weights_list = [
            torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]]),
            ]
        smooth_loss = []
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(crop_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c      = ob_class_index[nb_ob]
        cx,cy,r = ob_crop_boxes[nb_ob,:].type(torch.int32)
        r = r + 5
        xmin = torch.clamp(cx-r,0)
        ymin = torch.clamp(cy-r,0)
        pred = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        msk  = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        if pw == -1:
            pt_params_used['output_shape'] = [ph, r.item()]
            sigma2 = (pt_params_used['output_shape'][1]-1)**2 / (-2*math.log(weight_min))
            polar_weights = torch.exp(-torch.arange(pt_params_used['output_shape'][1], dtype=dtype, device=device)**2/(2*sigma2))
            polar_weights = polar_weights.repeat(ph, 1)

        if center_mode == 'fixed':
            p_center = torch.tensor((cy-ymin, cx-xmin), dtype=dtype, device=device)
        else:
            pred_msk = pred * (msk >= 0.5)
            cands = (pred_msk == torch.max(pred_msk)).nonzero()
            rand = torch.randperm(cands.shape[0])
            p_center = cands[rand[0], :]

        pred_polar = polar_transform(pred, center=p_center, radius=r, **pt_params_used)
        msk_polar = polar_transform(msk, center=p_center, radius=r, **pt_params_used)
        pred_polar = pred_polar * polar_weights
        
        msk_polar = msk_polar * (msk_polar > 0.5)
        flag = torch.sum(msk_polar,dim=1) > 0.5
        msk_polar = msk_polar[flag, :]
        pred_polar = pred_polar[flag, :]
        # print(pred_polar.shape, pred_polar.shape)
        
        if method=='gm':
            w = pred_polar**gpower
            prob = torch.sum(w*msk_polar, dim=1)/torch.sum(msk_polar, dim=1)
            prob = prob**(1.0/gpower)
        elif method=='expsumr':
            w = torch.exp(gpower*pred_polar)
            prob = torch.sum(pred_polar*w*msk_polar,dim=1)/torch.sum(w*msk_polar,dim=1)
        elif method=='explogs':
            w = torch.exp(gpower*pred_polar)
            prob = torch.log(torch.sum(w*msk_polar,dim=1))/gpower - torch.log(torch.sum(msk_polar, dim=1))/gpower
        ypred_pos[c.item()].append(prob)

        if smoothness_weight > 0:
            for w in pairwise_weights_list:
                weights = center_weight - w
                weights = weights.view(1, 1, 3, 3).to(device)
                aff_map = F.conv2d(pred_polar[None, None], weights, padding=1)
                cur_loss = aff_map**2
                cur_loss = torch.sum(cur_loss[:,:,1:-1,1:-1]*msk_polar[None, None, 1:-1, 1:-1])/(torch.sum(msk_polar[1:-1, 1:-1]+epsilon))
                smooth_loss.append(cur_loss)
    
    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss
    if smoothness_weight > 0:
        smooth_loss = torch.stack(smooth_loss).mean() * smoothness_weight
        losses = torch.cat([losses, smooth_loss.reshape(-1)])
    return losses


def mil_polar_approx_weighted2_sigmoid_loss(ypred, mask, crop_boxes, mode='all', 
        weight_min=0.5, center_mode='fixed', method='gm', gpower=4, 
        focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
        pt_params={"output_shape": [90, 30], "scaling": "linear"}, 
        epsilon=1e-6):
    """ Compute the mil unary loss from polar transformation.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
    Returns
        polar unary loss
    """
    assert mode in ['all', 'balance', 'focal', 'mil_focal']
    assert method in ['gm', 'expsumr', 'explogs']
    assert center_mode in ['fixed', 'estimated']
    pt_params_used = copy.deepcopy(pt_params)
    dtype, device = ypred.dtype, ypred.device
    d = pt_params["output_shape"][1]
    sigma2 = (d-1)**2 / (-2*math.log(weight_min))
    polar_weights = torch.exp(-torch.arange(d, dtype=dtype, device=device)**2/(2*sigma2))
    polar_weights = polar_weights.repeat(pt_params["output_shape"][0], 1)

    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index   = crop_boxes[:,0].type(torch.int32)
    ob_class_index = crop_boxes[:,1].type(torch.int32)
    ob_crop_boxes  = crop_boxes[:,2:]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(crop_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c      = ob_class_index[nb_ob]
        extra = 5
        cx,cy,r = ob_crop_boxes[nb_ob,:].type(torch.int32)
        r = r + extra
        xmin = torch.clamp(cx-r,0)
        ymin = torch.clamp(cy-r,0)
        pred = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        msk  = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]

        if center_mode == 'fixed':
            p_center = torch.tensor((cy-ymin, cx-xmin), dtype=dtype, device=device)
        else:
            pred_msk = pred * (msk >= 0.5)
            cands = (pred_msk == torch.max(pred_msk)).nonzero()
            rand = torch.randperm(cands.shape[0])
            p_center = cands[rand[0], :]

        pred_polar = polar_transform(pred, center=p_center, radius=r, **pt_params_used)
        msk_polar = polar_transform(msk, center=p_center, radius=r, **pt_params_used)
        
        msk_polar = msk_polar * (msk_polar > 0.5)
        flag = torch.sum(msk_polar,dim=1) > 0.5
        msk_polar = msk_polar[flag, :]
        pred_polar = pred_polar[flag, :]
        # print(pred_polar.shape, pred_polar.shape)
        
        if method=='gm':
            w = pred_polar**gpower * polar_weights[flag, :]
            prob = torch.sum(w*msk_polar, dim=1)/torch.sum(msk_polar, dim=1)
            prob = prob**(1.0/gpower)
        elif method=='expsumr':
            w = torch.exp(gpower*pred_polar) * polar_weights[flag, :]
            prob = torch.sum(pred_polar*w*msk_polar,dim=1)/torch.sum(w*msk_polar,dim=1)
        elif method=='explogs':
            w = torch.exp(gpower*pred_polar) * polar_weights[flag, :]
            prob = torch.log(torch.sum(w*msk_polar,dim=1))/gpower - torch.log(torch.sum(msk_polar, dim=1))/gpower
        ypred_pos[c.item()].append(prob)
    
    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss

    return losses


def mil_polar_approx_dist_weighted_sigmoid_loss(ypred, mask, crop_boxes, mode='all', 
        weight_min=0.5, center_mode='fixed', method='gm', gpower=4, 
        polar_dist=0, center_dist=1, 
        focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
        pt_params={"output_shape": [90, 30], "scaling": "linear"}, 
        epsilon=1e-6):
    """ Compute the mil unary loss from polar transformation.
    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
    Returns
        polar unary loss
    """
    assert mode in ['all', 'balance', 'focal', 'mil_focal']
    assert method in ['gm', 'expsumr', 'explogs']
    assert center_mode in ['fixed', 'estimated']
    pt_params_used = copy.deepcopy(pt_params)
    dtype, device = ypred.dtype, ypred.device
    d = pt_params["output_shape"][1] - polar_dist
    sigma2 = (d-1)**2 / (-2*math.log(weight_min))
    polar_weights = torch.exp(-torch.arange(d, dtype=dtype, device=device)**2/(2*sigma2))
    polar_weights = polar_weights.repeat(pt_params["output_shape"][0], 1)

    ypred =  torch.clamp(ypred, epsilon, 1-epsilon)
    num_classes = ypred.shape[1]
    ob_img_index   = crop_boxes[:,0].type(torch.int32)
    ob_class_index = crop_boxes[:,1].type(torch.int32)
    ob_crop_boxes  = crop_boxes[:,2:]
    ypred_pos = {c:[] for c in range(num_classes)}
    for nb_ob in range(crop_boxes.shape[0]):
        nb_img = ob_img_index[nb_ob]
        c      = ob_class_index[nb_ob]
        extra = 5
        cx,cy,r = ob_crop_boxes[nb_ob,:].type(torch.int32)
        r = r + extra
        xmin = torch.clamp(cx-r,0)
        ymin = torch.clamp(cy-r,0)
        pred = ypred[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]
        msk  = mask[nb_img,c,ymin:cy+r+1,xmin:cx+r+1]

        if center_mode == 'fixed':
            p_center = torch.tensor((cy-ymin, cx-xmin), dtype=dtype, device=device)
        else:
            pred_msk = pred * (msk >= 0.5)
            cands = (pred_msk == torch.max(pred_msk)).nonzero()
            rand = torch.randperm(cands.shape[0])
            p_center = cands[rand[0], :]
        pd_prob = pred[p_center[0]-center_dist:p_center[0]+center_dist+1, p_center[1]-center_dist:p_center[1]+center_dist+1]
        pd_prob = pd_prob.reshape(-1)

        pred_polar = polar_transform(pred, center=p_center, radius=r, **pt_params_used)
        msk_polar = polar_transform(msk, center=p_center, radius=r, **pt_params_used)
        pred_polar = pred_polar[:, polar_dist:] * polar_weights
        msk_polar = msk_polar[:, polar_dist:]
        
        msk_polar = msk_polar * (msk_polar > 0.5)
        flag = torch.sum(msk_polar,dim=1) > 0.5
        msk_polar = msk_polar[flag, :]
        pred_polar = pred_polar[flag, :]
        
        if method=='gm':
            w = pred_polar**gpower
            prob = torch.sum(w*msk_polar, dim=1)/torch.sum(msk_polar, dim=1)
            prob = prob**(1.0/gpower)
        elif method=='expsumr':
            w = torch.exp(gpower*pred_polar)
            prob = torch.sum(pred_polar*w*msk_polar,dim=1)/torch.sum(w*msk_polar,dim=1)
        elif method=='explogs':
            w = torch.exp(gpower*pred_polar)
            prob = torch.log(torch.sum(w*msk_polar,dim=1))/gpower - torch.log(torch.sum(msk_polar, dim=1))/gpower
        ypred_pos[c.item()].append(torch.cat([prob, pd_prob]))
    
    if mode=='focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        weight = 1 - mask # weights for negative samples
        weight = weight*(torch.rand(ypred.shape,dtype=ypred.dtype,device=ypred.device)<sampling_prob)
        losses = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            y_neg = ypred[:,c,:,:]
            y_neg = y_neg[(mask[:,c,:,:]<0.5)&(weight[:,c,:,:]>0.5)]
            bce_neg = -(1-alpha)*(y_neg**gamma)*torch.log(1-y_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                y_pos = torch.clamp(y_pos, epsilon, 1-epsilon)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    elif mode=='mil_focal':
        alpha = focal_params['alpha']
        gamma = focal_params['gamma']
        sampling_prob = focal_params['sampling_prob']
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -(1-alpha)*(ypred_neg**gamma)*torch.log(1-ypred_neg)
            if len(ypred_pos[c])>0:
                y_pos = torch.cat(ypred_pos[c], dim=0)
                bce_pos = -alpha*((1-y_pos)**gamma)*torch.log(y_pos)
                loss = (bce_neg.sum()+bce_pos.sum())/len(bce_pos)
            else:
                loss = bce_neg.sum()
            losses[c] = loss
    else:
        if method=='gm':
            ypred_g = ypred**gpower
        elif method=='expsumr': #alpha-softmax function
            ypred_g = torch.exp(gpower*ypred)
        elif method=='explogs': #alpha-quasimax function
            ypred_g = torch.exp(gpower*ypred)
        ## for negative class
        if method=='gm':
            v1 = (torch.sum(ypred_g*(1-mask), dim=2)/torch.sum(1-mask, dim=2))**(1.0/gpower)
            v2 = (torch.sum(ypred_g*(1-mask), dim=3)/torch.sum(1-mask, dim=3))**(1.0/gpower)
        elif method=='expsumr':
            v1 = torch.sum(ypred*ypred_g*(1-mask), dim=2)/torch.sum(ypred_g*(1-mask), dim=2)
            v2 = torch.sum(ypred*ypred_g*(1-mask), dim=3)/torch.sum(ypred_g*(1-mask), dim=3)
        elif method=='explogs':
            v1 = torch.log(torch.sum(ypred_g*(1-mask), dim=2))/gpower - torch.log(torch.sum(1-mask, dim=2))/gpower
            v2 = torch.log(torch.sum(ypred_g*(1-mask), dim=3))/gpower - torch.log(torch.sum(1-mask, dim=3))/gpower
        ypred_neg = torch.cat([v1,v2], dim=-1).permute(1,0,2)
        ypred_neg = torch.reshape(ypred_neg, (ypred_neg.shape[0],-1))

        losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=ypred.device)
        for c in range(num_classes):
            bce_neg = -torch.log(1-ypred_neg[c])
            if len(ypred_pos[c])>0:
                pred = torch.clamp(torch.cat(ypred_pos[c], dim=0), epsilon, 1-epsilon)
                bce_pos = -torch.log(pred)
                if mode=='all':
                    loss = (bce_pos.sum()+bce_neg.sum())/(len(bce_pos)+len(bce_neg))
                elif mode=='balance':
                    loss = (bce_pos.mean()+bce_neg.mean())/2
            else:
                loss = bce_neg.mean()
            losses[c] = loss

    return losses


def mil_pairwise_loss(ypred, mask, softmax=True, exp_coef=-1):
    """ Compute the pair-wise loss.

        As defined in Weakly Supervised Instance Segmentation using the Bounding Box Tightness Prior

    Args
        ypred: Tensor of predicted data from the network with shape (B, C, W, H).
        mask:  Tensor of mask with shape (B, C, W, H), bounding box regions with value 1 and 0 otherwise.
    Returns
        pair-wise loss for each category (C,)
    """
    device = ypred.device
    center_weight = torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    # TODO: modified this as one conv with 8 channels for efficiency
    pairwise_weights_list = [
            torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 0., 0.]]),  
            torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 1., 0.]]),  
            torch.tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]),  
            torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]])]
    ## pairwise loss for each col/row MIL
    num_classes = ypred.shape[1]
    if softmax:
        num_classes = num_classes - 1
    losses    = torch.zeros((num_classes,), dtype=ypred.dtype, device=device)
    for c in range(num_classes):
        pairwise_loss = []
        for w in pairwise_weights_list:
            weights = center_weight - w
            weights = weights.view(1, 1, 3, 3).to(device)
            aff_map = F.conv2d(ypred[:,c,:,:].unsqueeze(1), weights, padding=1)
            cur_loss = aff_map**2
            if exp_coef>0:
                cur_loss = torch.exp(exp_coef*cur_loss)-1
            cur_loss = torch.sum(cur_loss*mask[:,c,:,:].unsqueeze(1))/(torch.sum(mask[:,c,:,:]+1e-6))
            pairwise_loss.append(cur_loss)
        losses[c] = torch.mean(torch.stack(pairwise_loss))
    return losses

def size_constraint_loss(ypred, ytrue, is_true: bool = True):
    size_pred = torch.sum(ypred, dim=(0,2,3))
    size_true = torch.sum(ytrue, dim=(0,2,3))  
    losses = (size_pred/size_true-1)**2
    if is_true==False: ## weakly supervised size using bounding box, and ytrue is mask
        flag = size_pred<size_true
        losses[flag] = 0
    # print('-------',size_pred, size_true, losses)
    return losses


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious