import numpy as np
import torch
import torch.nn as nn
from .losses_func import *


# ypred: predicted network output
# ytrue: true segmentation mask
# mask:  mask with bounding box region equal to 1
# gt_boxes: bounding boxes of objects 


class CrossEntropyLoss():
    def __init__(self, mode='all', epsilon=1e-6):
        self.mode = mode
        self.epsilon = epsilon

    def __call__(self, ypred, ytrue):
        ypred = torch.clamp(ypred, self.epsilon, 1-self.epsilon)
        loss_pos = -ytrue*torch.log(ypred)
        loss_neg  = -(1-ytrue)*torch.log(1-ypred)

        loss_pos = torch.sum(loss_pos, dim=(0,2,3))
        loss_neg = torch.sum(loss_neg, dim=(0,2,3))
        nb_pos = torch.sum(ytrue,dim=(0,2,3))
        nb_neg = torch.sum(1-ytrue,dim=(0,2,3))

        if self.mode=='all':
            loss  = (loss_pos+loss_neg)/(nb_pos+nb_neg)
        elif self.mode=='balance':
            loss  = (loss_pos/nb_pos+loss_neg/nb_neg)/2
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, cutoff=0.5, epsilon=1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cutoff = cutoff
        self.epsilon = epsilon
        
    def __call__(self, ypred, ytrue):
        loss = focal_loss(ytrue, ypred,
                          alpha=self.alpha, gamma=self.gamma, 
                          cutoff=self.cutoff, epsilon=self.epsilon)
        return loss


class SmoothL1Loss(nn.Module):
    def __init__(self, sigma=3.0, size_average=True):
        super(SmoothL1Loss, self).__init__()
        self.sigma = sigma
        self.size_average = size_average
        
    def __call__(self, ypred_reg, ytrue_reg, weight_reg=None):
        loss = smooth_l1_loss(ytrue_reg, ypred_reg, weight_reg, sigma=self.sigma, size_average=self.size_average)
        return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, flag_weight=0, sampling_prob=1.0, dilation_diameter=11, alpha=0.25, gamma=2.0, cutoff=0.5, epsilon=1e-6):
        super(SigmoidFocalLoss, self).__init__()
        self.flag_weight = flag_weight
        if self.flag_weight in [1,2]:
            self.sampling_prob = sampling_prob
        else:
            self.sampling_prob = 1.0
        if self.flag_weight==2:
            self.dilation_diameter = dilation_diameter
        else:
            self.dilation_diameter = 11

        self.alpha = alpha
        self.gamma = gamma
        self.cutoff = cutoff
        self.epsilon = epsilon
        
    def __call__(self, ypred, ytrue):
        loss = sigmoid_focal_loss(ytrue, ypred, 
                                  flag_weight=self.flag_weight, 
                                  dilation_diameter=self.dilation_diameter,
                                  sampling_prob=self.sampling_prob,
                                  alpha=self.alpha, gamma=self.gamma, 
                                  cutoff=self.cutoff, epsilon=self.epsilon)
        return loss


class MILUnarySigmoidLoss():
    def __init__(self, mode='all', focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, epsilon=1e-6):
        super(MILUnarySigmoidLoss, self).__init__()
        self.mode = mode
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, ypred, mask, gt_boxes):
        loss = mil_unary_sigmoid_loss(ypred, mask, gt_boxes, mode=self.mode,
                                      focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


class MILUnaryApproxSigmoidLoss():
    def __init__(self, mode='all', method='gm', gpower=4, 
                 focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, epsilon=1e-6):
        super(MILUnaryApproxSigmoidLoss, self).__init__()
        self.mode = mode
        self.method = method
        self.gpower = gpower
        self.focal_params = focal_params
        self.epsilon = epsilon 
        
    def __call__(self, ypred, mask, gt_boxes):
        loss = mil_unary_approx_sigmoid_loss(ypred, mask, gt_boxes, mode=self.mode,
                                             method=self.method, gpower=self.gpower, 
                                             focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


class MILMaskUnaryApproxSigmoidLoss():
    def __init__(self, mode='all', threshold=0.5, method='gm', gpower=4, 
                 focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, epsilon=1e-6):
        super(MILMaskUnaryApproxSigmoidLoss, self).__init__()
        self.mode = mode
        self.threshold = threshold
        self.method = method
        self.gpower = gpower
        self.focal_params = focal_params
        self.epsilon = epsilon 
        
    def __call__(self, ypred, mask, gt_boxes):
        pred_mask = ypred > self.threshold
        loss = mil_unary_approx_sigmoid_loss(ypred*pred_mask, mask, gt_boxes, mode=self.mode,
                                             method=self.method, gpower=self.gpower, 
                                             focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


class MILUnaryParallelSigmoidLoss():
    def __init__(self, mode='all', angle_params=(-45,46,5), 
                       focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
                       obj_size=0, epsilon=1e-6):
        super(MILUnaryParallelSigmoidLoss, self).__init__()
        self.mode           = mode
        self.angle_params   = angle_params
        self.focal_params   = focal_params
        self.obj_size       = obj_size
        self.epsilon        = epsilon
        
    def __call__(self, ypred, mask, crop_boxes):
        loss = mil_parallel_unary_sigmoid_loss(ypred, mask, crop_boxes, angle_params=self.angle_params,
                                               mode=self.mode, focal_params=self.focal_params, 
                                               obj_size=self.obj_size, epsilon=self.epsilon)
        return loss


class MILUnaryParallelApproxSigmoidLoss():
    def __init__(self, mode='all', angle_params=(-45,46,5), method='gm', gpower=4, 
                 focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
                 obj_size=0, epsilon=1e-6):
        super(MILUnaryParallelApproxSigmoidLoss, self).__init__()
        self.mode           = mode
        self.angle_params   = angle_params
        self.method         = method
        self.gpower         = gpower
        self.focal_params   = focal_params
        self.obj_size       = obj_size
        self.epsilon        = epsilon
        
    def __call__(self, ypred, mask, crop_boxes):
        loss = mil_parallel_approx_sigmoid_loss(ypred, mask, crop_boxes, angle_params=self.angle_params,
                                               mode=self.mode, focal_params=self.focal_params, 
                                               obj_size=self.obj_size, epsilon=self.epsilon)
        return loss


class MILUnaryPolarSoftmaxLoss():
    def __init__(self, mode='all', 
        pt_params={"valid_radius": 25, "extra": 5, "output_shape": [90, 30], "scaling": "linear"},
        focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
        super(MILUnaryPolarSoftmaxLoss, self).__init__()
        self.mode = mode
        self.pt_params = pt_params
        self.focal_params = focal_params
        self.epsilon = epsilon
        
    def __call__(self, ypred, gt_points, bkg_gt, weights=None):
        loss = mil_polar_unary_softmax_loss(ypred, gt_points, bkg_gt, weights=weights, 
                                        pt_params=self.pt_params, mode=self.mode,
                                        focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


class MILUnaryPolarApproxSoftmaxLoss():
    def __init__(self, mode='all', method='gm', gpower=4,
        pt_params={"valid_radius": 25, "extra": 5, "output_shape": [90, 30], "scaling": "linear"},
        focal_params={'alpha':0.25, 'gamma':2.0}, epsilon=1e-6):
        super(MILUnaryPolarApproxSoftmaxLoss, self).__init__()
        self.mode = mode
        self.pt_params = pt_params
        self.focal_params = focal_params
        self.method = method
        self.gpower = gpower
        self.epsilon = epsilon
        
    def __call__(self, ypred, gt_points, bkg_gt, weights=None):
        loss = mil_polar_approx_softmax_loss(ypred, gt_points, bkg_gt, weights=weights, 
                                        method=self.method, gpower=self.gpower, 
                                        pt_params=self.pt_params, mode=self.mode,
                                        focal_params=self.focal_params, epsilon=self.epsilon)
        return loss


class MILUnaryPolarApproxSigmoidLoss():
    def __init__(self, mode='all', center_mode='fixed', method='gm', gpower=4, 
        pt_params={"output_shape": [90, 30], "scaling": "linear"},
        focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
        smoothness_weight=-1, epsilon=1e-6):
        super(MILUnaryPolarApproxSigmoidLoss, self).__init__()
        self.mode = mode
        self.center_mode = center_mode
        self.pt_params = pt_params
        self.focal_params = focal_params
        self.method = method
        self.gpower = gpower
        self.smoothness_weight = smoothness_weight
        self.epsilon = epsilon
        
    def __call__(self, ypred, mask, crop_boxes):
        loss = mil_polar_approx_sigmoid_loss(ypred, mask, crop_boxes, mode=self.mode, 
                center_mode=self.center_mode, method=self.method, gpower=self.gpower, 
                focal_params=self.focal_params, pt_params=self.pt_params, 
                smoothness_weight=self.smoothness_weight, epsilon=self.epsilon)
        return loss


class MILUnaryPolarApproxPseudoLabelSigmoidLoss():
    def __init__(self, mode='all', center_mode='fixed', method='gm', gpower=4, 
        pt_params={"output_shape": [90, 30], "scaling": "linear"},
        focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, 
        smoothness_weight=-1, pseudolabel=True, pseudolabel_threshold=0.8, epsilon=1e-6):
        super(MILUnaryPolarApproxPseudoLabelSigmoidLoss, self).__init__()
        self.mode = mode
        self.center_mode = center_mode
        self.pt_params = pt_params
        self.focal_params = focal_params
        self.method = method
        self.gpower = gpower
        self.smoothness_weight = smoothness_weight
        self.pseudolabel = pseudolabel
        self.pseudolabel_threshold = pseudolabel_threshold
        self.epsilon = epsilon
        
    def __call__(self, ypred, mask, crop_boxes):
        loss = mil_polar_approx_pseudolabel_sigmoid_loss(ypred, mask, crop_boxes, mode=self.mode, 
                center_mode=self.center_mode, method=self.method, gpower=self.gpower, 
                focal_params=self.focal_params, pt_params=self.pt_params, 
                smoothness_weight=self.smoothness_weight, pseudolabel=self.pseudolabel,
                pseudolabel_threshold=self.pseudolabel_threshold, epsilon=self.epsilon)
        return loss


class MILUnaryPolarApproxWeightedSigmoidLoss():
    def __init__(self, mode='all', weight_min=0.5, center_mode='fixed', method='gm', gpower=4, 
        pt_params={"output_shape": [90, 30], "scaling": "linear"},
        focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0},
        smoothness_weight=-1, epsilon=1e-6):
        super(MILUnaryPolarApproxWeightedSigmoidLoss, self).__init__()
        self.mode = mode
        self.weight_min = weight_min
        self.center_mode = center_mode
        self.pt_params = pt_params
        self.focal_params = focal_params
        self.method = method
        self.gpower = gpower
        self.smoothness_weight = smoothness_weight
        self.epsilon = epsilon
        
    def __call__(self, ypred, mask, crop_boxes):
        loss = mil_polar_approx_weighted_sigmoid_loss(ypred, mask, crop_boxes, mode=self.mode, 
                weight_min=self.weight_min, center_mode=self.center_mode, method=self.method, gpower=self.gpower, 
                focal_params=self.focal_params, pt_params=self.pt_params,
                smoothness_weight=self.smoothness_weight, epsilon=self.epsilon)
        return loss


class MILUnaryPolarApproxWeighted2SigmoidLoss():
    def __init__(self, mode='all', weight_min=0.5, center_mode='fixed', method='gm', gpower=4, 
        pt_params={"output_shape": [90, 30], "scaling": "linear"},
        focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, epsilon=1e-6):
        super(MILUnaryPolarApproxWeighted2SigmoidLoss, self).__init__()
        self.mode = mode
        self.weight_min = weight_min
        self.center_mode = center_mode
        self.pt_params = pt_params
        self.focal_params = focal_params
        self.method = method
        self.gpower = gpower
        self.epsilon = epsilon
        
    def __call__(self, ypred, mask, crop_boxes):
        loss = mil_polar_approx_weighted2_sigmoid_loss(ypred, mask, crop_boxes, mode=self.mode, 
                weight_min=self.weight_min, center_mode=self.center_mode, method=self.method, gpower=self.gpower, 
                focal_params=self.focal_params, pt_params=self.pt_params, epsilon=self.epsilon)
        return loss


class MILUnaryPolarBiDirectApproxWeightedSigmoidLoss():
    def __init__(self, mode='all', weight_min=0.5, center_mode='fixed', 
        method='gm', gpower=4, weight_min_neg=None, gpower_neg=None,
        pt_params={"output_shape": [90, 30], "scaling": "linear"},
        focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0},
        smoothness_weight=-1, epsilon=1e-6):
        super(MILUnaryPolarBiDirectApproxWeightedSigmoidLoss, self).__init__()
        self.mode = mode
        self.weight_min = weight_min
        self.center_mode = center_mode
        self.pt_params = pt_params
        self.focal_params = focal_params
        self.method = method
        self.gpower = gpower
        self.weight_min_neg = weight_min_neg
        self.gpower_neg = gpower_neg
        self.smoothness_weight = smoothness_weight
        self.epsilon = epsilon
        
    def __call__(self, ypred, mask, crop_boxes):
        loss = mil_polar_bidirect_approx_weighted_sigmoid_loss(ypred, mask, crop_boxes, mode=self.mode, 
                weight_min=self.weight_min, center_mode=self.center_mode, method=self.method, gpower=self.gpower, 
                weight_min_neg=self.weight_min_neg, gpower_neg=self.gpower_neg,
                focal_params=self.focal_params, pt_params=self.pt_params,
                smoothness_weight=self.smoothness_weight, epsilon=self.epsilon)
        return loss


class MILUnaryPolarApproxDistWeightedSigmoidLoss():
    def __init__(self, mode='all', weight_min=0.5, center_mode='fixed', method='gm', gpower=4, 
        polar_dist=1, center_dist=1, 
        pt_params={"output_shape": [90, 30], "scaling": "linear"},
        focal_params={'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}, epsilon=1e-6):
        super(MILUnaryPolarApproxDistWeightedSigmoidLoss, self).__init__()
        self.mode = mode
        self.weight_min = weight_min
        self.center_mode = center_mode
        self.polar_dist = polar_dist
        self.center_dist = center_dist
        self.pt_params = pt_params
        self.focal_params = focal_params
        self.method = method
        self.gpower = gpower
        self.epsilon = epsilon
        
    def __call__(self, ypred, mask, crop_boxes):
        loss = mil_polar_approx_dist_weighted_sigmoid_loss(ypred, mask, crop_boxes, mode=self.mode, 
                weight_min=self.weight_min, center_mode=self.center_mode, method=self.method,
                gpower=self.gpower, polar_dist=self.polar_dist, center_dist=self.center_dist,
                focal_params=self.focal_params, pt_params=self.pt_params, epsilon=self.epsilon)
        return loss


class MILPairwiseLoss():
    def __init__(self, softmax=True, exp_coef=-1):
        super(MILPairwiseLoss, self).__init__()
        self.softmax = softmax
        self.exp_coef = exp_coef
        
    def __call__(self, ypred, mask):
        loss = mil_pairwise_loss(ypred, mask, softmax=self.softmax, exp_coef=self.exp_coef)
        return loss


class DiceLoss():
    def __init__(self, smooth=1e-10):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def __call__(self, pred_mask, mask):
        dice = (2*torch.sum(pred_mask*mask,axis=(0,2,3))+self.smooth)/ \
                            (torch.sum(pred_mask,axis=(0,2,3))+\
                             np.sum(torch, axis=(0,2,3))+self.smooth)
        loss = 1 - dice
        return loss


class IouLoss():
    def __init__(self, mode=True, smooth=1e-10):
        super(IouLoss, self).__init__()
        assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
        self.mode = mode
        self.smooth = smooth

    def __call__(self, ypred_decode_reg, ytrue_decode_reg):
        loss = bbox_overlaps(ypred_decode_reg, ytrue_decode_reg, self.mode, 
                             True, self.smooth)
        loss = 1 - loss.mean()
        return loss
