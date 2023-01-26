import datetime
import os, sys
sys.path.insert(0, os.getcwd())
import time
import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision_wj.datasets.transform import random_transform_generator
from torchvision_wj.datasets.image import random_visual_effect_generator
from torchvision_wj.datasets.samplers import PatientSampler
from torchvision_wj.models.segwithbox.unetwithbox import UNetWithBox
from torchvision_wj.models.segwithbox.default_unet_net import *
from torchvision_wj.utils.losses import *
from torchvision_wj.utils.early_stop import EarlyStopping
from torchvision_wj.utils import config, utils
from torchvision_wj.utils.promise_utils import get_promise
from torchvision_wj.utils.engine import train_one_epoch, validate_loss
import torchvision_wj.utils.transforms as T
from torchvision_wj._C_promise import _C


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_exp', default=0, type=int,
                        help='the index of experiments')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print(args)

    n_exp = args.n_exp
    cfg = {'train_params': {'patience': 8, 'lr': 1e-4, 'batch_size': 16}, 'data_params': {'workers': 4}}
    _C = config.config_updates(_C, cfg)
    # mil baseline
    if n_exp == 0:
        cfg1 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnarySigmoidLoss',{'mode':'all', 'focal_params':{'alpha':0.25, 'gamma':2.0, 'sampling_prob':1.0}},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': 'residual_all_unary_pair'}}
        _C_array = []
        for c in [cfg1]:
            _C_array.append(config.config_updates(_C, c))  
    ## parallel transformation based mil
    elif n_exp == 1:
        angle = (-40,41,20)
        # angle = (-40,41,10)
        # angle = (-60,61,30)
        cfg1 = {'net_params': {'softmax': False, 'losses': 
                             [('MILUnaryParallelApproxSigmoidLoss',
                               {'angle_params':angle, 'mode':'focal', 'method':'expsumr', 'gpower':4},1),
                               ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
               'save_params': {'experiment_name': f'residual_parallel_approx_focal_{-angle[0]}_{angle[-1]}_expsumr=4_unary_pair'}}
        cfg2 = {'net_params': {'softmax': False, 'losses': 
                             [('MILUnaryParallelApproxSigmoidLoss',
                              {'angle_params':angle, 'mode':'focal', 'method':'explogs', 'gpower':6},1),
                              ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
               'save_params': {'experiment_name': f'residual_parallel_approx_focal_{-angle[0]}_{angle[-1]}_explogs=6_unary_pair'}}
        cfg3 = {'net_params': {'softmax': False, 'losses': 
                             [('MILUnaryParallelApproxSigmoidLoss',
                               {'angle_params':angle, 'mode':'focal', 'method':'expsumr', 'gpower':6},1),
                               ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
               'save_params': {'experiment_name': f'residual_parallel_approx_focal_{-angle[0]}_{angle[-1]}_expsumr=6_unary_pair'}}
        cfg4 = {'net_params': {'softmax': False, 'losses': 
                             [('MILUnaryParallelApproxSigmoidLoss',
                              {'angle_params':angle, 'mode':'focal', 'method':'explogs', 'gpower':4},1),
                              ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
               'save_params': {'experiment_name': f'residual_parallel_approx_focal_{-angle[0]}_{angle[-1]}_explogs=4_unary_pair'}}
        cfg5 = {'net_params': {'softmax': False, 'losses': 
                             [('MILUnaryParallelApproxSigmoidLoss',
                               {'angle_params':angle, 'mode':'focal', 'method':'expsumr', 'gpower':8},1),
                               ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
               'save_params': {'experiment_name': f'residual_parallel_approx_focal_{-angle[0]}_{angle[-1]}_expsumr=8_unary_pair'}}
        cfg6 = {'net_params': {'softmax': False, 'losses': 
                             [('MILUnaryParallelApproxSigmoidLoss',
                              {'angle_params':angle, 'mode':'focal', 'method':'explogs', 'gpower':8},1),
                              ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
               'save_params': {'experiment_name': f'residual_parallel_approx_focal_{-angle[0]}_{angle[-1]}_explogs=8_unary_pair'}}
        _C_array = []
        for c in [cfg1, cfg2, cfg3, cfg4, cfg5, cfg6]:
            _C_array.append(config.config_updates(_C, c))        
    ## polar transformation based mil
    elif n_exp == 2:
        osh = [90, 40]
        osh = [90, 30]
        osh = [90, 20]
        osh = [90, 10]
        osh = [60, 40]
        osh = [60, 30]
        osh = [60, 20]
        osh = [60, 10]
        osh = [120, 40]
        osh = [120, 30]
        osh = [120, 20]
        osh = [120, 10]
        cfg1 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': 0.5, 'center_mode': 'estimated', 'method': 'expsumr', 'gpower': 0.5,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_polarw_approx_focal_{osh[0]}_{osh[1]}_expsumr=0.5_unary_pair'}}
        cfg2 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': 0.5, 'center_mode': 'estimated', 'method': 'expsumr', 'gpower': 1,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_polarw_approx_focal_{osh[0]}_{osh[1]}_expsumr=1_unary_pair'}}
        cfg3 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': 0.5, 'center_mode': 'estimated', 'method': 'expsumr', 'gpower': 2,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_polarw_approx_focal_{osh[0]}_{osh[1]}_expsumr=2_unary_pair'}}
        _C_array = []
        for c in [cfg1, cfg2, cfg3]:
            _C_array.append(config.config_updates(_C, c))  
    elif n_exp == 3:
        osh = [90, 40]
        osh = [90, 30]
        osh = [90, 20]
        osh = [90, 10]
        osh = [60, 40]
        osh = [60, 30]
        osh = [60, 20]
        osh = [60, 10]
        osh = [120, 40]
        osh = [120, 30]
        osh = [120, 20]
        osh = [120, 10]
        cfg1 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': 0.5, 'center_mode': 'estimated', 'method': 'explogs', 'gpower': 0.5,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_polarw_approx_focal_{osh[0]}_{osh[1]}_explogs=0.5_unary_pair'}}
        cfg2 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': 0.5, 'center_mode': 'estimated', 'method': 'explogs', 'gpower': 1,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_polarw_approx_focal_{osh[0]}_{osh[1]}_explogs=1_unary_pair'}}
        cfg3 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': 0.5, 'center_mode': 'estimated', 'method': 'explogs', 'gpower': 2,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_polarw_approx_focal_{osh[0]}_{osh[1]}_explogs=2_unary_pair'}}
        _C_array = []
        for c in [cfg1, cfg2, cfg3]:
            _C_array.append(config.config_updates(_C, c)) 
    elif n_exp == 4:
        # margin = 5
        osh = [90, 30]
        osh = [90, 10]
         # margin = 10
        # osh = [60, 20]
        # osh = [120, 30]
        # margin = 10, random = True
        osh = [120, 30]
        # margin = 0
        osh = [90, 30]
        weight_min = 0.3
        cfg1 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': weight_min, 'center_mode': 'estimated', 'method': 'expsumr', 'gpower': 0.5,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_expsumr=0.5_unary_pair'}}
        cfg2 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': weight_min, 'center_mode': 'estimated', 'method': 'expsumr', 'gpower': 1,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_expsumr=1_unary_pair'}}
        cfg3 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': weight_min, 'center_mode': 'estimated', 'method': 'expsumr', 'gpower': 2,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_expsumr=2_unary_pair'}}
        _C_array = []
        for c in [cfg3]:
            _C_array.append(config.config_updates(_C, c)) 
    elif n_exp == 5:
        # margin = 5
        osh = [90, 30]
        osh = [120, 20]
        # margin = 10
        # osh = [60, 20]
        # osh = [60, 10]
        # margin = 10, random = True
        osh = [120, 40]
        # margin = 0
        osh = [60, 30]
        osh = [120, 40]
        weight_min = 0.8
        cfg1 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': weight_min, 'center_mode': 'estimated', 'method': 'explogs', 'gpower': 0.5,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                  ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_explogs=0.5_unary_pair'}}
        cfg2 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': weight_min, 'center_mode': 'estimated', 'method': 'explogs', 'gpower': 1,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                  ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_explogs=1_unary_pair'}}
        cfg3 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': weight_min, 'center_mode': 'estimated', 'method': 'explogs', 'gpower': 2,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                  ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_explogs=2_unary_pair'}}
        _C_array = []
        for c in [cfg1, cfg2, cfg3]:
            _C_array.append(config.config_updates(_C, c)) 
    ## polar transformation assisting mil (the proposed approach)
    elif n_exp == 6:
        osh = [90, 40]
        osh = [90, 30]
        osh = [90, 20]
        osh = [90, 10]
        osh = [60, 40]
        osh = [60, 30]
        osh = [60, 20]
        osh = [60, 10]
        osh = [120, 40]
        osh = [120, 30]
        osh = [120, 20]
        osh = [120, 10]
        cfg1 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': 0.5, 'center_mode': 'estimated', 'method': 'expsumr', 'gpower': 0.5,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                 ('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'expsumr', 'gpower':4},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_parallel_polarw_approx_focal_{osh[0]}_{osh[1]}_expsumr=0.5_unary_pair'}}
        cfg2 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': 0.5, 'center_mode': 'estimated', 'method': 'expsumr', 'gpower': 1,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                  ('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'expsumr', 'gpower':4},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_parallel_polarw_approx_focal_{osh[0]}_{osh[1]}_expsumr=1_unary_pair'}}
        cfg3 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': 0.5, 'center_mode': 'estimated', 'method': 'expsumr', 'gpower': 2,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                  ('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'expsumr', 'gpower':4},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_parallel_polarw_approx_focal_{osh[0]}_{osh[1]}_expsumr=2_unary_pair'}}
        _C_array = []
        for c in [cfg1, cfg2, cfg3]:
            _C_array.append(config.config_updates(_C, c)) 
    elif n_exp == 7:
        osh = [90, 40]
        osh = [90, 30]
        osh = [90, 20]
        osh = [90, 10]
        osh = [60, 40]
        osh = [60, 30]
        osh = [60, 20]
        osh = [60, 10]
        osh = [120, 40]
        osh = [120, 30]
        osh = [120, 20]
        osh = [120, 10]
        cfg1 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': 0.5, 'center_mode': 'estimated', 'method': 'explogs', 'gpower': 0.5,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                  ('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'explogs', 'gpower':6},1),
                                  ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_parallel_polarw_approx_focal_{osh[0]}_{osh[1]}_explogs=0.5_unary_pair'}}
        cfg2 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': 0.5, 'center_mode': 'estimated', 'method': 'explogs', 'gpower': 1,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                  ('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'explogs', 'gpower':6},1),
                                  ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_parallel_polarw_approx_focal_{osh[0]}_{osh[1]}_explogs=1_unary_pair'}}
        cfg3 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': 0.5, 'center_mode': 'estimated', 'method': 'explogs', 'gpower': 2,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                  ('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'explogs', 'gpower':6},1),
                                  ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_parallel_polarw_approx_focal_{osh[0]}_{osh[1]}_explogs=2_unary_pair'}}
        _C_array = []
        for c in [cfg1, cfg2, cfg3]:
            _C_array.append(config.config_updates(_C, c)) 
    elif n_exp == 8:
        # margin = 5
        osh = [90, 30]
        osh = [120, 10]
        # margin = 10
        osh = [60, 30]
        # margin = 10, random = True
        osh = [120, 10]
        osh = [90, 40]
        # margin = 0
        osh = [90, 20]
        osh = [90, 10]
        weight_min = 0.8
        cfg1 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': weight_min, 'center_mode': 'estimated', 'method': 'expsumr', 'gpower': 0.5,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                 ('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'expsumr', 'gpower':4},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_parallel_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_expsumr=0.5_unary_pair'}}
        cfg2 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': weight_min, 'center_mode': 'estimated', 'method': 'expsumr', 'gpower': 1,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                  ('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'expsumr', 'gpower':4},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_parallel_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_expsumr=1_unary_pair'}}
        cfg3 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': weight_min, 'center_mode': 'estimated', 'method': 'expsumr', 'gpower': 2,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                  ('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'expsumr', 'gpower':4},1),
                                 ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_parallel_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_expsumr=2_unary_pair'}}
        _C_array = []
        for c in [cfg1, cfg2, cfg3]:
            _C_array.append(config.config_updates(_C, c)) 
    elif n_exp == 9:
        # margin = 5
        osh = [90, 30]
        osh = [120, 10]
        # margin = 10
        osh = [90, 10]
        # margin = 10, random = True
        osh = [120, 10]
         # margin = 0
        osh = [90, 20]
        osh = [120, 20]
        weight_min = 0.8
        cfg1 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': weight_min, 'center_mode': 'estimated', 'method': 'explogs', 'gpower': 0.5,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                  ('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'explogs', 'gpower':6},1),
                                  ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_parallel_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_explogs=0.5_unary_pair'}}
        cfg2 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': weight_min, 'center_mode': 'estimated', 'method': 'explogs', 'gpower': 1,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                  ('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'explogs', 'gpower':6},1),
                                  ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_parallel_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_explogs=1_unary_pair'}}
        cfg3 = {'net_params': {'softmax': False, 'losses': 
                                [('MILUnaryPolarApproxWeightedSigmoidLoss',
                                  {'mode': 'focal', 'weight_min': weight_min, 'center_mode': 'estimated', 'method': 'explogs', 'gpower': 2,
                                   'pt_params':{"output_shape": osh, "scaling": "linear"},
                                   'focal_params': {'alpha':0.25, 'gamma':2.0, 'sampling_prob': 1.0}
                                  }, 1),
                                  ('MILUnaryParallelApproxSigmoidLoss',
                                  {'angle_params':(-40,41,20), 'mode':'focal', 'method':'explogs', 'gpower':6},1),
                                  ('MILPairwiseLoss',{'softmax':False, 'exp_coef':-1},10)]},
                   'save_params': {'experiment_name': f'residual_parallel_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_explogs=2_unary_pair'}}
        _C_array = []
        for c in [cfg1, cfg2, cfg3]:
            _C_array.append(config.config_updates(_C, c)) 
    
    for _C in _C_array:
        random = False
        margin = 0
        base_experiment_name = _C['save_params']['experiment_name']
        if margin == 0:
            cfg = {}
        else:
            if random:
                cfg = {'save_params': {'experiment_name': base_experiment_name + \
                            '_margin='+str(margin)+'_random'}}
            else:
                cfg = {'save_params': {'experiment_name': base_experiment_name + \
                            '_margin='+str(margin)}}
        _C_used = config.config_updates(_C, cfg)
        assert _C_used['save_params']['experiment_name'] is not None, \
                    "experiment_name has to be set"

        train_params       = _C_used['train_params']
        data_params        = _C_used['data_params']
        net_params         = _C_used['net_params']
        dataset_params     = _C_used['dataset']
        save_params        = _C_used['save_params']
        data_visual_aug    = data_params['data_visual_aug']
        data_transform_aug = data_params['data_transform_aug']

        output_dir = os.path.join(save_params['dir_save'],save_params['experiment_name'])
        os.makedirs(output_dir, exist_ok=True)
        if not train_params['test_only']:
            config.save_config_file(os.path.join(output_dir,'config.yaml'), _C_used)
            print("saving files to {:s}".format(output_dir))
        
        device = torch.device(_C_used['device'])

        def get_transform():
            transforms = []
            transforms.append(T.ToTensor())
            transforms.append(T.Normalizer(mode=data_params['normalizer_mode']))
            return T.Compose(transforms)

        if data_transform_aug['aug']:
            transform_generator = random_transform_generator(
                min_rotation=data_transform_aug['min_rotation'],
                max_rotation=data_transform_aug['max_rotation'],
                min_translation=data_transform_aug['min_translation'],
                max_translation=data_transform_aug['max_translation'],
                min_shear=data_transform_aug['min_shear'],
                max_shear=data_transform_aug['max_shear'],
                min_scaling=data_transform_aug['min_scaling'],
                max_scaling=data_transform_aug['max_scaling'],
                flip_x_chance=data_transform_aug['flip_x_chance'],
                flip_y_chance=data_transform_aug['flip_y_chance'],
                )
        else:
            transform_generator = None
        if data_visual_aug['aug']:
            visual_effect_generator = random_visual_effect_generator(
                contrast_range=data_visual_aug['contrast_range'],
                brightness_range=data_visual_aug['brightness_range'],
                hue_range=data_visual_aug['hue_range'],
                saturation_range=data_visual_aug['saturation_range']
                )
        else:
            visual_effect_generator = None
        print('---data augmentation---')
        print('transform: ',transform_generator)
        print('visual: ',visual_effect_generator)

        # Data loading code
        print("Loading data")
        dataset      = get_promise(root=dataset_params['root_path'], 
                                   image_folder=dataset_params['train_path'][0], 
                                   gt_folder=dataset_params['train_path'][1], 
                                   margin=margin, random=random,
                                   transforms=get_transform(),
                                   transform_generator=transform_generator,
                                   visual_effect_generator=visual_effect_generator)
        dataset_test = get_promise(root=dataset_params['root_path'], 
                                   image_folder=dataset_params['valid_path'][0], 
                                   gt_folder=dataset_params['valid_path'][1], 
                                   margin=margin, random=random,
                                   transforms=get_transform(),
                                   transform_generator=None, visual_effect_generator=None)
        
        print("Creating data loaders")
        train_sampler = torch.utils.data.RandomSampler(dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, train_params['batch_size'], drop_last=True)

        # test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_patient_sampler = PatientSampler(dataset_test, dataset_params['grp_regex'], shuffle=False)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler, num_workers=data_params['workers'],
            collate_fn=utils.collate_fn, pin_memory=True)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            batch_sampler=test_patient_sampler, num_workers=data_params['workers'],
            collate_fn=utils.collate_fn, pin_memory=True)

        print("Creating model with parameters: {}".format(net_params))
        net = eval(net_params['model_name'])(net_params['input_dim'], net_params['seg_num_classes'],
                                             net_params['softmax'])
        losses, loss_weights = [], []
        for loss in net_params['losses']:
            losses.append(eval(loss[0])(**loss[1]))
            loss_weights.append(loss[2])
        model = UNetWithBox(net, losses, loss_weights, softmax=net_params['softmax'])
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        if train_params['optimizer']=='SGD':
            optimizer = torch.optim.SGD(params, lr=train_params['lr'], 
                                        momentum=train_params['momentum'], weight_decay=train_params['weight_decay'])
        elif train_params['optimizer']=='Adam':
            optimizer = torch.optim.Adam(params, lr=train_params['lr'], betas=train_params['betas'])

        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
        #                                                     milestones=train_params['lr_steps'], 
        #                                                     gamma=train_params['lr_gamma'])
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                                  factor=train_params['factor'], 
                                                                  patience=train_params['patience'])
        early_stop = EarlyStopping(patience=2*train_params['patience'])

        torch.autograd.set_detect_anomaly(True)
        model.training = True
        if train_params['test_only']:
            val_metric_logger = validate_loss(model, data_loader_test, device)
        else:
            print("Start training")
            start_time = time.time()
            summary = {'epoch':[]}
            for epoch in range(train_params['start_epoch'], train_params['epochs']):
                if args.distributed:
                    train_sampler.set_epoch(epoch)

                metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, 
                                train_params['clipnorm'], train_params['print_freq'])
                
                
                if output_dir:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'args': args,
                        'epoch': epoch},
                        os.path.join(output_dir, 'model_{:02d}.pth'.format(epoch)))

                # evaluate after every epoch
                val_metric_logger = validate_loss(model, data_loader_test, device)

                # collect the results and save
                summary['epoch'].append(epoch)
                for name, meter in metric_logger.meters.items():
                    if name=='lr':
                        v = meter.global_avg
                    else:
                        v = float(np.around(meter.global_avg,4))
                    if epoch==0:
                        summary[name] = [v]
                    else:
                        summary[name].append(v)
                for name, meter in val_metric_logger.meters.items():
                    v = float(np.around(meter.global_avg,4))
                    if epoch==0:
                        summary[name] = [v]
                    else:
                        summary[name].append(v)
                summary_save = pd.DataFrame(summary)
                summary_save.to_csv(os.path.join(output_dir,'summary.csv'), index=False)

                # update lr scheduler
                val_loss = val_metric_logger.meters['val_loss'].global_avg
                lr_scheduler.step(val_loss)

                # # early stop check
                # if early_stop.step(val_loss):
                #     print('Early stop at epoch = {}'.format(epoch))
                #     break

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))

            ## plot training and validation loss
            plt.figure()
            plt.plot(summary_save['epoch'],summary_save['loss'],'-ro', label='train')
            plt.plot(summary_save['epoch'],summary_save['val_loss'],'-g+', label='valid')
            plt.legend(loc=0)
            plt.savefig(os.path.join(output_dir,'loss.jpg'))
