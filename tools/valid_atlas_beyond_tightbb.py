import os, sys
sys.path.insert(0, os.getcwd())
import warnings 
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torchvision_wj.datasets.samplers import PatientSampler
from torchvision_wj.models.segwithbox.unetwithbox import UNetWithBox
from torchvision_wj.models.segwithbox.default_unet_net import *
from torchvision_wj import pd_utils 
from torchvision_wj.utils.losses import *
from torchvision_wj.utils import config, utils
from torchvision_wj.utils.atlas_utils import get_atlas
import torchvision_wj.utils.transforms as T


@torch.no_grad()
def evaluate(epoch, model, data_loader, image_names, device, threshold, save_detection=None, smooth=1e-10):
    file_2d = os.path.join(save_detection,'dice_2d.xlsx')
    file_3d = os.path.join(save_detection,'dice_3d.xlsx')
    torch.set_num_threads(1)
    model.eval()
    
    nn = 0
    dice_2d, dice_3d = {k:[] for k in range(len(threshold))}, {k:[] for k in range(len(threshold))}
    for images, targets in data_loader:
        nn = nn + 1
        # print("{}/{}".format(nn,len(data_loader))) 

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        gt = torch.stack([t["masks"] for t in targets], dim=0)
        gt = gt.bool()
        _, outputs = model(images, targets)
        
        for out in outputs:
            for n_th,th in enumerate(threshold):
                pred = out>th
                pred = pred[:gt.shape[0],:gt.shape[1],:gt.shape[2],:gt.shape[3]]
                intersect = pred&gt
                v_dice_2d = (2*torch.sum(intersect,dim=(1,2,3))+smooth)/(torch.sum(pred,dim=(1,2,3))+torch.sum(gt,dim=(1,2,3))+smooth)
                v_dice_3d = (2*torch.sum(intersect)+smooth)/(torch.sum(pred)+torch.sum(gt)+smooth)
                dice_2d[n_th].append(v_dice_2d.cpu().numpy())
                dice_3d[n_th].append(v_dice_3d.cpu().numpy())

    dice_2d = [np.hstack(dice_2d[key]) for key in dice_2d.keys()]
    dice_3d = [np.hstack(dice_3d[key]) for key in dice_3d.keys()]
    dice_2d = np.vstack(dice_2d).T
    dice_3d = np.vstack(dice_3d).T
    
    dice_2d = pd.DataFrame(data=dice_2d, columns=threshold)
    dice_3d = pd.DataFrame(data=dice_3d, columns=threshold)
    
    pd_utils.append_df_to_excel(file_2d, dice_2d, sheet_name=str(epoch), index=False)
    pd_utils.append_df_to_excel(file_3d, dice_3d, sheet_name=str(epoch), index=False)

    mean_2d = np.mean(dice_2d, axis=0)
    std_2d = np.std(dice_2d, axis=0)
    loc2 = np.argmax(mean_2d)
    mean_3d = np.mean(dice_3d, axis=0)
    std_3d = np.std(dice_3d, axis=0)
    loc3 = np.argmax(mean_3d)
    print('2d mean: {}({})'.format(mean_2d.iloc[loc2],std_2d.iloc[loc2]))
    print('3d mean: {}({})'.format(mean_3d.iloc[loc3],std_3d.iloc[loc3]))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_exp', default=62, type=int,
                        help='the index of experiments')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print(args)

    def get_polar_name(approx_method, osh, alpha, weight_min):
        if approx_method == 'softmax':
            if weight_min == 0.5:
                folder = f'enet_polarw_approx_focal_{osh[0]}_{osh[1]}_expsumr={alpha}_unary_pair'
            else:
                folder = f'enet_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_expsumr={alpha}_unary_pair'
        else:
            if weight_min == 0.5:
                folder = f'enet_polarw_approx_focal_{osh[0]}_{osh[1]}_explogs={alpha}_unary_pair'
            else:
                folder = f'enet_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_explogs={alpha}_unary_pair'
        return folder
    
    def get_polar_assisting_name(approx_method, osh, alpha, weight_min, angle):
        if approx_method == 'softmax':
            if weight_min == 0.5:
                folder = f'enet_parallel_polarw_approx_focal_{osh[0]}_{osh[1]}_expsumr={alpha}_unary_pair'
            else:
                folder = f'enet_parallel_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_expsumr={alpha}_unary_pair'
        else:
            if weight_min == 0.5:
                folder = f'enet_parallel_polarw_approx_focal_{osh[0]}_{osh[1]}_explogs={alpha}_unary_pair'
            else:
                folder = f'enet_parallel_polarw_{weight_min}_approx_focal_{osh[0]}_{osh[1]}_explogs={alpha}_unary_pair'
        if angle!=30:
            folder = f'{folder}_angle={angle}'
        return folder

    def get_polar_names(osh, weight_min, approx_methods=['softmax', 'quasimax']):
        folders = []
        for approx_method in approx_methods:
            for alpha in [0.5, 1, 2]:
                folder = get_polar_name(approx_method, osh, alpha, weight_min)
                folders.append(folder)
        return folders

    def get_polar_assisting_names(osh, weight_min, approx_methods=['softmax', 'quasimax'], angle=30):
        folders = []
        for approx_method in approx_methods:
            for alpha in [0.5, 1, 2]:
                folder = get_polar_assisting_name(approx_method, osh, alpha, weight_min, angle)
                folders.append(folder)
        return folders

    n_exp = args.n_exp
    dir_save_root = os.path.join('results','atlas')
    threshold = [0.001,0.005,0.01]+list(np.arange(0.05,0.9,0.05))
    if n_exp==0:
        experiment_names = ['enet_all_unary_pair']
    elif n_exp==1:
        angle = (-40,41,20)
        angle = (-40,41,10)
        angle = (-60,61,30)
        experiment_names = [
            f'enet_parallel_approx_focal_{-angle[0]}_{angle[-1]}_expsumr=4_unary_pair',
            f'enet_parallel_approx_focal_{-angle[0]}_{angle[-1]}_explogs=4_unary_pair',
            f'enet_parallel_approx_focal_{-angle[0]}_{angle[-1]}_expsumr=6_unary_pair',
            f'enet_parallel_approx_focal_{-angle[0]}_{angle[-1]}_explogs=6_unary_pair',
            f'enet_parallel_approx_focal_{-angle[0]}_{angle[-1]}_expsumr=8_unary_pair',
            f'enet_parallel_approx_focal_{-angle[0]}_{angle[-1]}_explogs=8_unary_pair',
        ] 
    elif n_exp==2:
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
        experiment_names = get_polar_names(osh, 0.5)
    elif n_exp==3:
        weight_min = 0.8
        # margin = 0
        osh1, osh2 = [90, 40], [60, 40]
        experiment_names = get_polar_names(osh1, weight_min, ['softmax']) + \
            get_polar_names(osh2, weight_min, ['quasimax'])
        # margin = 5
        osh1, osh2 = [60, 20], [120, 40]
        osh1, osh2 = [90, 20], [60, 10]
        experiment_names = get_polar_names(osh1, weight_min, ['softmax']) + \
            get_polar_names(osh2, weight_min, ['quasimax'])
        # margin = 10
        osh1, osh2 = [120, 30], [120, 20]
        experiment_names = get_polar_names(osh1, weight_min, ['softmax']) + \
            get_polar_names(osh2, weight_min, ['quasimax'])
        # margin = 10, random = True
        osh1, osh2 = [120, 20], [120, 10]
        experiment_names = get_polar_names(osh1, weight_min, ['softmax']) + \
            get_polar_names(osh2, weight_min, ['quasimax'])
    ## polar transformation assisting mil (the proposed approach)
    elif n_exp==4:
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
        experiment_names = get_polar_assisting_names(osh, 0.5)
        # margin = 5
        angle_params = (-40,41,20)
        experiment_names = get_polar_assisting_names(osh, 0.5, angle=angle_params[-1])
        # margin = 10
        angle_params1, angle_params2 = (-40,41,20), (-40,41,10)
        experiment_names = get_polar_assisting_names(osh, 0.5, ['softmax'], angle=angle_params1[-1]) + \
            get_polar_assisting_names(osh, 0.5, ['quasimax'], angle=angle_params2[-1])
        # margin = 10, random = True
        angle_params = (-40,41,20)
        experiment_names = get_polar_assisting_names(osh, 0.5, ['softmax'], angle=angle_params[-1])
        experiment_names = ['enet_parallel_polarw_approx_focal_120_10_explogs=2_unary_pair_angle=20']
    elif n_exp==5:
        weight_min = 0.7
        # margin = 0
        osh1, osh2 = [60, 40], [90, 10]
        experiment_names = get_polar_assisting_names(osh1, weight_min, ['softmax']) #+ \
            # get_polar_assisting_names(osh2, weight_min, ['quasimax'])
        # margin = 5
        osh, angle_params = [60, 30], (-40,41,20)
        experiment_names = get_polar_assisting_names(osh, weight_min, angle=angle_params[-1])
        # margin = 10
        osh1, osh2 = [60, 30], [60, 40]
        angle_params1, angle_params2 = (-40,41,20), (-40,41,10)
        experiment_names = get_polar_assisting_names(osh1, weight_min, ['softmax'], angle=angle_params1[-1]) + \
            get_polar_assisting_names(osh2, weight_min, ['quasimax'], angle=angle_params2[-1])
        # margin = 10, random = True
        osh1, osh2 = [60, 40], [60, 30]
        angle_params = (-40,41,20)
        experiment_names = get_polar_assisting_names(osh1, weight_min, ['softmax'], angle=angle_params[-1]) + \
            get_polar_assisting_names(osh2, weight_min, ['quasimax'], angle=angle_params[-1])

    for experiment_name in experiment_names:
        margin = 10
        random = True
        if margin > 0:
            experiment_name = f"{experiment_name}_margin={margin}"
        if random:
            experiment_name = f"{experiment_name}_random"
        print(experiment_name)
        output_dir = os.path.join(dir_save_root, experiment_name)
        _C = config.read_config_file(os.path.join(output_dir, 'config.yaml'))
        assert _C['save_params']['experiment_name']==experiment_name, "experiment_name is not right"
        cfg = {'data_params': {'workers': 0}}
        _C = config.config_updates(_C, cfg)

        train_params       = _C['train_params']
        data_params        = _C['data_params']
        net_params         = _C['net_params']
        anchor_params      = _C['anchor_params']
        gt_anchor_params   = _C['gt_anchor_params']
        dataset_params     = _C['dataset']
        nms_params         = _C['nms_params']
        save_params        = _C['save_params']
        print('workers = %d' % data_params['workers'])

        device = torch.device(_C['device'])

        def get_transform():
            transforms = []
            transforms.append(T.ToTensor())
            transforms.append(T.Normalizer(mode=data_params['normalizer_mode']))
            return T.Compose(transforms)

        # Data loading code
        print("Loading data")
        dataset_test = get_atlas(root=dataset_params['root_path'], 
                                    image_folder=dataset_params['valid_path'][0], 
                                    gt_folder=dataset_params['valid_path'][1], 
                                    transforms=get_transform(),
                                    transform_generator=None, visual_effect_generator=None)
        image_names = dataset_test.image_names
        
        print("Creating data loaders")
        test_patient_sampler = PatientSampler(dataset_test, dataset_params['grp_regex'], shuffle=False)

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
        
        file_2d = os.path.join(output_dir,'dice_2d.xlsx')
        file_3d = os.path.join(output_dir,'dice_3d.xlsx')
        # if os.path.exists(file_2d):
        #     os.remove(file_2d)
        # if os.path.exists(file_3d):
        #     os.remove(file_3d)
        for epoch in range(40):
            model_file = 'model_{:02d}'.format(epoch)
            print('loading model {}.pth'.format(model_file))
            checkpoint = torch.load(os.path.join(output_dir, model_file+'.pth'), map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
        
            print('start evaluating {} ...'.format(epoch))
            # suffix = '_nms_score={:.2f}_iou={:.2f}'.format(nms_params['nms_score_threshold'],nms_params['nms_iou_threshold'])
                                        
            # save_detection=os.path.join(output_dir, model_file)
            # if save_detection is not None:
            #     if not os.path.exists(save_detection):
            #         os.makedirs(save_detection, exist_ok=True)
            model.training = False
            evaluate(epoch, model, data_loader_test, image_names=image_names, device=device, threshold=threshold, save_detection=output_dir)
            
        
        dice_2d_all = pd.read_excel(file_2d, sheet_name=None)
        dice_3d_all = pd.read_excel(file_3d, sheet_name=None)
