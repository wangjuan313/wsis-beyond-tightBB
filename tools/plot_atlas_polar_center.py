import os, sys
sys.path.insert(0, os.getcwd())
import cv2
import imageio
import warnings 
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torchvision_wj.datasets.samplers import PatientSampler
from torchvision_wj.models.segwithbox.unetwithbox import UNetWithBox
from torchvision_wj.models.segwithbox.default_unet_net import *
from torchvision_wj.utils.losses import *
from torchvision_wj.utils import config, utils
from torchvision_wj.utils.atlas_utils import get_atlas
import torchvision_wj.utils.transforms as T


@torch.no_grad()
def visualize(epoch, model, data_loader, patients_dict, device, save_detection=None):
    torch.set_num_threads(1)
    model.eval()
    patients = list(patients_dict.keys())
    nn = 0
    for images, targets in data_loader:
        name = patients[nn]
        # name = image_names[nn]
        nn = nn + 1
        # print("{}/{}".format(nn,len(data_loader))) 

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        gt = torch.stack([t["masks"] for t in targets], dim=0)
        gt = gt.bool()
        _, outputs = model(images, targets)
        pred = outputs[0]
        pred = pred[:, :, :gt.shape[2], :gt.shape[3]]
        index = gt.sum(dim=(1,2,3)).nonzero().view(-1)
        
        alpha = 0.4
        for loc in index:
            box_pd = gt[loc, 0] * pred[loc, 0]
            if epoch == 0:
                img = 255 * (images[loc] - images[loc].min())/(images[loc].max()-images[loc].min())
                img = img[0].cpu().numpy().astype(np.uint8)
                mask = (gt[loc, 0]*255).cpu().numpy().astype(np.uint8)
                gt_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                gt_mask[:,:,0] = mask
                gt_img = [img, img, img]
                gt_img = np.stack(gt_img, axis=2)
                gt_img = cv2.addWeighted(gt_mask, alpha, gt_img, 1 - alpha, 0)
                gt_img_org = np.copy(gt_img)
            else:
                gt_img = cv2.imread(os.path.join(save_detection, f"{name}_{loc}.png"))

            center = torch.nonzero(box_pd==box_pd.max(), as_tuple=True)
            center = (center[1][0].item(), center[0][0].item())
            x, y = center[0], center[1]
            # gt_img = (gt[loc, 0].cpu().numpy()*255).astype(np.uint8)
            # gt_img = np.stack([gt_img, gt_img, gt_img], axis=2)
            # gt_img = cv2.circle(gt_img, center, radius=2, color=(0, 0, 255), thickness=1)
            gt_img = cv2.line(gt_img, (x-2,y), (x+2,y), color=(0, 0, 255), thickness=1)
            gt_img = cv2.line(gt_img, (x,y-2), (x,y+2), color=(0, 0, 255), thickness=1)
            cv2.imwrite(os.path.join(save_detection, f"{name}_{loc}.png"), gt_img)

            gt_img = gt_img[:, :, ::-1]
            gif_file = os.path.join(save_detection, f"{name}_{loc}.gif")
            if epoch == 0:
                gt_img_org = gt_img_org[:, :, ::-1]
                frames = [gt_img_org, gt_img]
            else:
                frames = imageio.mimread(gif_file)
                frames = [f[:,:,:3] for f in frames] + [gt_img]
            imageio.mimsave(gif_file, frames, 'GIF', duration=0.5)
    

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_exp', default=0, type=int,
                        help='the index of experiments')
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print(args)

    n_exp = args.n_exp
    dir_save_root = os.path.join('results','atlas')
    if n_exp==1:
        experiment_name = 'enet_parallel_polarw_approx_focal_60_40_expsumr=0.5_unary_pair'
    elif n_exp==2:
        experiment_name = 'enet_parallel_polarw_approx_focal_90_10_explogs=1_unary_pair'
    elif n_exp==3:
        experiment_name = 'enet_parallel_polarw_0.3_approx_focal_60_40_expsumr=2_unary_pair_angle=20_margin=10_random'
    elif n_exp==4:
        experiment_name = 'enet_parallel_polarw_approx_focal_60_30_explogs=2_unary_pair_angle=20_margin=10_random'

    print(experiment_name)
    output_dir = os.path.join(dir_save_root, experiment_name)
    _C = config.read_config_file(os.path.join(output_dir, 'config.yaml'))
    assert _C['save_params']['experiment_name']==experiment_name, "experiment_name is not right"
    cfg = {'data_params': {'workers': 0}}
    _C = config.config_updates(_C, cfg)

    # config_new = {'device': 'cpu'}
    # _C = config.config_updates(_C, config_new)

    train_params       = _C['train_params']
    data_params        = _C['data_params']
    net_params         = _C['net_params']
    anchor_params      = _C['anchor_params']
    gt_anchor_params   = _C['gt_anchor_params']
    dataset_params     = _C['dataset']
    nms_params         = _C['nms_params']
    save_params        = _C['save_params']

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
    patients_dict = test_patient_sampler.idx_map

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
    
    save_detection = os.path.join(output_dir, "centers_log")
    print(save_detection)
    os.makedirs(save_detection, exist_ok=True)
    for epoch in range(50):
        model_file = 'model_{:02d}'.format(epoch)
        print('loading model {}.pth'.format(model_file))
        checkpoint = torch.load(os.path.join(output_dir, model_file+'.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
        print('start evaluating {} ...'.format(epoch))
        model.training = False
        visualize(epoch, model, data_loader_test, patients_dict=patients_dict, device=device, save_detection=save_detection)
        



    
