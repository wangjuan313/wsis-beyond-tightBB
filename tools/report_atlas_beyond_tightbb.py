import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd


def performance_summary(folder, dice_2d_all, T=None):
    epochs = list(dice_2d_all.keys())
    threshold = list(dice_2d_all[epochs[0]].keys())
    assert len(epochs)==40, len(epochs)
    mean_2d_array, std_2d_array = [], []
    for key in dice_2d_all.keys():
        mean_2d_array.append(np.mean(np.asarray(dice_2d_all[key]),axis=0))
        std_2d_array.append(np.std(np.asarray(dice_2d_all[key]),axis=0))
    mean_2d_array = np.vstack(mean_2d_array)
    std_2d_array = np.vstack(std_2d_array)
    if T is None:
        max_mean = np.max(mean_2d_array)
        ind = np.where(mean_2d_array==np.max(mean_2d_array))
        max_std  = std_2d_array[ind][0]
        epoch = epochs[ind[0][0]]
        th = threshold[ind[1][0]]
        # print('theshold: ', len(threshold))
        # print(len(epochs))
        print('{:s}: {:.3f}({:.3f}), th={:0.3f}, epoch={:s}'.format(folder, max_mean, max_std, th, epoch))
    else:
        loc = np.where(np.abs(np.asarray(threshold)-T)<1e-4)[0][0]
        mean_v = mean_2d_array[:,loc]
        std_v = std_2d_array[:,loc]
        max_mean = np.max(mean_v)
        ind = np.where(mean_v==np.max(mean_v))
        max_std  = std_v[ind[0][0]]
        epoch = epochs[ind[0][0]]
        print('{:s}: {:.3f}({:.3f}), th={:0.3f}, epoch={:s}'.format(folder, max_mean, max_std, T, epoch))
        

if __name__ == "__main__":
    dir_save_root = os.path.join('results','atlas')
    def get_polar_name(approx_method, osh, alpha, weight_min, margin, random=False):
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
        if margin > 0:
            folder = f"{folder}_margin={margin}"
        if random:
            folder = f"{folder}_random"
        return folder
    
    def get_polar_assisting_name(approx_method, osh, alpha, weight_min, margin, angle=30, random=False):
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
            folder = f"{folder}_angle={angle}"
        if margin > 0:
            folder = f"{folder}_margin={margin}"
        if random:
            folder = f"{folder}_random"
        return folder


    def get_generalized_mil_names(angle, margin, random=False):
        folders = [
            f'enet_parallel_approx_focal_{angle[0]}_{angle[-1]}_expsumr=4_unary_pair_margin={margin}',
            f'enet_parallel_approx_focal_{angle[0]}_{angle[-1]}_explogs=4_unary_pair_margin={margin}',
            f'enet_parallel_approx_focal_{angle[0]}_{angle[-1]}_expsumr=6_unary_pair_margin={margin}',
            f'enet_parallel_approx_focal_{angle[0]}_{angle[-1]}_explogs=6_unary_pair_margin={margin}',
            f'enet_parallel_approx_focal_{angle[0]}_{angle[-1]}_expsumr=8_unary_pair_margin={margin}',
            f'enet_parallel_approx_focal_{angle[0]}_{angle[-1]}_explogs=8_unary_pair_margin={margin}',
        ] 
        if random:
            folders = [f"{f}_random" for f in folders]
        return folders

    def get_generalized_mil_name(mode, angle, sigma, margin, random=False):
        if mode == 'softmax':
            mode_name = 'expsumr'
        else:
            mode_name = 'explogs'
        folder = f'enet_parallel_approx_focal_{angle[0]}_{angle[-1]}_{mode_name}={sigma}_unary_pair_margin={margin}'
        if random:
            folder = f"{folder}_random"
        return folder

    margin = 0
    folders =  [
        ### baseline
        'enet_all_unary_pair',

        ### generalized mil
        'enet_parallel_approx_focal_60_30_expsumr=4_unary_pair',
        'enet_parallel_approx_focal_60_30_explogs=4_unary_pair',

        # ### polar mil
        get_polar_name('softmax', [90, 40], 0.5, 0.5, margin),
        get_polar_name('quasimax', [60, 40], 0.5, 0.5, margin),

        ### proposed
        get_polar_assisting_name('softmax', [60, 40], 0.5, 0.5, margin),
        get_polar_assisting_name('quasimax', [90, 10], 1, 0.5, margin),
    ]
    # margin = 5
    # angle = 20
    # folders =  [
    #     ### baseline
    #     f'enet_all_unary_pair_margin={margin}',

    #     ### generalized mil
    #     get_generalized_mil_name('softmax', [40,20], 4, margin),
    #     get_generalized_mil_name('quasimax', [40,20], 8, margin),

    #     ### polar mil
    #     get_polar_name('softmax', [60, 20], 0.5, 0.5, margin),
    #     get_polar_name('quasimax', [120, 40], 0.5, 0.5, margin),

    #     ### proposed
    #     get_polar_assisting_name('softmax', [60, 30], 0.5, 0.5, margin, angle=angle), 
    #     get_polar_assisting_name('quasimax', [60, 30], 1, 0.7, margin, angle=angle),
    # ]
    # margin = 10
    # angle1, angle2 = 20, 10
    # folders =  [
    #     ### baseline
    #     f'enet_all_unary_pair_margin={margin}',

    #     ### generalized mil
    #     get_generalized_mil_name('softmax', [40,20], 6, margin),
    #     get_generalized_mil_name('quasimax', [40,10], 4, margin),

    #     ### polar mil
    #     get_polar_name('softmax', [120, 30], 0.5, 0.5, margin),
    #     get_polar_name('quasimax', [120, 20], 1, 0.5, margin),
        
    #     ### proposed
    #     get_polar_assisting_name('softmax', [60, 30], 0.5, 0.7, margin, angle=angle1), 
    #     get_polar_assisting_name('quasimax', [60, 40], 0.5, 0.2, margin, angle=angle2),
    # ]
    # margin, random = 10, True
    # angle = 20
    # folders =  [
    #     ### baseline
    #     f'enet_all_unary_pair_margin={margin}_random',

    #     ### generalized mil
    #     get_generalized_mil_name('softmax', [40,20], 8, margin, random=random),
    #     get_generalized_mil_name('quasimax', [40,20], 4, margin, random=random),

    #     ### polar mil
    #     # approx_method, osh, alpha, weight_min, margin, random=False
    #     get_polar_name('softmax', [120, 20], 0.5, 0.8, margin, random=random),
    #     get_polar_name('quasimax', [120, 10], 0.5, 0.5, margin, random=random),
      
        
    #     ### proposed
    #     get_polar_assisting_name('softmax', [60, 40], 2, 0.3, margin, angle=angle, random=random),
    #     get_polar_assisting_name('quasimax', [60, 30], 2, 0.5, margin, angle=angle, random=random),
    # ]
    
    # print(folders)
    metrics = ['dice_2d', 'dice_3d']
    metrics = ['dice_3d']
    for metric in metrics:
        print('performance summary: {:s}'.format(metric).center(50,"#"))
        for k, folder in enumerate(folders):
            output_dir = os.path.join(dir_save_root, folder)
            file_name = os.path.join(output_dir,metric+'.xlsx')
            results = pd.read_excel(file_name, sheet_name=None)
            performance_summary(folder, results)