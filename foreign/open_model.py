import os
import torch
import argparse
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main(args):
    
    ckpt_dict = {}
    for file_name in os.listdir(args.weight_dir):
        if file_name.endswith('.pth'):
            ckpt_num = file_name.split('.')[0].split('_')[-1]
            if ckpt_num == "final":
                continue
            ckpt_dict[int(ckpt_num)] = os.path.join(args.weight_dir, file_name)

    ckpt_dict = dict(sorted(ckpt_dict.items()))
    weight_list = []
    for ckpt_num, ckpt_path in ckpt_dict.items():
        ckpt = torch.load(ckpt_path)

        model_dict = ckpt["model"]
        for key_name in model_dict.keys():
            if key_name in args.param_name:
                weight_list.append(model_dict[key_name].cpu().flatten().numpy())
    weight_list = np.stack(weight_list)
    print(weight_list.shape)

    pca = PCA(n_components=2)
    weights_embedded = pca.fit_transform(weights_matrix)

    plt.figure(figsize=(8, 6))
    plt.plot(weights_embedded[:, 0], weights_embedded[:, 1], '-o')
    for i in range(num_steps):
        plt.text(weights_embedded[i, 0], weights_embedded[i, 1], f'Step {i}')
    plt.title('Weight Trajectory during Training')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='coco', choices=['voc', 'coco'])
    parser.add_argument('--src_path', type=str, default='', help='Path to the main checkpoint')
    parser.add_argument('--weight_dir', type=str, default='')
    parser.add_argument('--param_name', type=str, default='')
    # parser.add_argument('--save-dir', type=str, default='', required=True, help='Save directory')
    # parser.add_argument('--method', choices=['remove', 'randinit'], required=True,
    #                     help='remove = remove the final layer of the base detector. '
    #                          'randinit = randomly initialize novel weights.')
    args = parser.parse_args()

    
    main(args)