import os
import os.path as osp
import numpy as np
import cv2
import pickle
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
from multiprocessing import Pool

def extract_single(opt, dataloader, exp):
    save_folder = opt['save_folder']
    grad_thres = opt["gradient_threshold"]

    all_img_grad = []

    pbar = tqdm(total=len(dataloader), desc='Processing images')

    pool = Pool(opt['n_thread'])
    img_grad_results = []

    for i, data in enumerate(dataloader):
        img, _ = data
        img = img.squeeze(0).numpy().transpose(1, 2, 0)  # Convert to HWC format
        img_grad_results.append(pool.apply_async(worker, args=(img, opt, grad_thres), callback=lambda _: pbar.update(1)))
    
    pool.close()
    pool.join()

    for result in img_grad_results:
        all_img_grad.extend(result.get())

    pbar.close()

    # Calculate the mean of over gradient mean
    overall_grad_mean = np.mean(all_img_grad)
    print(f'Exp {exp} overall gradient mean: {overall_grad_mean}')

    # save_folder
    with open(osp.join(save_folder, 'CIFAR10_grad_means_exp{}.pkl'.format(exp)), 'wb') as f:
        pickle.dump(all_img_grad, f)

    print('All subprocesses done.')

def worker(img, opt, grad_thres):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1)
    grad_x = abs(grad_x)
    grad_y = abs(grad_y)
    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    grad_mean = np.mean(grad)

    return [grad_mean]

def main():
    # Parameters
    opt = {
        'n_thread': 5,
        'compression_level': 3,
        'gradient_threshold': None,
        'save_folder': 'path_to_save_grad_means',
        'crop_sz': 32 
        # For low-resolution images (CIFAR-10 here), we do not need to crop the images, so L is 32.
        # For high-resolution images (like ImageNet-1K), we need to crop images to patches with 32×32.
    }

    if not osp.exists(opt['save_folder']):
        os.makedirs(opt['save_folder'])

    data_path = '~/data_cifar'  
    subset_path = '../results/bin_cifar_010'  # save_path
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)

    for exp in range(10):  
        subset_file = osp.join(subset_path, 'select_indices_CIFAR10_exp{}.npy'.format(exp))
        if not osp.exists(subset_file):
            print(f'Subset file for exp {exp} not found.')
            continue

        subset_indices = np.load(subset_file)
        subset = torch.utils.data.Subset(trainset, subset_indices)
        subset_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=opt['n_thread'])

        extract_single(opt, subset_loader, exp)

if __name__ == '__main__':
    main()