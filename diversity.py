import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import numpy as np
import os
import random
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from PIL import Image

class InstanceMeanHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.instance_mean = None

    def hook_fn(self, module, input, output):
        self.instance_mean = torch.mean(input[0], dim=[2, 3])

    def close(self):
        self.hook.remove()

class MLPHead(nn.Module):
    def __init__(self, dim_in, dim_feat, dim_h=None):
        super(MLPHead, self).__init__()
        if dim_h is None:
            dim_h = dim_in

        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.ReLU(inplace=True),
            nn.Linear(dim_h, dim_feat),
        )

    def forward(self, x):
        x = self.head(x)
        return F.normalize(x, dim=1, p=2)

class MultiTransform:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str(self.transform)

class ContrastLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(ContrastLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, return_logits=False):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        if return_logits:
            return loss, anchor_dot_contrast
        return loss

class MemoryBank:
    def __init__(self, size, dim):
        self.size = size
        self.dim = dim
        self.ptr = 0
        self.bank = torch.randn(size, dim)
        self.bank = F.normalize(self.bank, dim=1)

    @torch.no_grad()
    def update(self, features):
        batch_size = features.size(0)
        if self.ptr + batch_size > self.size:
            self.ptr = 0
        self.bank[self.ptr:self.ptr + batch_size] = features
        self.ptr += batch_size

    def get(self):
        return self.bank

def load_npy(file_path):
    data = np.load(file_path)
    if data.ndim != 1:
        raise ValueError(f"Unexpected data shape: {data.shape}")
    return data

def load_images_from_indices(indices, dataset):
    return [dataset[i][0] for i in indices]

def calculate_data_diversity(images, model, head, device):
    model.eval()
    head.eval()

    transform = MultiTransform([
        transforms.Compose([
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(),
        ]),
        transforms.Compose([
            transforms.RandomResizedCrop(size=32),
            transforms.RandomHorizontalFlip(),
        ])
    ])

    embeddings = []
    with torch.no_grad():
        for img in images:
            global_view, local_view = transform(img)
            global_view = global_view.unsqueeze(0).to(device)
            local_view = local_view.unsqueeze(0).to(device)

            global_embedding = model(global_view)
            local_embedding = model(local_view)

            global_feat = head(global_embedding)
            local_feat = head(local_embedding)

            embeddings.append(global_feat.cpu().numpy())
            embeddings.append(local_feat.cpu().numpy())

    embeddings = np.vstack(embeddings)
    cosine_sim = np.dot(embeddings, embeddings.T)
    diversity_score = 1 / cosine_sim.mean()
    return diversity_score

def main(bin_dir_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=transform)

    teacher_model = models.resnet18(weights='IMAGENET1K_V1')
    cmi_feature_dim = teacher_model.fc.in_features
    teacher_model.fc = nn.Identity()
    teacher_model = teacher_model.to(device)

    head_dim = 128
    head = MLPHead(dim_in=cmi_feature_dim, dim_feat=head_dim).to(device)

    diversity_scores = []

    for i in range(10):
        bin_file_path = os.path.join(bin_dir_path, f'select_indices_CIFAR10_exp{i}.npy')
        indices = load_npy(bin_file_path)
        images = load_images_from_indices(indices, cifar10_train)

        diversity_score = calculate_data_diversity(images, teacher_model, head, device)
        diversity_scores.append(diversity_score)

        print(f'Bin {i} Diversity Score: {diversity_score}')

    print(f'All Diversity Scores: {diversity_scores}')

if __name__ == "__main__":
    bin_dir_path = '../results/bin_cifar_010'
    main(bin_dir_path)
