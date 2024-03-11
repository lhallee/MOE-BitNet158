import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from datasets import load_dataset


def calculate_mean_std(dataset):
    imgs = [np.array(img) for img in dataset['image']]
    mean = np.mean(imgs)
    std = np.std(imgs)
    return mean, std


class VisionDataset(TorchDataset):
    def __init__(self, dataset, mean=1, std=1, norm=False):
        self.dataset = dataset
        self.size = np.array(self.dataset[0]['image']).shape[0]
        self.imgs = [np.array(img).reshape(self.size, self.size) for img in self.dataset['image']]
        self.labels = [label for label in self.dataset['label']]
        
        if norm:  
            self.imgs = [(img - mean) / std for img in self.imgs]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'inputs_embeds': img, 'labels': label}


def vision_collator(batch):
    embeds = torch.stack([item['inputs_embeds'].squeeze(0) for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {'inputs_embeds': embeds, 'labels': labels}


def get_vision_dataset(data_path, norm=True):
    dataset = load_dataset(data_path)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    train_mean, train_std = calculate_mean_std(train_dataset)
    train_dataset = VisionDataset(train_dataset, train_mean, train_std, norm=norm)
    test_dataset = VisionDataset(test_dataset, train_mean, train_std, norm=norm)
    return train_dataset, test_dataset