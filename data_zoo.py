import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms


def get_mnist(norm=True):
    train_dataset = load_dataset('mnist', split='train')
    test_dataset = load_dataset('mnist', split='test')

    def calculate_mean_std(dataset):
        pixel_values = torch.tensor(np.array([np.array(item['image']).flatten() for item in dataset]),
                                    dtype=torch.float32)
        mean = pixel_values.mean()
        std = pixel_values.std()
        return mean, std
    
    def data_collator(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        return {'inputs_embeds': pixel_values, 'labels': labels}

    if norm:
        train_mean, train_std = calculate_mean_std(train_dataset)
        
        test_mean, test_std = calculate_mean_std(test_dataset)
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=train_mean, std=train_std)])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=test_mean, std=test_std)])
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset.set_transform(lambda x: {'pixel_values': train_transform(np.array(x['image']))})
    test_dataset.set_transform(lambda x: {'pixel_values': test_transform(np.array(x['image']))})
    
    return train_dataset, test_dataset, data_collator

