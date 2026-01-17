import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from experiment_utils import get_transforms

# Variety Map
VARIETY_MAP = {
    'alig': 0, 'bessra': 1, 'kenta': 2, 'kintichi': 3,
    'Deglet Nour dryer': 4, 'Deglet Nour oily': 4, 'Deglet Nour oily treated': 4,
    'Deglet Nour semi-dryer': 4, 'Deglet Nour semi-dryer treated': 4,
    'Deglet Nour semi-oily': 4, 'Deglet Nour semi-oily treated': 4
}

# Maturity Map: 0:Dry, 1:Semi-Dry, 2:Oily, 3:Semi-Oily, 4:NA
MATURITY_MAP = {
    'alig': 4, 'bessra': 4, 'kenta': 4, 'kintichi': 4,
    'Deglet Nour dryer': 0, 
    'Deglet Nour semi-dryer': 1, 'Deglet Nour semi-dryer treated': 1,
    'Deglet Nour oily': 2, 'Deglet Nour oily treated': 2,
    'Deglet Nour semi-oily': 3, 'Deglet Nour semi-oily treated': 3
}

# Treatment Map: 0:Untreated, 1:Treated
TREATMENT_MAP = {
    'alig': 0, 'bessra': 0, 'kenta': 0, 'kintichi': 0,
    'Deglet Nour dryer': 0, 'Deglet Nour oily': 0, 'Deglet Nour oily treated': 1,
    'Deglet Nour semi-dryer': 0, 'Deglet Nour semi-dryer treated': 1,
    'Deglet Nour semi-oily': 0, 'Deglet Nour semi-oily treated': 1
}

class LIMTICMultiTaskDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.dataset = datasets.ImageFolder(self.split_dir)
        self.classes = self.dataset.classes
        self.samples = self.dataset.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target_idx = self.samples[idx]
        class_name = self.classes[target_idx]
        
        sample = self.dataset.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            
        variety = VARIETY_MAP[class_name]
        maturity = MATURITY_MAP[class_name]
        treatment = TREATMENT_MAP[class_name]
        
        return sample, (variety, maturity, treatment)

def get_multitask_dataloaders(root_dir, batch_size=32):
    train_transform, val_transform = get_transforms(augment=True)
    train_dataset = LIMTICMultiTaskDataset(root_dir, 'Train', transform=train_transform)
    val_dataset = LIMTICMultiTaskDataset(root_dir, 'Validation', transform=val_transform)
    test_dataset = LIMTICMultiTaskDataset(root_dir, 'Test', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
