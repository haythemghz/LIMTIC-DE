import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from experiment_utils import get_transforms
import re

class LIMTICMultiViewDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.dataset = datasets.ImageFolder(self.split_dir)
        self.classes = self.dataset.classes
        
        # Group samples by fruit ID
        # Filename example: alig_100_jpg, alig_101_jpg...
        # We assume 6 consecutive images belong to the same fruit.
        self.grouped_samples = self._group_by_fruit(self.dataset.samples)

    def _group_by_fruit(self, samples):
        # Dictionary mapping (class, group_id) -> list of 6 paths
        groups = {}
        for path, label in samples:
            filename = os.path.basename(path)
            # Find the fruit index (idx // 6)
            match = re.search(r'_(\d+)_jpg', filename)
            if match:
                idx = int(match.group(1))
                fruit_id = idx // 6
                key = (label, fruit_id)
                if key not in groups:
                    groups[key] = []
                groups[key].append(path)
        
        # Sort paths within each group to maintain some order (v1, v2...)
        final_groups = []
        for (label, fruit_id), paths in groups.items():
            if len(paths) >= 1: # Use any available views
                paths.sort()
                final_groups.append((paths, label))
        
        print(f"Split: {os.path.basename(self.split_dir)} | Grouped Fruits: {len(final_groups)}")
        return final_groups

    def __len__(self):
        return len(self.grouped_samples)

    def __getitem__(self, idx):
        paths, target = self.grouped_samples[idx]
        images = []
        for path in paths:
            img = self.dataset.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
        
        # Return a stack of 6 images: [6, 3, 224, 224]
        return torch.stack(images), target

def get_multiview_dataloaders(root_dir, batch_size=8): # Small batch size for multi-view
    train_transform, val_transform = get_transforms(augment=True)
    
    train_dataset = LIMTICMultiViewDataset(root_dir, 'Train', transform=train_transform)
    val_dataset = LIMTICMultiViewDataset(root_dir, 'Validation', transform=val_transform)
    test_dataset = LIMTICMultiViewDataset(root_dir, 'Test', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
