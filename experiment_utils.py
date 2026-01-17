<<<<<<< HEAD
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import re

# Dataset Norm Config
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_transforms(augment=False):
    """
    Returns training and validation transforms.
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    return train_transform, val_transform

class LIMTICDataset(Dataset):
    def __init__(self, root_dir, split, view_mode='all', transform=None):
        """
        Args:
            root_dir (str): Path to dataset root (containing Train, Validation, Test).
            split (str): 'Train', 'Validation', or 'Test'.
            view_mode (str): 'all', 'full', 'side'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split_dir = os.path.join(root_dir, split)
        self.view_mode = view_mode
        self.transform = transform
        
        # Use ImageFolder to load all samples first
        self.dataset = datasets.ImageFolder(self.split_dir)
        self.classes = self.dataset.classes
        
        # Filter samples based on view_mode
        self.samples = self._filter_samples(self.dataset.samples)

    def _filter_samples(self, samples):
        if self.view_mode == 'all':
            return samples
        
        filtered_samples = []
        for path, label in samples:
            filename = os.path.basename(path)
            # Try to extract the index from filename (e.g., alig_100_jpg...)
            # Regex to find the number after the first underscore and before _jpg
            match = re.search(r'_(\d+)_jpg', filename)
            if match:
                idx = int(match.group(1))
                # Hypothesis: 0, 1 are Full Views; 2, 3, 4, 5 are Side Views
                # The pattern repeats every 6 images
                view_idx = idx % 6
                
                if self.view_mode == 'full':
                    if view_idx in [0, 1]:
                        filtered_samples.append((path, label))
                elif self.view_mode == 'side':
                    if view_idx in [2, 3, 4, 5]:
                        filtered_samples.append((path, label))
            else:
                # If regex fails, keep the sample to be safe, or log warning?
                # For now, we assume matching failed means we keep it if mode is all,
                # but drop if specific mode since we can't determine.
                print(f"Warning: Could not determine view for {filename}, skipping.")
        
        print(f"Split: {os.path.basename(self.split_dir)} | Mode: {self.view_mode} | Samples: {len(filtered_samples)}/{len(samples)}")
        return filtered_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.dataset.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def get_dataloaders(root_dir, batch_size=32, view_mode='all', augment=False):
    train_transform, val_transform = get_transforms(augment=augment)
    
    train_dataset = LIMTICDataset(root_dir, 'Train', view_mode, transform=train_transform)
    val_dataset = LIMTICDataset(root_dir, 'Validation', view_mode, transform=val_transform)
    test_dataset = LIMTICDataset(root_dir, 'Test', view_mode, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, train_dataset.classes
=======
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import re

# Dataset Norm Config
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_transforms(augment=False):
    """
    Returns training and validation transforms.
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    return train_transform, val_transform

class LIMTICDataset(Dataset):
    def __init__(self, root_dir, split, view_mode='all', transform=None):
        """
        Args:
            root_dir (str): Path to dataset root (containing Train, Validation, Test).
            split (str): 'Train', 'Validation', or 'Test'.
            view_mode (str): 'all', 'full', 'side'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split_dir = os.path.join(root_dir, split)
        self.view_mode = view_mode
        self.transform = transform
        
        # Use ImageFolder to load all samples first
        self.dataset = datasets.ImageFolder(self.split_dir)
        self.classes = self.dataset.classes
        
        # Filter samples based on view_mode
        self.samples = self._filter_samples(self.dataset.samples)

    def _filter_samples(self, samples):
        if self.view_mode == 'all':
            return samples
        
        filtered_samples = []
        for path, label in samples:
            filename = os.path.basename(path)
            # Try to extract the index from filename (e.g., alig_100_jpg...)
            # Regex to find the number after the first underscore and before _jpg
            match = re.search(r'_(\d+)_jpg', filename)
            if match:
                idx = int(match.group(1))
                # Hypothesis: 0, 1 are Full Views; 2, 3, 4, 5 are Side Views
                # The pattern repeats every 6 images
                view_idx = idx % 6
                
                if self.view_mode == 'full':
                    if view_idx in [0, 1]:
                        filtered_samples.append((path, label))
                elif self.view_mode == 'side':
                    if view_idx in [2, 3, 4, 5]:
                        filtered_samples.append((path, label))
            else:
                # If regex fails, keep the sample to be safe, or log warning?
                # For now, we assume matching failed means we keep it if mode is all,
                # but drop if specific mode since we can't determine.
                print(f"Warning: Could not determine view for {filename}, skipping.")
        
        print(f"Split: {os.path.basename(self.split_dir)} | Mode: {self.view_mode} | Samples: {len(filtered_samples)}/{len(samples)}")
        return filtered_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.dataset.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def get_dataloaders(root_dir, batch_size=32, view_mode='all', augment=False):
    train_transform, val_transform = get_transforms(augment=augment)
    
    train_dataset = LIMTICDataset(root_dir, 'Train', view_mode, transform=train_transform)
    val_dataset = LIMTICDataset(root_dir, 'Validation', view_mode, transform=val_transform)
    test_dataset = LIMTICDataset(root_dir, 'Test', view_mode, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, train_dataset.classes
>>>>>>> origin/main
