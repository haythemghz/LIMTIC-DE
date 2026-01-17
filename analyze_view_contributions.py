import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from experiment_utils import LIMTICDataset
import numpy as np
import os
import pandas as pd

# Config
DATA_DIR = '../LIMTIC-DE_Dataset/LIMTIC-DE_Dataset'
CHECKPOINT_DIR = 'results_ablation'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(model_path, num_classes=11):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    return model.to(DEVICE)

def evaluate_per_class(model, dataloader, device):
    model.eval()
    correct_pred = {classname: 0 for classname in dataloader.dataset.classes}
    total_pred = {classname: 0 for classname in dataloader.dataset.classes}
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[dataloader.dataset.classes[label]] += 1
                total_pred[dataloader.dataset.classes[label]] += 1
                
    accuracies = {}
    for classname, correct_count in correct_pred.items():
        total_count = total_pred[classname]
        if total_count > 0:
            accuracies[classname] = float(correct_count) / total_count
        else:
            accuracies[classname] = 0.0
            
    return accuracies

def main():
    print("Analyzing View Contribution Per Class...")
    
    # Define models to compare
    configs = [
        {'name': 'Full View Only', 'ckpt': 'vit_full_augFalse_seed42_best.pth', 'view_mode': 'full'},
        {'name': 'Side View Only', 'ckpt': 'vit_side_augFalse_seed42_best.pth', 'view_mode': 'side'}
        # Note: We can also add 'All Views' if we have the checkpoint
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nProcessing {config['name']}...")
        ckpt_path = os.path.join(CHECKPOINT_DIR, config['ckpt'])
        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint not found: {ckpt_path}")
            continue
            
        # Helper to get transforms and loaders
        # We need the validation/test transform
        # We import get_dataloaders/get_transforms but we can just use LIMTICDataset directly with simple transform logic
        # Ideally we use get_dataloaders from experiment_utils but it returns 3 loaders.
        # Let's instantiate Test dataset directly.
        from experiment_utils import get_transforms
        _, val_transform = get_transforms(augment=False)
        
        # IMPORTANT: We evaluate on the TEST set of that specific view mode
        # This tells us how well it performs on ITS OWN domain.
        # But we want to see per-class performance.
        dataset = LIMTICDataset(DATA_DIR, 'Test', view_mode=config['view_mode'], transform=val_transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        model = get_model(ckpt_path)
        accuracies = evaluate_per_class(model, loader, DEVICE)
        results[config['name']] = accuracies
        
    # Create DataFrame comparison
    df = pd.DataFrame(results)
    df['Delta (Side - Full)'] = df['Side View Only'] - df['Full View Only']
    
    print("\n--- Per-Class Accuracy Comparison (Test Set) ---")
    # Sort by Delta to see where Side helps most
    df = df.sort_values(by='Delta (Side - Full)', ascending=False)
    print(df.to_string(float_format="{:.4f}".format))
    
    # Save to CSV for retrieval
    df.to_csv('view_comparison.csv')
    print("Saved view_comparison.csv")

    
    # Add maturity analysis if classes allows
    # Nour classes: 'Deglet Nour ...'
    print("\n--- Maturity Analysis (Nour) ---")
    nour_classes = [c for c in df.index if 'Deglet Nour' in c]
    if nour_classes:
        nour_df = df.loc[nour_classes]
        print(nour_df.mean().to_frame(name='Average Accuracy').T.to_string(float_format="{:.4f}".format))

if __name__ == "__main__":
    main()
