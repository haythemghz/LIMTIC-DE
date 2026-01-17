import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import experiment_utils
from sklearn.metrics import accuracy_score, f1_score

def get_model(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1

def main():
    parser = argparse.ArgumentParser(description='Resolution Sensitivity Study')
    parser.add_argument('--res', type=int, required=True, help='Resolution to train on')
    parser.add_argument('--data_dir', type=str, default='../LIMTIC-DE_Dataset/LIMTIC-DE_Dataset', help='Dataset root')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Adjust batch size for high resolution to avoid OOM
    if args.res >= 512:
        args.batch_size = 8
    if args.res >= 1024:
        args.batch_size = 2 # 1280 will be tight on 4GB

    print(f"\n===== Training at Resolution {args.res}x{args.res} (BS={args.batch_size}) =====")

    # Custom transforms for resolution
    train_transform = transforms.Compose([
        transforms.Resize((args.res, args.res)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.res, args.res)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = experiment_utils.LIMTICDataset(args.data_dir, 'Train', transform=train_transform)
    val_ds = experiment_utils.LIMTICDataset(args.data_dir, 'Validation', transform=val_transform)
    test_ds = experiment_utils.LIMTICDataset(args.data_dir, 'Test', transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = len(train_ds.classes)
    model = get_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_f1 = 0
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc, f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} - Loss: {loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        
    # Final Test
    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"\nFinal Test (Res={args.res}): Acc={test_acc:.4f}, F1={test_f1:.4f}")
    
    # Save results
    os.makedirs('results_resolution', exist_ok=True)
    with open(f"results_resolution/res_{args.res}.txt", "w") as f:
        f.write(f"Resolution: {args.res}\nAccuracy: {test_acc:.4f}\nF1: {test_f1:.4f}\n")

if __name__ == '__main__':
    main()
