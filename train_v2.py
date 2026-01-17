<<<<<<< HEAD
import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import experiment_utils

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_model(model_name, num_classes, pretrained=True):
    if model_name == 'vit':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_labels, all_preds

def main():
    parser = argparse.ArgumentParser(description='LIMTIC-DE Training Script')
    parser.add_argument('--data_dir', type=str, default='../LIMTIC-DE_Dataset/LIMTIC-DE_Dataset', help='Path to dataset root')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'resnet', 'efficientnet'], help='Model architecture')
    parser.add_argument('--view_mode', type=str, default='all', choices=['all', 'full', 'side'], help='Viewpoint filtering mode')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    run_name = f"{args.model}_{args.view_mode}_aug{args.augment}_seed{args.seed}"
    print(f"Starting run: {run_name}")
    
    # Data Loading
    train_loader, val_loader, test_loader, classes = experiment_utils.get_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        view_mode=args.view_mode,
        augment=args.augment
    )
    print(f"Classes: {classes}")
    
    # Model
    model = get_model(args.model, len(classes)).to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training Loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{run_name}_best.pth"))
            print("Saved new best model.")
            
    # Final Test
    print("Loading best model for testing...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f"{run_name}_best.pth"), map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save detailed metrics
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(report)
    
    cm = confusion_matrix(y_true, y_pred)
    with open(os.path.join(args.save_dir, f"{run_name}_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        np.savetxt(f, cm, fmt='%d')

    print("Experiment completed.")

if __name__ == '__main__':
    main()
=======
import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import experiment_utils

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_model(model_name, num_classes, pretrained=True):
    if model_name == 'vit':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'resnet':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_labels, all_preds

def main():
    parser = argparse.ArgumentParser(description='LIMTIC-DE Training Script')
    parser.add_argument('--data_dir', type=str, default='../LIMTIC-DE_Dataset/LIMTIC-DE_Dataset', help='Path to dataset root')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'resnet', 'efficientnet'], help='Model architecture')
    parser.add_argument('--view_mode', type=str, default='all', choices=['all', 'full', 'side'], help='Viewpoint filtering mode')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    run_name = f"{args.model}_{args.view_mode}_aug{args.augment}_seed{args.seed}"
    print(f"Starting run: {run_name}")
    
    # Data Loading
    train_loader, val_loader, test_loader, classes = experiment_utils.get_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        view_mode=args.view_mode,
        augment=args.augment
    )
    print(f"Classes: {classes}")
    
    # Model
    model = get_model(args.model, len(classes)).to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training Loop
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"{run_name}_best.pth"))
            print("Saved new best model.")
            
    # Final Test
    print("Loading best model for testing...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f"{run_name}_best.pth"), map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save detailed metrics
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(report)
    
    cm = confusion_matrix(y_true, y_pred)
    with open(os.path.join(args.save_dir, f"{run_name}_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        np.savetxt(f, cm, fmt='%d')

    print("Experiment completed.")

if __name__ == '__main__':
    main()
>>>>>>> origin/main
