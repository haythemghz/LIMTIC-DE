import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import experiment_utils
from metric_learning_utils import ArcMarginProduct
from sklearn.metrics import classification_report, accuracy_score

class MetricModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super(MetricModel, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_size)
        self.arcface = ArcMarginProduct(embedding_size, num_classes, s=30.0, m=0.50)

    def forward(self, x, labels=None):
        embeddings = self.backbone(x)
        if labels is not None:
            return self.arcface(embeddings, labels)
        return embeddings

def evaluate_metric(model, loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    # For evaluation, we compute logits using the weight matrix of the ArcFace head
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            # Logits calculation: cos(theta) * s
            logits = torch.matmul(torch.nn.functional.normalize(embeddings), 
                                 torch.nn.functional.normalize(model.arcface.weight, dim=1).t())
            logits *= model.arcface.s
            
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    return acc, report

def main():
    parser = argparse.ArgumentParser(description='LIMTIC-DE Metric Learning')
    parser.add_argument('--data_dir', type=str, default='../LIMTIC-DE_Dataset/LIMTIC-DE_Dataset', help='Path to dataset root')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='results_metric', help='Directory to save results')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, classes = experiment_utils.get_dataloaders(args.data_dir, batch_size=args.batch_size)
    num_classes = len(classes)

    model = MetricModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        val_acc, _ = evaluate_metric(model, val_loader, device, classes)
        print(f"Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "resnet_arcface_best.pth"))

    # Final Test
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "resnet_arcface_best.pth")))
    test_acc, report = evaluate_metric(model, test_loader, device, classes)
    print(f"Test Acc: {test_acc:.4f}\n{report}")
    
    with open(os.path.join(args.save_dir, "metric_results.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)

if __name__ == '__main__':
    main()
