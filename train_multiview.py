import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from multiview_utils import get_multiview_dataloaders
import time

class LateFusionResNet(nn.Module):
    def __init__(self, num_classes=11, num_views=6):
        super(LateFusionResNet, self).__init__()
        self.num_views = num_views
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Shared backbone
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: [batch, num_views, 3, 224, 224]
        batch_size = x.size(0)
        num_views = x.size(1)
        # Reshape to process all views at once: [batch*num_views, 3, 224, 224]
        x = x.view(-1, 3, 224, 224)
        features = self.backbone(x)
        # Reshape back to [batch, num_views, in_features]
        features = features.view(batch_size, num_views, -1)
        # Late Fusion: Average pooling across views
        fused_features = torch.mean(features, dim=1)
        out = self.classifier(fused_features)
        return out

def train_multiview(epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_multiview_dataloaders('c:/Users/Dell/Desktop/Article_Deglet_Nour/LIMTIC-DE_Dataset/LIMTIC-DE_Dataset', batch_size=1)
    
    model = LateFusionResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}")

    print("Multi-View Training Complete.")

if __name__ == "__main__":
    train_multiview(epochs=2)
