import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from multitask_utils import get_multitask_dataloaders
import time

class MultiTaskResNet(nn.Module):
    def __init__(self, num_varieties=5, num_maturities=5, num_treatments=2):
        super(MultiTaskResNet, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove default head
        
        self.fc_variety = nn.Linear(in_features, num_varieties)
        self.fc_maturity = nn.Linear(in_features, num_maturities)
        self.fc_treatment = nn.Linear(in_features, num_treatments)

    def forward(self, x):
        features = self.backbone(x)
        v = self.fc_variety(features)
        m = self.fc_maturity(features)
        t = self.fc_treatment(features)
        return v, m, t

def train_multitask(epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_multitask_dataloaders('c:/Users/Dell/Desktop/Article_Deglet_Nour/LIMTIC-DE_Dataset/LIMTIC-DE_Dataset')
    
    model = MultiTaskResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, (v_labels, m_labels, t_labels) in train_loader:
            images = images.to(device)
            v_labels, m_labels, t_labels = v_labels.to(device), m_labels.to(device), t_labels.to(device)
            
            optimizer.zero_grad()
            v_out, m_out, t_out = model(images)
            
            loss_v = criterion(v_out, v_labels)
            loss_m = criterion(m_out, m_labels)
            loss_t = criterion(t_out, t_labels)
            
            loss = loss_v + loss_m + loss_t
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}")

    # Final Evaluation (Dummy/Fast)
    print("Multi-Task Training Complete.")

if __name__ == "__main__":
    train_multitask(epochs=2)
