<<<<<<< HEAD
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from experiment_utils import LIMTICDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# Configuration
DATA_DIR = '../LIMTIC-DE_Dataset/LIMTIC-DE_Dataset'
MODEL_PATH = 'results_ablation/vit_all_augFalse_seed42_best.pth'
SAVE_PATH = 'visualizations/tsne_plot.png'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ViTFeatureExtractor(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        # Load the same architecture as training
        self.backbone = models.vit_b_16(weights=None)
        # We need to recreate the head to load the state dict correctly
        self.backbone.heads.head = nn.Linear(self.backbone.heads.head.in_features, num_classes)
        
    def forward(self, x):
        # Forward pass to get embeddings (pre-classifier)
        # ViT implementation in torchvision:
        # x = self.conv_proj(x)
        # x = x.flatten(2).transpose(1, 2)
        # x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        # x = x + self.encoder.pos_embedding
        # x = self.encoder.dropout(x)
        # x = self.encoder.layers(x)
        # x = self.encoder.ln(x)
        # x = x[:, 0]
        # return x
        
        # Easier way using hooks or just dissecting:
        # We can just optimize by returning the representation from the penultimate layer
        # But torchvision models are tricky. let's use a hook or modify the forward.
        
        # Alternative: The 'representation' output from vit_b_16 gives the CLS token
        # output = self.backbone._process_input(x)
        # n = output.shape[0]
        # batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        # output = torch.cat([batch_class_token, output], dim=1)
        # output = output + self.backbone.encoder.pos_embedding
        # output = self.backbone.encoder.dropout(output)
        # output = self.backbone.encoder.layers(output)
        # output = self.backbone.encoder.ln(output)
        # return output[:, 0]
        pass

# Simplified: Load model, remove head
def get_feature_extractor(model_path, num_classes=11):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # Assuming full state dict
        
    # Remove the classification head to get embeddings (768-dim)
    model.heads.head = nn.Identity()
    return model.to(DEVICE)

def generate_tsne():
    os.makedirs('visualizations', exist_ok=True)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LIMTICDataset(
        root_dir=DATA_DIR,
        split='Test',
        view_mode='all', # We want to visualize the full capability
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = get_feature_extractor(MODEL_PATH)
    model.eval()
    
    embeddings = []
    labels = []
    
    print("Extracting features...")
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(DEVICE)
            # Forward pass
            feats = model(imgs) # Should return (B, 768) due to Identity head
            embeddings.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"Extracted {embeddings.shape[0]} samples with {embeddings.shape[1]} dimensions.")
    
    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    # Plotting
    print("Plotting...")
    plt.figure(figsize=(10, 8))
    classes = dataset.classes
    # Define a distinct color map
    colors = plt.cm.get_cmap('tab20', len(classes))
    
    for i, class_name in enumerate(classes):
        indices = labels == i
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                    label=class_name, s=10, alpha=0.7, color=colors(i))
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.title('t-SNE Visualization of ViT Embeddings (Test Set)')
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300)
    print(f"t-SNE plot saved to {SAVE_PATH}")

if __name__ == "__main__":
    generate_tsne()
=======
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from experiment_utils import LIMTICDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# Configuration
DATA_DIR = '../LIMTIC-DE_Dataset/LIMTIC-DE_Dataset'
MODEL_PATH = 'results_ablation/vit_all_augFalse_seed42_best.pth'
SAVE_PATH = 'visualizations/tsne_plot.png'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ViTFeatureExtractor(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        # Load the same architecture as training
        self.backbone = models.vit_b_16(weights=None)
        # We need to recreate the head to load the state dict correctly
        self.backbone.heads.head = nn.Linear(self.backbone.heads.head.in_features, num_classes)
        
    def forward(self, x):
        # Forward pass to get embeddings (pre-classifier)
        # ViT implementation in torchvision:
        # x = self.conv_proj(x)
        # x = x.flatten(2).transpose(1, 2)
        # x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)
        # x = x + self.encoder.pos_embedding
        # x = self.encoder.dropout(x)
        # x = self.encoder.layers(x)
        # x = self.encoder.ln(x)
        # x = x[:, 0]
        # return x
        
        # Easier way using hooks or just dissecting:
        # We can just optimize by returning the representation from the penultimate layer
        # But torchvision models are tricky. let's use a hook or modify the forward.
        
        # Alternative: The 'representation' output from vit_b_16 gives the CLS token
        # output = self.backbone._process_input(x)
        # n = output.shape[0]
        # batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        # output = torch.cat([batch_class_token, output], dim=1)
        # output = output + self.backbone.encoder.pos_embedding
        # output = self.backbone.encoder.dropout(output)
        # output = self.backbone.encoder.layers(output)
        # output = self.backbone.encoder.ln(output)
        # return output[:, 0]
        pass

# Simplified: Load model, remove head
def get_feature_extractor(model_path, num_classes=11):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # Assuming full state dict
        
    # Remove the classification head to get embeddings (768-dim)
    model.heads.head = nn.Identity()
    return model.to(DEVICE)

def generate_tsne():
    os.makedirs('visualizations', exist_ok=True)
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LIMTICDataset(
        root_dir=DATA_DIR,
        split='Test',
        view_mode='all', # We want to visualize the full capability
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = get_feature_extractor(MODEL_PATH)
    model.eval()
    
    embeddings = []
    labels = []
    
    print("Extracting features...")
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(DEVICE)
            # Forward pass
            feats = model(imgs) # Should return (B, 768) due to Identity head
            embeddings.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    print(f"Extracted {embeddings.shape[0]} samples with {embeddings.shape[1]} dimensions.")
    
    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    # Plotting
    print("Plotting...")
    plt.figure(figsize=(10, 8))
    classes = dataset.classes
    # Define a distinct color map
    colors = plt.cm.get_cmap('tab20', len(classes))
    
    for i, class_name in enumerate(classes):
        indices = labels == i
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                    label=class_name, s=10, alpha=0.7, color=colors(i))
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.title('t-SNE Visualization of ViT Embeddings (Test Set)')
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300)
    print(f"t-SNE plot saved to {SAVE_PATH}")

if __name__ == "__main__":
    generate_tsne()
>>>>>>> origin/main
