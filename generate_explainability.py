import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
import experiment_utils
import matplotlib.pyplot as plt

def get_attention_rollout(model, input_tensor):
    # This is a simplified version of Attention Rollout for torchvision ViT
    # We need to hook the attention weights
    attentions = []
    
    def hook_fn(module, input, output):
        # The attention matrix is computed inside the MultiheadAttention layer
        # However, torchvision doesn't easily expose it.
        # We will use a more direct approach: extracting the scaled dot product
        pass

    # For torchvision ViT, the attention is computed in EncoderBlock.self_attention
    # It's better to implement a wrapper or use a library, but we'll try to do it manually
    # by modifying the forward pass of the attention blocks if possible, 
    # or by extracting weights if they were stored (not the case here).
    
    # ALTERNATIVE: Use Grad-CAM on the last attention block's normalization layer
    # This is often more robust for torchvision models.
    return None

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        grad_cam = torch.relu(grad_cam)
        
        # Normalize
        grad_cam = grad_cam - grad_cam.min()
        grad_cam = grad_cam / (grad_cam.max() + 1e-8)
        
        return grad_cam.detach().cpu().numpy()[0, 0], target_class

def overlay_heatmap(img, heatmap):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img) / 255.0
    heatmap_float = np.float32(heatmap) / 255.0
    overlay = heatmap_float * 0.4 + img_float * 0.6
    overlay = np.uint8(255 * overlay)
    return overlay

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '../LIMTIC-DE_Dataset/LIMTIC-DE_Dataset'
    model_path = 'results_ablation/efficientnet_all_augFalse_seed42_best.pth'
    output_dir = 'visualizations/explainability'
    os.makedirs(output_dir, exist_ok=True)

    # Load Classes
    _, _, _, classes = experiment_utils.get_dataloaders(data_dir, batch_size=1)
    num_classes = len(classes)

    # Load Model (EfficientNet for robust Grad-CAM)
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Target Layer for EfficientNet-B0 (last conv layer before pooling)
    target_layer = model.features[-1]
    cam = GradCAM(model, target_layer)

    # Define pairs to visualize
    pairs = [
        ('Deglet Nour oily', 'Deglet Nour oily treated'),
        ('Deglet Nour semi-dryer', 'Deglet Nour semi-oily')
    ]

    _, val_transform = experiment_utils.get_transforms(augment=False)
    
    # Basic Inverse Transform for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    fig, axes = plt.subplots(len(pairs), 4, figsize=(16, 8))
    
    for row, (class_a, class_b) in enumerate(pairs):
        for col, class_name in enumerate([class_a, class_b]):
            # Find an image for this class in Test set
            class_idx = classes.index(class_name)
            test_dir = os.path.join(data_dir, 'Test', class_name)
            img_name = os.listdir(test_dir)[random.randint(0, 10)] # Pick a random sample from first 10
            img_path = os.path.join(test_dir, img_name)
            
            # Load and Preprocess
            img_pil = Image.open(img_path).convert('RGB')
            img_res = img_pil.resize((224, 224))
            input_tensor = val_transform(img_res).unsqueeze(0).to(device)
            
            # Generate CAM
            heatmap, pred_idx = cam.generate(input_tensor, target_class=class_idx)
            
            # Post-process image for display
            img_disp = np.array(img_res)
            heatmap_res = cv2.resize(heatmap, (224, 224))
            overlay = overlay_heatmap(img_disp, heatmap_res)
            
            # Plot
            axes[row, col*2].imshow(img_disp)
            axes[row, col*2].set_title(f"Orig: {class_name}\n(Pred: {classes[pred_idx]})")
            axes[row, col*2].axis('off')
            
            axes[row, col*2+1].imshow(overlay)
            axes[row, col*2+1].set_title("Grad-CAM Highlight")
            axes[row, col*2+1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'explainability_comparison.png')
    plt.savefig(save_path)
    print(f"Saved explainability visualization to {save_path}")

if __name__ == '__main__':
    import random
    main()
