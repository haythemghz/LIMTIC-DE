import os
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import numpy as np
import experiment_utils
from PIL import Image, ImageFilter, ImageEnhance

class IndustrialValidator:
    def __init__(self, model_type, model_path, dataset_root):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        # Initialize architecture
        if model_type == 'vit':
            self.model = models.vit_b_16()
            self.model.heads.head = nn.Linear(self.model.heads.head.in_features, 11)
        elif model_type == 'resnet':
            self.model = models.resnet50()
            self.model.fc = nn.Linear(self.model.fc.in_features, 11)
        elif model_type == 'efficientnet':
            self.model = models.efficientnet_b0()
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 11)
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.dataset_root = dataset_root
        _, self.val_transform = experiment_utils.get_transforms(augment=False)

    def measure_throughput(self, batch_size=1, warmups=10, iterations=100):
        print(f"\n--- Measuring Throughput ({self.model_type}, BS={batch_size}) ---")
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        # Warmup
        for _ in range(warmups):
            _ = self.model(dummy_input)
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        for _ in range(iterations):
            _ = self.model(dummy_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()
        
        avg_latency = (end - start) / iterations * 1000 # ms
        throughput_sec = 1000 / avg_latency * batch_size
        throughput_min = throughput_sec * 60
        
        print(f"Avg Latency: {avg_latency:.2f} ms")
        print(f"Throughput: {throughput_min:.0f} units/minute")
        return avg_latency, throughput_min

    def test_robustness(self, condition='blur', levels=[1, 2, 3]):
        print(f"\n--- Industrial Robustness Test: {condition} ---")
        test_dataset = experiment_utils.LIMTICDataset(self.dataset_root, 'Test', transform=None)
        
        results = {}
        for level in levels:
            correct = 0
            total = 0
            
            for img_path, label in test_dataset.samples:
                img = Image.open(img_path).convert('RGB')
                
                # Apply Industrial Distortion
                if condition == 'blur':
                    # Motion Blur simulation
                    img = img.filter(ImageFilter.GaussianBlur(radius=level))
                elif condition == 'lighting':
                    # Varying illumination
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(1.0 + (level - 2) * 0.3) # level 1=0.7, 2=1.0, 3=1.3
                
                # Preprocess
                img_tensor = self.val_transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(img_tensor)
                    _, pred = torch.max(output, 1)
                    
                total += 1
                if pred.item() == label:
                    correct += 1
            
            acc = 100 * correct / total
            results[level] = acc
            print(f"Level {level}: Accuracy {acc:.2f}%")
        return results

def main():
    data_dir = '../LIMTIC-DE_Dataset/LIMTIC-DE_Dataset'
    results_dir = 'results_ablation'
    
    models_to_test = [
        ('vit', os.path.join(results_dir, 'vit_all_augFalse_seed42_best.pth')),
        ('resnet', '../limtic_de_experiments/results_ablation/resnet50_placeholder.pth'), # Need to check actual name
        ('efficientnet', os.path.join(results_dir, 'efficientnet_all_augFalse_seed42_best.pth'))
    ]
    
    # Correction: The resnet name might be different. Let's list result dir.
    
    # We'll run for EfficientNet and ViT first as we confirmed those exist.
    for m_type, m_path in [models_to_test[0], models_to_test[2]]:
        if not os.path.exists(m_path):
            print(f"Skipping {m_type}, path not found: {m_path}")
            continue
            
        validator = IndustrialValidator(m_type, m_path, data_dir)
        latency, tput = validator.measure_throughput()
        
        blur_results = validator.test_robustness('blur', levels=[0, 1, 2])
        light_results = validator.test_robustness('lighting', levels=[1, 2, 3])
        
        # Save results
        with open(f"results_ablation/industrial_{m_type}.txt", "w") as f:
            f.write(f"Industrial Validation for {m_type}\n")
            f.write(f"Latency: {latency:.2f} ms | Throughput: {tput:.0f} units/min\n")
            f.write(f"Blur Robustness: {blur_results}\n")
            f.write(f"Lighting Robustness: {light_results}\n")

if __name__ == "__main__":
    main()
