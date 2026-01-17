import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from experiment_utils import LIMTICDataset
import numpy as np

def add_gaussian_noise(image, std=0.1):
    return image + torch.randn_like(image) * std

class StressTest:
    def __init__(self, model_path, dataset_root):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load a ResNet50 as dummy if model_path is not found
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(self.device)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 11)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.dataset_root = dataset_root

    def evaluate(self, transform):
        test_dataset = LIMTICDataset(self.dataset_root, 'Test', transform=transform)
        loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def run_resolution_test(self):
        print("\n--- Resolution Stress Test ---")
        resolutions = [224, 112, 56, 28]
        results = {}
        for res in resolutions:
            transform = transforms.Compose([
                transforms.Resize((res, res)),
                transforms.Resize((224, 224)), # Upscale back for model
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            acc = self.evaluate(transform)
            results[res] = acc
            print(f"Res {res}x{res}: {acc:.2f}%")
        return results

    def run_noise_test(self):
        print("\n--- Noise Stress Test ---")
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.4]
        results = {}
        base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        for level in noise_levels:
            def noise_transform(img):
                img = base_transform(img)
                return add_gaussian_noise(img, std=level)
            
            # Custom evaluate for noise_transform
            test_dataset = LIMTICDataset(self.dataset_root, 'Test', transform=None)
            loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in loader:
                    images = torch.stack([noise_transform(transforms.ToPILImage()(img)) for img in images]).to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            results[level] = acc
            print(f"Noise Std {level}: {acc:.2f}%")
        return results

import os
if __name__ == "__main__":
    tester = StressTest('best_model.pth', 'c:/Users/Dell/Desktop/Article_Deglet_Nour/LIMTIC-DE_Dataset/LIMTIC-DE_Dataset')
    # Start tests
    tester.run_resolution_test()
    # tester.run_noise_test() # Skip noise for now to avoid overhead
