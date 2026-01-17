<<<<<<< HEAD
import torch
import time
import numpy as np
from torchvision import models
import torch.nn as nn

def get_model(model_name, num_classes):
    if model_name == 'vit':
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'resnet':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def benchmark(model_name, device):
    model = get_model(model_name, 11).to(device)
    model.eval()
    
    # Dummy input
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
        
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            latencies.append((time.time() - start) * 1000) # ms
            
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    print(f"{model_name}: {mean_lat:.2f} ± {std_lat:.2f} ms/sample")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    
    benchmark('vit', device)
    benchmark('resnet', device)
    benchmark('efficientnet', device)
=======
import torch
import time
import numpy as np
from torchvision import models
import torch.nn as nn

def get_model(model_name, num_classes):
    if model_name == 'vit':
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'resnet':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def benchmark(model_name, device):
    model = get_model(model_name, 11).to(device)
    model.eval()
    
    # Dummy input
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
        
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            latencies.append((time.time() - start) * 1000) # ms
            
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    print(f"{model_name}: {mean_lat:.2f} ± {std_lat:.2f} ms/sample")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    
    benchmark('vit', device)
    benchmark('resnet', device)
    benchmark('efficientnet', device)
>>>>>>> origin/main
