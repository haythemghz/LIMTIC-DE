import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from sklearn.metrics import accuracy_score, silhouette_score
from tqdm import tqdm
import experiment_utils
from train_metric import MetricModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = '../LIMTIC-DE_Dataset/LIMTIC-DE_Dataset'
    model_path = 'results_metric/resnet_arcface_best.pth'
    
    # Load Classes
    _, _, test_loader, classes = experiment_utils.get_dataloaders(data_dir, batch_size=32)
    num_classes = len(classes)

    # Class Mapping (Variety level)
    variety_map = {
        0: 'Deglet Nour', 1: 'Deglet Nour', 2: 'Deglet Nour', 3: 'Deglet Nour', 4: 'Deglet Nour', 5: 'Deglet Nour', 6: 'Deglet Nour',
        7: 'Alig', 8: 'Bessra', 9: 'Kenta', 10: 'Kintichi'
    }
    
    maturity_map = {
        0: 'Dry', 1: 'Oily', 2: 'Oily', 3: 'Semi-Dry', 4: 'Semi-Dry', 5: 'Semi-Oily', 6: 'Semi-Oily'
    }

    # Inverse mapping for Variety Grouping
    variety_names = ['Deglet Nour', 'Alig', 'Bessra', 'Kenta', 'Kintichi']
    
    # Load Metric Model
    model = MetricModel(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_embeddings = []
    all_fine_labels = []
    
    print("Extracting embeddings for Q3-Q4 validation...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_fine_labels.append(labels.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_fine_labels = np.concatenate(all_fine_labels, axis=0)
    
    # --- Q3: Cross-Stage Morphological Generalization ---
    # We evaluate Variety classification accuracy across different Maturity sub-groups
    variety_preds = []
    variety_true = [variety_map[l] for l in all_fine_labels]
    
    # Predict Variety by taking the max logit and mapping to variety
    weights_norm = torch.nn.functional.normalize(model.arcface.weight, dim=1).detach().cpu().numpy()
    embeddings_norm = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    logits = np.dot(embeddings_norm, weights_norm.T) * model.arcface.s
    fine_preds = np.argmax(logits, axis=1)
    variety_preds = [variety_map[l] for l in fine_preds]
    
    print("\n--- Analysis of Q3: Cross-Stage Generalization ---")
    maturity_scores = {}
    # Only for Deglet Nour (0-6)
    for m_idx, m_name in maturity_map.items():
        mask = (all_fine_labels == m_idx)
        if np.any(mask):
            acc = accuracy_score(np.array(variety_true)[mask], np.array(variety_preds)[mask])
            maturity_scores[m_name] = maturity_scores.get(m_name, []) + [acc]

    # Average scores per maturity stage
    final_m_scores = {k: np.mean(v) for k, v in maturity_scores.items()}
    for m, score in final_m_scores.items():
        print(f"Variety Accuracy at '{m}' stage: {score:.4f}")
    
    avg_gen = np.mean(list(final_m_scores.values()))
    std_gen = np.std(list(final_m_scores.values()))
    print(f"Mean Generalization Accuracy: {avg_gen:.4f} (±{std_gen:.4f})")

    # --- Q4: Texture Fingerprinting for Traceability ---
    # We use Silhouette Score on embeddings grouped by Variety to prove cluster tightness
    variety_labels_numeric = [variety_names.index(variety_map[l]) for l in all_fine_labels]
    sil_score = silhouette_score(all_embeddings, variety_labels_numeric)
    
    print("\n--- Analysis of Q4: Texture Fingerprinting ---")
    print(f"Variety Silhouette Score: {sil_score:.4f}")
    
    # Texture Discriminability Index (TDI) - ratio of inter-class to intra-class distance
    # Simplified version: Silhouette score already provides this.
    
    results_path = "results_metric/q3_q4_validation.txt"
    with open(results_path, "w") as f:
        f.write("Q3 & Q4 Experimental Validation Results\n")
        f.write("=======================================\n")
        f.write(f"Q3: Cross-Stage Generalization Accuracy (Variety): {avg_gen:.4f}\n")
        f.write(f"Q3: Multi-Stage Consistency (Std Dev): {std_gen:.4f}\n")
        for m, s in final_m_scores.items():
            f.write(f"  - Maturity Stage {m}: {s:.4f}\n")
        f.write(f"\nQ4: Texture Fingerprint (Silhouette Score): {sil_score:.4f}\n")

    print(f"\nValidation complete. Results saved to {results_path}")

if __name__ == '__main__':
    main()
