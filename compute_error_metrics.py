import numpy as np
import os

# Read confusion matrix from metrics file
def parse_confusion_matrix(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the confusion matrix section
    lines = content.split('\n')
    cm_lines = []
    in_cm = False
    for line in lines:
        if 'Confusion Matrix:' in line:
            in_cm = True
            continue
        if in_cm:
            if line.strip() and line.strip()[0] == '[':
                # Parse numpy array row
                row = line.strip().strip('[').strip(']').split()
                cm_lines.append([int(x.strip('[]')) for x in row if x.strip('[]').isdigit()])
    
    return np.array(cm_lines) if cm_lines else None

def compute_confusion_entropy(cm):
    """Compute per-class confusion entropy: H_c = -sum(p_ij * log(p_ij)) for i != j"""
    n_classes = cm.shape[0]
    entropies = []
    
    for i in range(n_classes):
        row = cm[i, :].astype(float)
        total = row.sum()
        if total == 0:
            entropies.append(0)
            continue
        probs = row / total
        # Entropy excluding self (diagonal)
        off_diag_probs = np.concatenate([probs[:i], probs[i+1:]])
        off_diag_probs = off_diag_probs[off_diag_probs > 0]  # Remove zeros
        if len(off_diag_probs) == 0:
            entropies.append(0)
        else:
            entropy = -np.sum(off_diag_probs * np.log2(off_diag_probs))
            entropies.append(entropy)
    
    return entropies

# Process the best model metrics file
metrics_file = 'results_ablation/vit_all_augFalse_seed42_metrics.txt'
if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    
    # Parse class names
    class_names = ['Alig', 'Bessra', 'Kenta', 'Kentichi', 'Nour Dry', 'Nour Dry Tr.', 
                   'Nour Oily', 'Nour Oily Tr.', 'Nour S-Dry', 'Nour S-Dry Tr.', 'Nour S-Oily']
    
    # Find confusion matrix
    cm = None
    for i, line in enumerate(lines):
        if 'Confusion Matrix' in line:
            cm_lines = []
            for j in range(i+1, min(i+12, len(lines))):
                row_str = lines[j].strip()
                if row_str.startswith('['):
                    row = [int(x) for x in row_str.strip('[]').split() if x.strip('[]').isdigit()]
                    cm_lines.append(row)
            if cm_lines:
                cm = np.array(cm_lines)
            break
    
    if cm is not None:
        entropies = compute_confusion_entropy(cm)
        
        print("--- Per-Class Confusion Entropy ---")
        for name, ent in zip(class_names[:len(entropies)], entropies):
            print(f"{name}: H = {ent:.3f}")
        print(f"\nMean Entropy: {np.mean(entropies):.3f}")
        print(f"Max Entropy (Most Confused): {class_names[np.argmax(entropies)]}")
        print(f"Min Entropy (Least Confused): {class_names[np.argmin(entropies)]}")
        
        # Generate LaTeX
        print("\n--- LaTeX Snippet ---")
        print(f"Per-class confusion entropy analysis reveals that:")
        print(f"- Most confused class: {class_names[np.argmax(entropies)]} (H = {max(entropies):.2f})")
        print(f"- Least confused class: {class_names[np.argmin(entropies)]} (H = {min(entropies):.2f})")
        print(f"- Mean entropy: {np.mean(entropies):.2f}")
    else:
        print("Could not parse confusion matrix")
else:
    print(f"File not found: {metrics_file}")
