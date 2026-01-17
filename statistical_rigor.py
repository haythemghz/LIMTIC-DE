import numpy as np
import scipy.stats as stats

def compute_ci(data, level=0.95):
    n = len(data)
    m = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + level) / 2., n-1)
    return m, m-h, m+h

def cohen_d(x, y):
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    nx, ny = len(x), len(y)
    dobj = np.sqrt(((nx-1)*sx + (ny-1)*sy) / (nx+ny-2))
    return (mx - my) / dobj

def analyze_model_pair(name_a, scores_a, name_b, scores_b):
    m_a, low_a, high_a = compute_ci(scores_a)
    m_b, low_b, high_b = compute_ci(scores_b)
    d = cohen_d(scores_a, scores_b)
    
    print(f"--- {name_a} vs {name_b} ---")
    print(f"{name_a}: {m_a:.2f}% (95% CI: [{low_a:.2f}, {high_a:.2f}])")
    print(f"{name_b}: {m_b:.2f}% (95% CI: [{low_b:.2f}, {high_b:.2f}])")
    print(f"Cohen's d Effect Size: {d:.4f}")
    
    if abs(d) > 0.8:
        print("Effect Size: Large")
    elif abs(d) > 0.5:
        print("Effect Size: Medium")
    elif abs(d) > 0.2:
        print("Effect Size: Small")
    else:
        print("Effect Size: Negligible")
    print()

if __name__ == "__main__":
    # Example data (interpolated from known baselines)
    vit_scores = [96.2, 95.8, 96.5, 96.0, 96.3]
    resnet_scores = [89.0, 88.5, 89.2, 88.8, 89.5]
    efficient_scores = [93.7, 93.2, 94.0, 93.5, 93.8]
    
    analyze_model_pair("ViT-B/16", vit_scores, "ResNet50", resnet_scores)
    analyze_model_pair("ViT-B/16", vit_scores, "EfficientNet-B0", efficient_scores)
