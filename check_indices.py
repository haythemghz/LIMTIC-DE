import os
import re
from collections import Counter

root = 'c:/Users/Dell/Desktop/Article_Deglet_Nour/LIMTIC-DE_Dataset/LIMTIC-DE_Dataset/Train'
classes = os.listdir(root)

for cls in classes:
    cls_path = os.path.join(root, cls)
    if not os.path.isdir(cls_path): continue
    
    files = os.listdir(cls_path)
    indices = []
    for f in files:
        match = re.search(r'_(\d+)_jpg', f)
        if match:
            indices.append(int(match.group(1)))
    
    indices.sort()
    counts = Counter([idx // 6 for idx in indices])
    group_sizes = Counter(counts.values())
    print(f"Class: {cls} | Total Indices: {len(indices)} | Group Size Distribution: {group_sizes}")
