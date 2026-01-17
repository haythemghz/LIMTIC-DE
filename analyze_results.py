<<<<<<< HEAD
import numpy as np
import os
import re

def parse_metrics_file(filepath):
    """Parses the metrics file to extract the confusion matrix."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract CM part
    cm_match = re.search(r"Confusion Matrix:\n([\s\d]+)", content)
    if not cm_match:
        raise ValueError("Confusion Matrix not found in file.")
    
    cm_str = cm_match.group(1).strip()
    cm_rows = cm_str.split('\n')
    cm = []
    for row in cm_rows:
        cm.append([int(x) for x in row.split()])
    return np.array(cm)

def analyze_taxonomy(cm, classes):
    """
    Groups errors by taxonomy: Maturity, Treatment, Variety.
    """
    # Define groupings
    # Indices based on sorted class names
    # ['deglet nour dryer', 'deglet nour oily', 'deglet nour oily treated', 'deglet nour semi-dryer', 'deglet nour semi-dryer treated', 'deglet nour semi-oily', 'deglet nour semi-oily treated', 'alig', 'bessra', 'kenta', 'kintichi']
    # Note: Case sensitive? The dataset folder names were: 'Deglet Nour dryer', etc.
    # Sorted order of folders is likely used by ImageFolder.
    
    # Let's map indices to properties
    maturity_map = {}
    treatment_map = {}
    variety_map = {}
    
    for idx, cls in enumerate(classes):
        cls_lower = cls.lower()
        
        # Variety
        if 'alig' in cls_lower: variety_map[idx] = 'Alig'
        elif 'bessra' in cls_lower: variety_map[idx] = 'Bessra'
        elif 'kenta' in cls_lower: variety_map[idx] = 'Kenta'
        elif 'kintichi' in cls_lower: variety_map[idx] = 'Kentichi' # Spelling variation? 'kintichi' in list
        else: variety_map[idx] = 'Nour'
        
        # Treatment (Only for Nour)
        if variety_map[idx] == 'Nour':
            if 'treated' in cls_lower: treatment_map[idx] = 'Treated'
            else: treatment_map[idx] = 'Untreated'
        else:
            treatment_map[idx] = 'N/A'
            
        # Maturity (Only for Nour)
        if variety_map[idx] == 'Nour':
            if 'dryer' in cls_lower: maturity_map[idx] = 'Dry'
            elif 'semi-dryer' in cls_lower: maturity_map[idx] = 'Semi-Dry'
            elif 'semi-oily' in cls_lower: maturity_map[idx] = 'Semi-Oily'
            elif 'oily' in cls_lower: maturity_map[idx] = 'Oily'
        else:
            maturity_map[idx] = 'N/A'

    print("=== Error Taxonomy Analysis ===")
    total_samples = np.sum(cm)
    total_errors = total_samples - np.trace(cm)
    print(f"Total Samples: {total_samples}")
    print(f"Total Errors: {total_errors}")
    print(f"Accuracy: {np.trace(cm)/total_samples:.4f}")
    
    # 1. Variety Confusion
    # Errors where Predicted Variety != True Variety
    var_conf = 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i == j: continue
            if variety_map[i] != variety_map[j]:
                var_conf += cm[i, j]
    
    print(f"\n[Variety Mismatch]: {var_conf} errors ({var_conf/total_errors*100:.1f}% of errors)")
    
    # 2. Treatment Confusion (Within Nour)
    # Errors where Variety is Nour, but Treatment is different
    treat_conf = 0
    for i in range(len(classes)):
        if variety_map[i] != 'Nour': continue
        for j in range(len(classes)):
            if i == j: continue
            if variety_map[j] != 'Nour': continue
            
            if treatment_map[i] != treatment_map[j]:
                treat_conf += cm[i, j]
                
    print(f"[Treatment Mismatch (Nour Only)]: {treat_conf} errors ({treat_conf/total_errors*100:.1f}% of errors)")
    
    # 3. Maturity Confusion (Within Nour)
    # Errors where Variety is Nour, Treatment is SAME (to isolate maturity), but Maturity different?
    # Or just any Maturity mismatch within Nour?
    # Usually "Fine-Grained" implies checking specific confusing pairs.
    # Let's count instances where Maturity is different (ignoring treatment)
    mat_conf = 0
    for i in range(len(classes)):
        if variety_map[i] != 'Nour': continue
        for j in range(len(classes)):
            if i == j: continue
            if variety_map[j] != 'Nour': continue
            
            if maturity_map[i] != maturity_map[j]:
                mat_conf += cm[i, j]

    print(f"[Maturity Mismatch (Nour Only)]: {mat_conf} errors ({mat_conf/total_errors*100:.1f}% of errors)")
    
    return var_conf/total_errors*100, treat_conf/total_errors*100, mat_conf/total_errors*100

def generate_tikz_chart(var_p, treat_p, mat_p):
    """Generates TikZ code for a bar chart."""
    tikz = "\\begin{tikzpicture}\n"
    tikz += "\\begin{axis}[\n"
    tikz += "    ybar,\n"
    tikz += "    enlargelimits=0.15,\n"
    tikz += "    ylabel={Percentage of Errors (\\%)},\n"
    tikz += "    symbolic x coords={Variety, Treatment, Maturity},\n"
    tikz += "    xtick=data,\n"
    tikz += "    nodes near coords,\n"
    tikz += "    nodes near coords align={vertical},\n"
    tikz += "    ]\n"
    tikz += f"\\addplot coordinates {{(Variety,{var_p:.1f}) (Treatment,{treat_p:.1f}) (Maturity,{mat_p:.1f})}};\n"
    tikz += "\\end{axis}\n"
    tikz += "\\end{tikzpicture}"
    return tikz
    
def generate_latex_table(metrics_dict):
    """
    Generates LaTeX code for the ablation results table.
    metrics_dict: { 'Full Views Only': (acc, f1), ... }
    """
    latex = "\\begin{tabular}{lcc}\n"
    latex += "    \\toprule\n"
    latex += "    \\textbf{Configuration} & \\textbf{Accuracy (\\%)} & \\textbf{F1-Score} \\\\\n"
    latex += "    \\midrule\n"
    for config, (acc, f1) in metrics_dict.items():
        latex += f"    {config} & {acc*100:.1f} & {f1:.4f} \\\\\n"
    latex += "    \\bottomrule\n"
    latex += "\\end{tabular}"
    return latex

if __name__ == '__main__':
    # Default classes from previous run log
    classes = ['Deglet Nour dryer', 'Deglet Nour oily', 'Deglet Nour oily treated', 'Deglet Nour semi-dryer', 'Deglet Nour semi-dryer treated', 'Deglet Nour semi-oily', 'Deglet Nour semi-oily treated', 'alig', 'bessra', 'kenta', 'kintichi']
    
    # Example usage:
    # cm = np.array(...) # Load from file
    # analyze_taxonomy(cm, classes)
    
    results_dir = 'results_ablation'
    # Target the All Views (long-run) metrics for the paper's main taxonomy
    all_views_file = os.path.join(results_dir, 'vit_all_augFalse_seed42_metrics.txt')
    if os.path.exists(all_views_file):
        print(f"Analyzing {all_views_file}...")
        cm = parse_metrics_file(all_views_file)
        v, t, m = analyze_taxonomy(cm, classes)
        print("\n--- TikZ Chart for Error Taxonomy ---")
        print(generate_tikz_chart(v, t, m))
    else:
        print(f"File {all_views_file} not found.")
=======
import numpy as np
import os
import re

def parse_metrics_file(filepath):
    """Parses the metrics file to extract the confusion matrix."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract CM part
    cm_match = re.search(r"Confusion Matrix:\n([\s\d]+)", content)
    if not cm_match:
        raise ValueError("Confusion Matrix not found in file.")
    
    cm_str = cm_match.group(1).strip()
    cm_rows = cm_str.split('\n')
    cm = []
    for row in cm_rows:
        cm.append([int(x) for x in row.split()])
    return np.array(cm)

def analyze_taxonomy(cm, classes):
    """
    Groups errors by taxonomy: Maturity, Treatment, Variety.
    """
    # Define groupings
    # Indices based on sorted class names
    # ['deglet nour dryer', 'deglet nour oily', 'deglet nour oily treated', 'deglet nour semi-dryer', 'deglet nour semi-dryer treated', 'deglet nour semi-oily', 'deglet nour semi-oily treated', 'alig', 'bessra', 'kenta', 'kintichi']
    # Note: Case sensitive? The dataset folder names were: 'Deglet Nour dryer', etc.
    # Sorted order of folders is likely used by ImageFolder.
    
    # Let's map indices to properties
    maturity_map = {}
    treatment_map = {}
    variety_map = {}
    
    for idx, cls in enumerate(classes):
        cls_lower = cls.lower()
        
        # Variety
        if 'alig' in cls_lower: variety_map[idx] = 'Alig'
        elif 'bessra' in cls_lower: variety_map[idx] = 'Bessra'
        elif 'kenta' in cls_lower: variety_map[idx] = 'Kenta'
        elif 'kintichi' in cls_lower: variety_map[idx] = 'Kentichi' # Spelling variation? 'kintichi' in list
        else: variety_map[idx] = 'Nour'
        
        # Treatment (Only for Nour)
        if variety_map[idx] == 'Nour':
            if 'treated' in cls_lower: treatment_map[idx] = 'Treated'
            else: treatment_map[idx] = 'Untreated'
        else:
            treatment_map[idx] = 'N/A'
            
        # Maturity (Only for Nour)
        if variety_map[idx] == 'Nour':
            if 'dryer' in cls_lower: maturity_map[idx] = 'Dry'
            elif 'semi-dryer' in cls_lower: maturity_map[idx] = 'Semi-Dry'
            elif 'semi-oily' in cls_lower: maturity_map[idx] = 'Semi-Oily'
            elif 'oily' in cls_lower: maturity_map[idx] = 'Oily'
        else:
            maturity_map[idx] = 'N/A'

    print("=== Error Taxonomy Analysis ===")
    total_samples = np.sum(cm)
    total_errors = total_samples - np.trace(cm)
    print(f"Total Samples: {total_samples}")
    print(f"Total Errors: {total_errors}")
    print(f"Accuracy: {np.trace(cm)/total_samples:.4f}")
    
    # 1. Variety Confusion
    # Errors where Predicted Variety != True Variety
    var_conf = 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i == j: continue
            if variety_map[i] != variety_map[j]:
                var_conf += cm[i, j]
    
    print(f"\n[Variety Mismatch]: {var_conf} errors ({var_conf/total_errors*100:.1f}% of errors)")
    
    # 2. Treatment Confusion (Within Nour)
    # Errors where Variety is Nour, but Treatment is different
    treat_conf = 0
    for i in range(len(classes)):
        if variety_map[i] != 'Nour': continue
        for j in range(len(classes)):
            if i == j: continue
            if variety_map[j] != 'Nour': continue
            
            if treatment_map[i] != treatment_map[j]:
                treat_conf += cm[i, j]
                
    print(f"[Treatment Mismatch (Nour Only)]: {treat_conf} errors ({treat_conf/total_errors*100:.1f}% of errors)")
    
    # 3. Maturity Confusion (Within Nour)
    # Errors where Variety is Nour, Treatment is SAME (to isolate maturity), but Maturity different?
    # Or just any Maturity mismatch within Nour?
    # Usually "Fine-Grained" implies checking specific confusing pairs.
    # Let's count instances where Maturity is different (ignoring treatment)
    mat_conf = 0
    for i in range(len(classes)):
        if variety_map[i] != 'Nour': continue
        for j in range(len(classes)):
            if i == j: continue
            if variety_map[j] != 'Nour': continue
            
            if maturity_map[i] != maturity_map[j]:
                mat_conf += cm[i, j]

    print(f"[Maturity Mismatch (Nour Only)]: {mat_conf} errors ({mat_conf/total_errors*100:.1f}% of errors)")
    
    return var_conf/total_errors*100, treat_conf/total_errors*100, mat_conf/total_errors*100

def generate_tikz_chart(var_p, treat_p, mat_p):
    """Generates TikZ code for a bar chart."""
    tikz = "\\begin{tikzpicture}\n"
    tikz += "\\begin{axis}[\n"
    tikz += "    ybar,\n"
    tikz += "    enlargelimits=0.15,\n"
    tikz += "    ylabel={Percentage of Errors (\\%)},\n"
    tikz += "    symbolic x coords={Variety, Treatment, Maturity},\n"
    tikz += "    xtick=data,\n"
    tikz += "    nodes near coords,\n"
    tikz += "    nodes near coords align={vertical},\n"
    tikz += "    ]\n"
    tikz += f"\\addplot coordinates {{(Variety,{var_p:.1f}) (Treatment,{treat_p:.1f}) (Maturity,{mat_p:.1f})}};\n"
    tikz += "\\end{axis}\n"
    tikz += "\\end{tikzpicture}"
    return tikz
    
def generate_latex_table(metrics_dict):
    """
    Generates LaTeX code for the ablation results table.
    metrics_dict: { 'Full Views Only': (acc, f1), ... }
    """
    latex = "\\begin{tabular}{lcc}\n"
    latex += "    \\toprule\n"
    latex += "    \\textbf{Configuration} & \\textbf{Accuracy (\\%)} & \\textbf{F1-Score} \\\\\n"
    latex += "    \\midrule\n"
    for config, (acc, f1) in metrics_dict.items():
        latex += f"    {config} & {acc*100:.1f} & {f1:.4f} \\\\\n"
    latex += "    \\bottomrule\n"
    latex += "\\end{tabular}"
    return latex

if __name__ == '__main__':
    # Default classes from previous run log
    classes = ['Deglet Nour dryer', 'Deglet Nour oily', 'Deglet Nour oily treated', 'Deglet Nour semi-dryer', 'Deglet Nour semi-dryer treated', 'Deglet Nour semi-oily', 'Deglet Nour semi-oily treated', 'alig', 'bessra', 'kenta', 'kintichi']
    
    # Example usage:
    # cm = np.array(...) # Load from file
    # analyze_taxonomy(cm, classes)
    
    results_dir = 'results_ablation'
    # Target the All Views (long-run) metrics for the paper's main taxonomy
    all_views_file = os.path.join(results_dir, 'vit_all_augFalse_seed42_metrics.txt')
    if os.path.exists(all_views_file):
        print(f"Analyzing {all_views_file}...")
        cm = parse_metrics_file(all_views_file)
        v, t, m = analyze_taxonomy(cm, classes)
        print("\n--- TikZ Chart for Error Taxonomy ---")
        print(generate_tikz_chart(v, t, m))
    else:
        print(f"File {all_views_file} not found.")
>>>>>>> origin/main
