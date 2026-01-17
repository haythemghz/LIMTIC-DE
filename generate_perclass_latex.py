import pandas as pd

# Read the CSV
df = pd.read_csv('view_comparison.csv', index_col=0)

# Generate LaTeX table
print("\\begin{table}[ht]")
print("    \\centering")
print("    \\caption{Per-class accuracy comparison across viewpoint configurations.}")
print("    \\label{tab:per_class_view}")
print("    \\resizebox{\\textwidth}{!}{")
print("    \\begin{tabular}{lccc}")
print("        \\toprule")
print("        \\textbf{Class} & \\textbf{Full View (\\%)} & \\textbf{Side View (\\%)} & \\textbf{$\\Delta$ (Side - Full)} \\\\")
print("        \\midrule")

for idx, row in df.iterrows():
    full = row['Full View Only'] * 100
    side = row['Side View Only'] * 100
    delta = row['Delta (Side - Full)'] * 100
    delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
    print(f"        {idx} & {full:.1f} & {side:.1f} & {delta_str} \\\\")

print("        \\midrule")
print(f"        \\textbf{{Average}} & \\textbf{{{df['Full View Only'].mean()*100:.1f}}} & \\textbf{{{df['Side View Only'].mean()*100:.1f}}} & \\textbf{{{df['Delta (Side - Full)'].mean()*100:+.1f}}} \\\\")
print("        \\bottomrule")
print("    \\end{tabular}")
print("    }")
print("\\end{table}")
