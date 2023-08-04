import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("summary.tsv", sep = "\t")
df = df.drop(["peptide_charge", "group"], axis = 1)
correlation_matrix = df.corr()

# Reorder rows and columns using hierarchical clustering
corr_matrix = np.array(correlation_matrix)
order = np.argsort(np.sum(corr_matrix, axis=0))[::-1]
corr_matrix = corr_matrix[:, order][order, :]

# Increase the figure size and adjust cell size
plt.figure(figsize=(15, 12))
sns.set(font_scale=1.2)  # Adjust the font scale globally for better readability

# Set font properties for the numbers inside the cells
annot_font = {'size': 12, 'weight': 'normal'}
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5,
            mask=np.triu(corr_matrix), xticklabels=correlation_matrix.columns[order],
            yticklabels=correlation_matrix.columns[order], square=True)

# Adjust the x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotates x-axis labels and aligns them to the right

# Adjust the font size for y-axis labels
plt.yticks(fontsize=12)

plt.title("Correlation")

# Save the heatmap as a large PNG file
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()
