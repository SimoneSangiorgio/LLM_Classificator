import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Dati della matrice di confusione
conf_matrix = np.array([
    [63, 3, 6],   # cultural agnostic
    [5, 57, 14],   # cultural exclusive
    [18, 14, 75]   # cultural representative
])

# Etichette
labels = ['cultural agnostic', 'cultural exclusive', 'cultural representative']

# Creazione della heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Matrice di Confusione')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()