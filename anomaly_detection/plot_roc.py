import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Load predictions
df = pd.read_parquet("test_predictions.parquet")

y_true = df["true_label"].values
scores = df["score"].values

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_true, scores)
roc_auc = roc_auc_score(y_true, scores)

# Plot
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")  # random baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)

plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")

print("ROC curve saved as roc_curve.png")
print("AUC =", roc_auc)
