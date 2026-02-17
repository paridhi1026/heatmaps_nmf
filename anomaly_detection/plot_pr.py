import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# Load predictions
df = pd.read_parquet("test_predictions.parquet")

y_true = df["true_label"].values
scores = df["score"].values

# Compute Precision-Recall
precision, recall, thresholds = precision_recall_curve(y_true, scores)
pr_auc = auc(recall, precision)

# Baseline (random classifier)
baseline = y_true.mean()

# Plot
plt.figure()
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
plt.axhline(y=baseline, linestyle="--", label="Random baseline")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)

plt.savefig("pr_curve.png", dpi=150, bbox_inches="tight")

print("PR curve saved as pr_curve.png")
print("PR AUC =", pr_auc)
print("Baseline (anomaly rate) =", baseline)
