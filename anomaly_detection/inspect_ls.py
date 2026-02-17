import pandas as pd

# Load test predictions
df = pd.read_parquet("test_predictions.parquet")

# Pick random LS
row = df.sample(1).iloc[0]

score = row["score"]
true_label = row["true_label"]
pred_label = row["pred_label"]

# Recover threshold (minimum score among predicted anomalies)
threshold = df[df["pred_label"] == 1]["score"].min()

print("\nRANDOM LS")
print("Reconstruction Error (score):", score)
print("Threshold:", threshold)
print("True Label:", true_label)
print("Predicted Label:", pred_label)

if pred_label == 1:
    print("LS classified as: ANOMALY")
else:
    print("LS classified as: NORMAL")
