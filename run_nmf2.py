# run_nmf.py
import pyarrow.parquet as pq
import numpy as np
import sys
from nmf import *

# Load parquet
table = pq.read_table(sys.argv[1])
df = table.to_pandas()

# 1) Build features once (unsupervised)
X_all, meta_all, shape = build_feature_matrix(
    df,
    target_shape=None,
    feature_transform='log+flatten',
)

# 2) Train a base NMF on all data to get reconstruction errors
base_n_components = 10
base_nmf, _ = train_nmf_detector(
    X_all,
    n_components=base_n_components,
    alpha=0.0,
    l1_ratio=0.0,
    random_state=0
)
all_scores = nmf_score(base_nmf, X_all)

# 3) Define pseudo-labels: top frac_anom as anomalies
frac_anom = 0.05  # 5% highest error as anomalies
threshold_pseudo = np.quantile(all_scores, 1.0 - frac_anom)
pseudo_labels = (all_scores >= threshold_pseudo).astype(int)

df["is_anomaly"] = pseudo_labels

print("Pseudo labels created: anomaly fraction =",
      pseudo_labels.mean())

result = fit_and_evaluate_nmf(
    df,
    label_col='is_anomaly',
    n_components=15,          # pick one value
    test_fraction=0.2,
    threshold_method='f1',    # you can keep f1 if implemented
    random_state=0,
    target_shape=shape
)
# reuse inferred shape


print("Final best metrics:", result["metrics"])

meta = result['meta_test']
print(meta.head())

# Optional: save the detector
# save_detector(result['nmf'], result['shape'], path='nmf_detector.joblib')
