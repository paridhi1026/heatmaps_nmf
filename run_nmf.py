import pyarrow.parquet as pq
import numpy as np
import sys
from nmf import *

# Load parquet
table = pq.read_table(sys.argv[1])
df = table.to_pandas()


# Creating artificial anomaly labels for testing, otherwise training will not work
#Note - this is artificial, we have to import the anamolies for known issues(todo)
df["is_anomaly"] = False
df.loc[df.sample(frac=0.05, random_state=0).index, "is_anomaly"] = True

# check columns
print("Columns:", df.columns)


# Run NMF
result = fit_and_evaluate_nmf(df, label_col='is_anomaly', n_components=15, test_fraction=0.2)
print(result['metrics'])

meta = result['meta_test']
meta["score"] = result["test_scores"]
meta["true_label"] = result["y_test"]
meta["pred_label"] = (meta["score"] >= result["threshold"]).astype(int)

print(meta.head())


#to save the results
#save_detector(result['nmf'], result['shape'], path='nmf_detector.joblib')

