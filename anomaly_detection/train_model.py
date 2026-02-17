# train_model.py

import pandas as pd
import pyarrow.parquet as pq
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    auc
)

from nmf2 import (
    build_feature_matrix,
    train_nmf_detector,
    nmf_score,
    pick_threshold_by_labels
)



# Load parquet

def load_parquet(path):
    table = pq.read_table(path)
    return table.to_pandas()


if __name__ == "__main__":

    print("Loading master dataset...")
    df = load_parquet("master_dataset.parquet")

    print("Total shape:", df.shape)
    print(df["is_anomaly"].value_counts())

    
    # Separate good & anomaly
   
    df_good = df[df["is_anomaly"] == 0]
    df_anom = df[df["is_anomaly"] == 1]

    print("Good samples:", len(df_good))
    print("Anomaly samples:", len(df_anom))

   
    # GOOD split: 60/20/20
    
    good_train, good_temp = train_test_split(
        df_good,
        test_size=0.4,
        random_state=42
    )

    good_val, good_test = train_test_split(
        good_temp,
        test_size=0.5,
        random_state=42
    )

    
    # Anomaly split: 50/50
    
    anom_val, anom_test = train_test_split(
        df_anom,
        test_size=0.5,
        random_state=42
    )

   
    # Final datasets

    train_df = good_train
    val_df = pd.concat([good_val, anom_val], ignore_index=True)
    test_df = pd.concat([good_test, anom_test], ignore_index=True)

    print("Train size:", train_df.shape)
    print("Validation size:", val_df.shape)
    print("Test size:", test_df.shape)

    
    # Build TRAIN features
    
    X_train, _, shape = build_feature_matrix(
        train_df,
        target_shape=None,
        feature_transform='log+flatten',
        per_sample_normalize=True
    )

    # Train only on normal data
    nmf_model, _ = train_nmf_detector(
        X_train,
        n_components=15,
        random_state=0
    )

    
    # VALIDATION (choose threshold)
    
    X_val, _, _ = build_feature_matrix(
        val_df,
        target_shape=shape,
        feature_transform='log+flatten',
        per_sample_normalize=True
    )

    y_val = val_df["is_anomaly"].astype(int).values
    val_scores = nmf_score(nmf_model, X_val)

    thr_info = pick_threshold_by_labels(
        val_scores,
        y_val,
        method='f1',
        plot=False
    )

    threshold = thr_info["threshold"]

    print("\nChosen threshold (from validation):", threshold)

    
    # TEST evaluation
    
    X_test, meta_test, _ = build_feature_matrix(
        test_df,
        target_shape=shape,
        feature_transform='log+flatten',
        per_sample_normalize=True
    )

    y_test = test_df["is_anomaly"].astype(int).values
    test_scores = nmf_score(nmf_model, X_test)

    y_pred = (test_scores >= threshold).astype(int)

    
    # Metrics
    
    roc_auc = roc_auc_score(y_test, test_scores)

    precision, recall, _ = precision_recall_curve(y_test, test_scores)
    pr_auc = auc(recall, precision)

    f1 = f1_score(y_test, y_pred)

   
    # Confusion Matrix
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

   
    print("\n FINAL TEST METRICS ")
    print("ROC AUC :", round(roc_auc, 4))
    print("PR  AUC :", round(pr_auc, 4))
    print("F1 Score:", round(f1, 4))
    print("\nCONFUSION MATRIX ")
    print("True Negative  (TN):", tn)
    print("False Positive (FP):", fp)
    print("False Negative (FN):", fn)
    print("True Positive  (TP):", tp)

    
    # Save predictions
    
    meta_test["score"] = test_scores
    meta_test["true_label"] = y_test
    meta_test["pred_label"] = y_pred

    meta_test.to_parquet("test_predictions.parquet", index=False)

    print("\nSaved test_predictions.parquet")

