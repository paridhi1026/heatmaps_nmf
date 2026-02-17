# nmf.py
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, auc
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# ---------- Utilities (unchanged) ----------

def ensure_2d_array(z):
    z = np.stack(z, axis=0) if isinstance(z, (list, np.ndarray)) and np.array(z).ndim == 1 else np.array(z)
    return z.astype(float)

def resize_2d(z, new_shape):
    nx_new, ny_new = new_shape
    nx, ny = z.shape
    if (nx, ny) == (nx_new, ny_new):
        return z.copy()
    x_orig = np.linspace(0, 1, nx)
    y_orig = np.linspace(0, 1, ny)
    x_new = np.linspace(0, 1, nx_new)
    y_new = np.linspace(0, 1, ny_new)
    tmp = np.empty((nx, ny_new), dtype=float)
    for i in range(nx):
        tmp[i, :] = np.interp(y_new, y_orig, z[i, :])
    out = np.empty((nx_new, ny_new), dtype=float)
    for j in range(ny_new):
        out[:, j] = np.interp(x_new, x_orig, tmp[:, j])
    return out

"""
# ---------- Build feature matrix (enhanced) ----------
def build_feature_matrix(df,
                         target_shape=None,
                         force_nonnegative=True,
                         feature_transform='log+flatten',
                         per_sample_normalize=True, debug_index=None):
   
    rows = df.copy().reset_index(drop=True)
    shapes = [ensure_2d_array(r["data"]).shape for _, r in rows.iterrows()]

    if target_shape is None:
        from collections import Counter
        common_shape = Counter(shapes).most_common(1)[0][0]
        target_shape = common_shape

    nx, ny = target_shape
    X_list = []
    for idx, r in rows.iterrows():
        z = ensure_2d_array(r["data"])

    if z.shape != (nx, ny):
        z = resize_2d(z, (nx, ny))

 
    if force_nonnegative and z.min() < 0:
        z = z - z.min()

    #  normalize
    if per_sample_normalize:
        s = z.sum()
        if s > 0:
            z = z / s

    # print one ls
    if debug_index is not None and idx == debug_index:
        print(f"\nAfter normalization — lumisection index {idx}")
        print(z[:5, :5])
        print("Sum after normalization:", z.sum())

    # feature transform
    if feature_transform == 'log+flatten':
        v = np.log1p(z).ravel()
    else:
        v = z.ravel()

    X_list.append(v)

#2
    for _, r in rows.iterrows():
        z = ensure_2d_array(r["data"])
        if z.shape != (nx, ny):
            z = resize_2d(z, (nx, ny))

        # per-sample normalization
        if per_sample_normalize:
            s = z.sum()
            if s > 0:
                z = z / s

        # shift to nonnegative if needed
        if force_nonnegative and z.min() < 0:
            z = z - z.min()

       
        if feature_transform == 'log+flatten':
            v = np.log1p(z).ravel()
        else:
            v = z.ravel()

        X_list.append(v)
    X = np.vstack(X_list)
    return X, rows, (nx, ny)
#2
"""

def build_feature_matrix(df,
                         target_shape=None,
                         force_nonnegative=True,
                         feature_transform='log+flatten',
                         per_sample_normalize=True,
                         debug_index=None):

    rows = df.copy().reset_index(drop=True)

    shapes = [ensure_2d_array(r["data"]).shape for _, r in rows.iterrows()]

    if target_shape is None:
        from collections import Counter
        target_shape = Counter(shapes).most_common(1)[0][0]

    nx, ny = target_shape
    X_list = []

    for idx, r in rows.iterrows():

        z = ensure_2d_array(r["data"])

        if z.shape != (nx, ny):
            z = resize_2d(z, (nx, ny))

        if force_nonnegative and z.min() < 0:
            z = z - z.min()

        if per_sample_normalize:
            s = z.sum()
            if s > 0:
                z = z / s

        if debug_index is not None and idx == debug_index:
            print(f"\nAfter normalization — lumisection index {idx}")
            print(z[:5, :5])
            print("Sum after normalization:", z.sum())

        if feature_transform == 'log+flatten':
            v = np.log1p(z).ravel()
        else:
            v = z.ravel()

        X_list.append(v)

    X = np.vstack(X_list)

    return X, rows, (nx, ny)




# ---------- NMF training & scoring (unchanged logic) ----------

def train_nmf_detector(X_train,
                       n_components=10,
                       init='nndsvda',
                       alpha=0.0,
                       l1_ratio=0.0,
                       random_state=0,
                       max_iter=500):
    assert (X_train >= -1e-9).all(), "NMF requires nonnegative input."
    nmf = NMF(
        n_components=n_components,
        init=init,
        alpha_W=alpha,
        alpha_H=alpha,
        l1_ratio=l1_ratio,
        random_state=random_state,
        max_iter=max_iter
    )
    W = nmf.fit_transform(X_train)
    H = nmf.components_
    X_recon = np.dot(W, H)
    rec_err = np.mean((X_train - X_recon) ** 2, axis=1)
    return nmf, rec_err

def nmf_score(nmf, X):
    W = nmf.transform(X)
    H = nmf.components_
    X_recon = np.dot(W, H)
    rec_err = np.mean((X - X_recon) ** 2, axis=1)
    return rec_err

# ---------- Threshold selection (unchanged) ----------

def pick_threshold_by_labels(scores, labels, method='roc', plot=False):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if method == 'roc':
        fpr, tpr, thr = roc_curve(labels, scores)
        youden_idx = np.argmax(tpr - fpr)
        thr_opt = thr[youden_idx]
        result = {'threshold': thr_opt, 'fpr': fpr[youden_idx], 'tpr': tpr[youden_idx]}
    elif method == 'f1':
        thresholds = np.unique(scores)
        best_f1 = -1
        best_thr = thresholds[0]
        for t in thresholds:
            preds = (scores >= t).astype(int)
            f1 = f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = t
        result = {'threshold': best_thr, 'f1': best_f1}
    elif method == 'percentile':
        thr = np.percentile(scores[labels == 0], 95)
        result = {'threshold': float(thr)}
    else:
        raise ValueError("Unknown method")
    if plot:
        plt.figure()
        plt.hist(scores[labels == 0], bins=100, alpha=0.6, label='normal')
        plt.hist(scores[labels == 1], bins=100, alpha=0.6, label='anomaly')
        plt.axvline(result['threshold'], color='k', linestyle='--',
                    label=f"thr {result['threshold']:.3g}")
        plt.legend()
        plt.xlabel('reconstruction error')
        plt.show()
    return result

# ---------- End-to-end helper with simple hyperparam tuning ----------

def fit_and_evaluate_nmf(df,
                         label_col='is_anomaly',
                         n_components_list=(5, 10, 15),
                         alpha_list=(0.0, 0.01),
                         l1_ratio_list=(0.0, 0.5),
                         test_fraction=0.2,
                         threshold_method='f1',
                         random_state=0,
                         target_shape=None):
    """
    Small grid search over (n_components, alpha, l1_ratio) and choose the combination
    with best F1 on the test split.
    """
    df_labeled = df.dropna(subset=[label_col]).reset_index(drop=True)
    if df_labeled.empty:
        raise ValueError("No labeled rows found in df")

    X, meta, shape = build_feature_matrix(
        df_labeled,
        target_shape=target_shape,
        feature_transform='log+flatten',
        per_sample_normalize=True
    )
    labels = df_labeled[label_col].astype(int).values

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, labels, meta, test_size=test_fraction, stratify=labels,
        random_state=random_state
    )

    best_config = None
    best_metrics = None
    best_result = None

    for n_components in n_components_list:
        for alpha in alpha_list:
            for l1_ratio in l1_ratio_list:
                # train on normal-only if possible
                X_train_normal = X_train[y_train == 0]
                if len(X_train_normal) < max(10, n_components * 2):
                    X_train_fit = X_train
                else:
                    X_train_fit = X_train_normal

                nmf, train_err = train_nmf_detector(
                    X_train_fit,
                    n_components=n_components,
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    random_state=random_state
                )

                test_scores = nmf_score(nmf, X_test)
                thr_info = pick_threshold_by_labels(
                    test_scores, y_test,
                    method=threshold_method,
                    plot=False
                )
                threshold = thr_info['threshold']
                y_pred = (test_scores >= threshold).astype(int)

                auc_roc = roc_auc_score(y_test, test_scores)
                prec, rec, _ = precision_recall_curve(y_test, test_scores)
                aupr = auc(rec, prec)
                f1 = f1_score(y_test, y_pred)
       	    
                print(f"[Config] {n_components=}, {alpha=}, {l1_ratio=}, ROC AUC={auc_roc:.4f}, F1={f1:.4f}")
                metrics = {
                    'roc_auc': float(auc_roc),
                    'pr_auc': float(aupr),
                    'f1': float(f1),
                    'threshold': float(threshold),
                    'n_components': n_components,
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    **{k: v for k, v in thr_info.items() if k != 'threshold'}
                }

                if (best_metrics is None) or (metrics['f1'] > best_metrics['f1']):
                    best_metrics = metrics
                    best_config = (n_components, alpha, l1_ratio)
                    best_result = {
                        'nmf': nmf,
                        'threshold': threshold,
                        'metrics': metrics,
                        'meta_test': meta_test.copy(),
                        'y_test': y_test.copy(),
                        'test_scores': test_scores,
                        'shape': shape
                    }

    # attach predictions to meta_test in best_result
    meta = best_result['meta_test']
    meta['score'] = best_result['test_scores']
    meta['true_label'] = best_result['y_test']
    meta['pred_label'] = (meta['score'] >= best_result['threshold']).astype(int)
    best_result['meta_test'] = meta

    print("Best NMF config:",
          "n_components=", best_config[0],
          "alpha=", best_config[1],
          "l1_ratio=", best_config[2])
    print("Best metrics:", best_metrics)

    return best_result

# ---------- Save / load (unchanged) ----------

def save_detector(nmf, target_shape, scaler=None, path="nmf_detector.joblib"):
    joblib.dump({'nmf': nmf, 'shape': target_shape, 'scaler': scaler}, path)
    print("Saved detector to", path)

def load_detector(path="nmf_detector.joblib"):
    d = joblib.load(path)
    return d['nmf'], d['shape'], d.get('scaler', None)
