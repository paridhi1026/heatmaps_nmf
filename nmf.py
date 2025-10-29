# Requires: numpy, pandas, matplotlib, scikit-learn, joblib
# pip install numpy pandas matplotlib scikit-learn joblib

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, auc
import joblib
import matplotlib.pyplot as plt

# ---------- Utilities ----------

def ensure_2d_array(z):
    """Make sure z is a 2D numpy float array."""
    z = np.stack(z, axis=0) if isinstance(z, (list, np.ndarray)) and np.array(z).ndim == 1 else np.array(z)
    return z.astype(float)

def resize_2d(z, new_shape):
    """
    Resize 2D array z to new_shape (nx_new, ny_new) using simple bilinear interpolation.
    This avoids extra dependencies (no skimage). Works reasonably for histogram grids.
    """
    nx_new, ny_new = new_shape
    nx, ny = z.shape
    if (nx, ny) == (nx_new, ny_new):
        return z.copy()

    # Coordinates of original and new grid centers
    x_orig = np.linspace(0, 1, nx)
    y_orig = np.linspace(0, 1, ny)
    x_new = np.linspace(0, 1, nx_new)
    y_new = np.linspace(0, 1, ny_new)

    # First interpolate along y for each x
    tmp = np.empty((nx, ny_new), dtype=float)
    for i in range(nx):
        tmp[i, :] = np.interp(y_new, y_orig, z[i, :])

    # Then interpolate along x for each new y
    out = np.empty((nx_new, ny_new), dtype=float)
    for j in range(ny_new):
        out[:, j] = np.interp(x_new, x_orig, tmp[:, j])

    return out

# ---------- Build feature matrix ----------

def build_feature_matrix(df,
                         target_shape=None,
                         force_nonnegative=True,
                         feature_transform='flatten'):
    """
    From DataFrame rows (each row has 'data' as array-of-arrays), build matrix X (n_samples x n_features).

    Parameters
    ----------
    df : pandas.DataFrame
        Rows to use. Must include columns: 'run_number','ls_number','data','x_min','x_max','y_min','y_max'.
    target_shape : tuple or None
        If None, infer the most common shape among rows and use that. Otherwise (nx, ny) to resize all arrays to.
    force_nonnegative : bool
        Make data nonnegative by shifting if needed (NMF requires nonnegative).
    feature_transform : 'flatten' or 'log+flatten'
        Flatten each 2D histogram to produce feature vectors. 'log+flatten' applies log1p before flattening.
    Returns
    -------
    X : np.ndarray (n_samples, n_features)
    meta : pandas.DataFrame (same rows as df but index aligned) -- useful for mapping results back
    shape : (nx, ny) the per-sample 2D grid shape used
    """
    # keep index & metadata
    rows = df.copy().reset_index(drop=True)
    shapes = [ensure_2d_array(r["data"]).shape for _, r in rows.iterrows()]

    if target_shape is None:
        # pick the most common shape
        from collections import Counter
        common_shape = Counter(shapes).most_common(1)[0][0]
        target_shape = common_shape

    nx, ny = target_shape

    X_list = []
    for _, r in rows.iterrows():
        z = ensure_2d_array(r["data"])
        if z.shape != (nx, ny):
            z = resize_2d(z, (nx, ny))
        # Optionally shift to nonnegative
        if force_nonnegative and z.min() < 0:
            z = z - z.min()
        if feature_transform == 'log+flatten':
            v = np.log1p(z).ravel()
        else:
            v = z.ravel()
        X_list.append(v)

    X = np.vstack(X_list)
    return X, rows, (nx, ny)

# ---------- Train NMF detector ----------

def train_nmf_detector(X_train,
                       n_components=10,
                       init='nndsvda',
                       alpha=0.0,
                       l1_ratio=0.0,
                       random_state=0,
                       max_iter=500):
    """
    Fit NMF on X_train (n_samples x n_features). Returns fitted NMF and reconstruction errors for train set.
    """
    # X_train must be nonnegative
    assert (X_train >= -1e-9).all(), "NMF requires nonnegative input; shift or clip values prior."

    nmf = NMF(n_components=n_components,
              init=init,
              alpha_W=alpha,
              alpha_H=alpha,
              l1_ratio=l1_ratio,
              random_state=random_state,
              max_iter=max_iter)
    W = nmf.fit_transform(X_train)   # (n_samples, n_components)
    H = nmf.components_              # (n_components, n_features)
    X_recon = np.dot(W, H)
    # Reconstruction error per sample (MSE)
    rec_err = np.mean((X_train - X_recon)**2, axis=1)
    return nmf, rec_err

# ---------- Scoring new samples ----------

def nmf_score(nmf, X):
    """Compute reconstruction error (MSE per sample) for X using fitted nmf"""
    W = nmf.transform(X)
    H = nmf.components_
    X_recon = np.dot(W, H)
    rec_err = np.mean((X - X_recon)**2, axis=1)
    return rec_err

# ---------- Threshold selection and evaluation ----------

def pick_threshold_by_labels(scores, labels, method='roc', plot=False):
    """
    Given anomaly scores (higher => more anomalous) and binary labels (1=anomaly), pick threshold.
    method: 'roc' (maximize tpr-fpr, use Youden), 'f1' (maximize f1), 'percentile' (95th percentile of normal scores).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if method == 'roc':
        fpr, tpr, thr = roc_curve(labels, scores)
        youden_idx = np.argmax(tpr - fpr)
        thr_opt = thr[youden_idx]
        result = {'threshold': thr_opt, 'fpr': fpr[youden_idx], 'tpr': tpr[youden_idx]}
    elif method == 'f1':
        prec, rec, thr = precision_recall_curve(labels, scores)
        # precision_recall_curve doesn't give thresholds for last point; compute f1 for thresholds
        # derive thresholds from sorted unique scores
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
        # threshold based on normal (label==0) scores
        thr = np.percentile(scores[labels == 0], 95)
        result = {'threshold': float(thr)}
    else:
        raise ValueError("Unknown method")
    if plot:
        plt.figure()
        plt.hist(scores[labels==0], bins=100, alpha=0.6, label='normal')
        plt.hist(scores[labels==1], bins=100, alpha=0.6, label='anomaly')
        plt.axvline(result['threshold'], color='k', linestyle='--', label=f"thr {result['threshold']:.3g}")
        plt.legend()
        plt.xlabel('reconstruction error')
        plt.show()
    return result

# ---------- End-to-end helper ----------

def fit_and_evaluate_nmf(df, label_col='is_anomaly', n_components=10, test_fraction=0.2,
                         threshold_method='roc', random_state=0, target_shape=None):
    """
    Fit NMF anomaly detector using labeled data in df. Returns model, threshold, and metrics.
    df must contain a column label_col with 0 (normal) / 1 (anomaly) for rows to use.
    """
    # Only keep rows that have labels
    df_labeled = df.dropna(subset=[label_col]).reset_index(drop=True)
    if df_labeled.empty:
        raise ValueError("No labeled rows found in df")

    # Build features (use consistent shape across dataset)
    X, meta, shape = build_feature_matrix(df_labeled, target_shape=target_shape)
    labels = df_labeled[label_col].astype(int).values

    # Optionally split into train/test so we can evaluate generalization
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, labels, meta, test_size=test_fraction, stratify=labels, random_state=random_state)

    # Ideally train NMF on only the normal samples (semi-supervised)
    X_train_normal = X_train[y_train == 0]
    if len(X_train_normal) < max(10, n_components*2):
        # fallback: use full training data if too few normal examples
        X_train_fit = X_train
    else:
        X_train_fit = X_train_normal

    nmf, train_err = train_nmf_detector(X_train_fit, n_components=n_components, random_state=random_state)

    # Score test set
    test_scores = nmf_score(nmf, X_test)

    # Pick threshold using the labeled test set (or use train if preferred)
    thr_info = pick_threshold_by_labels(test_scores, y_test, method=threshold_method, plot=True)
    threshold = thr_info['threshold']

    # Predictions
    y_pred = (test_scores >= threshold).astype(int)

    # Metrics
    auc_roc = roc_auc_score(y_test, test_scores)
    prec, rec, _ = precision_recall_curve(y_test, test_scores)
    aupr = auc(rec, prec)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        'roc_auc': float(auc_roc),
        'pr_auc': float(aupr),
        'f1': float(f1),
        'threshold': float(threshold),
        **{k: v for k, v in thr_info.items() if k != 'threshold'}
    }

    return {
        'nmf': nmf,
        'threshold': threshold,
        'metrics': metrics,
        'meta_test': meta_test,
        'y_test': y_test,
        'test_scores': test_scores,
        'shape': shape
    }

# ---------- Save / load ----------

def save_detector(nmf, target_shape, scaler=None, path="nmf_detector.joblib"):
    joblib.dump({'nmf': nmf, 'shape': target_shape, 'scaler': scaler}, path)
    print("Saved detector to", path)

def load_detector(path="nmf_detector.joblib"):
    d = joblib.load(path)
    return d['nmf'], d['shape'], d.get('scaler', None)

# ---------- Example usage ----------

# Assuming you have `df_all` containing many runs/LS with a column 'is_anomaly' where experts labeled anomalies.
# X_all, meta_all, shape = build_feature_matrix(df_all, target_shape=None)
# result = fit_and_evaluate_nmf(df_all, label_col='is_anomaly', n_components=15, test_fraction=0.2)
# print(result['metrics'])
# save_detector(result['nmf'], result['shape'], path='nmf_detector.joblib')
