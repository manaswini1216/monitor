# monitoring/dashboard/app.py
"""
Final Auto MLOps Monitoring Dashboard (centered uploads).
- Upload (center): models (one or more), training CSV, optional logs CSV
- Auto-detect target, detect classification/regression
- Auto-encode & align features to each model
- Multi-model comparison, SHAP integration via shap_utils.py
- Advanced visuals + drift checks
"""

import os
import sys
import io
import json
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# sklearn metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error,
    precision_recall_curve
)

st.set_page_config(page_title="Auto MLOps Monitoring — Final", layout="wide")

# Try importing shap utils (graceful fallback)
try:
    from shap_utils import compute_shap_values, shap_summary_plot_html, shap_force_plot_html
except Exception:
    compute_shap_values = None
    shap_summary_plot_html = None
    shap_force_plot_html = None

# -------------------------
# Helper utilities
# -------------------------
def load_pickle_from_fileobj(fobj):
    fobj.seek(0)
    try:
        return pickle.load(fobj)
    except Exception:
        fobj.seek(0)
        return pickle.loads(fobj.read())

def auto_detect_target_column(df: pd.DataFrame) -> str:
    candidates = ["target", "label", "y", "class", "outcome", "survived"]
    for c in candidates:
        if c in df.columns:
            return c
    # numeric small-unique candidates
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            uniq = df[c].nunique(dropna=True)
            if 2 <= uniq <= 5 and uniq / max(1, len(df)) < 0.5:
                return c
    # low-cardinality categorical
    for c in df.columns:
        uniq = df[c].nunique(dropna=True)
        if uniq <= 10 and uniq < 0.5 * len(df):
            return c
    return df.columns[-1]

def detect_problem_type(series: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    uniq = series.nunique(dropna=True)
    if uniq <= 20 and (uniq / max(1, len(series)) < 0.5):
        return "classification"
    return "regression"

def auto_encode_df(df: pd.DataFrame, max_onehot:int=30) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    encoders = {}
    parts = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        parts.append(df[numeric_cols].fillna(0))
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    for c in cat_cols:
        s = df[c].astype("object").fillna("<<MISSING>>")
        nunique = s.nunique(dropna=False)
        if nunique <= max_onehot:
            # one-hot via pandas - deterministic
            dummies = pd.get_dummies(s, prefix=c, dummy_na=False)
            parts.append(dummies)
            encoders[c] = {"type":"onehot", "categories": list(s.unique())}
        else:
            labels, uniques = pd.factorize(s)
            parts.append(pd.Series(labels, name=c))
            encoders[c] = {"type":"label", "classes": list(uniques)}
    if parts:
        enc = pd.concat(parts, axis=1)
    else:
        enc = pd.DataFrame(index=df.index)
    return enc, encoders

def align_features_to_model(X_enc: pd.DataFrame, model) -> pd.DataFrame:
    X = X_enc.copy()
    expected_names = None
    expected_count = None
    try:
        expected_names = list(model.feature_names_in_)
    except Exception:
        expected_names = None
    try:
        expected_count = int(model.n_features_in_)
    except Exception:
        expected_count = None

    if expected_names:
        # add missing
        for en in expected_names:
            if en not in X.columns:
                X[en] = 0
        # drop extras
        extras = [c for c in X.columns if c not in expected_names]
        if extras:
            X = X.drop(columns=extras)
        X = X[expected_names]
        return X
    elif expected_count is not None:
        curr = X.shape[1]
        if curr < expected_count:
            for i in range(curr, expected_count):
                X[f"missing_{i}"] = 0
        elif curr > expected_count:
            X = X.iloc[:, :expected_count]
        return X
    else:
        return X.select_dtypes(include=[np.number]).fillna(0)

def compute_classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str,Any]:
    out = {}
    try:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) == 2:
            out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
            out["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
        else:
            out["precision"] = None
            out["recall"] = None
            out["f1_score"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        if y_proba is not None and len(np.unique(y_true)) == 2:
            arr = np.array(y_proba)
            if arr.ndim==2 and arr.shape[1]>=2:
                probs = arr[:,1]
            else:
                probs = arr.ravel()
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, probs))
            except Exception:
                out["roc_auc"] = None
        else:
            out["roc_auc"] = None
    except Exception as e:
        out["error"] = str(e)
    return out

def compute_regression_metrics(y_true, y_pred) -> Dict[str,Any]:
    out = {}
    try:
        out["rmse"] = float(mean_squared_error(y_true, y_pred, squared=False))
        out["mae"] = float(mean_absolute_error(y_true, y_pred))
        out["r2"] = float(r2_score(y_true, y_pred))
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / (np.array(y_true) + 1e-8))) * 100.0
        out["mape"] = float(mape)
    except Exception as e:
        out["error"] = str(e)
    return out

def psi(expected, actual, buckets=10) -> float:
    expected = np.array(expected).ravel()
    actual = np.array(actual).ravel()
    try:
        quantiles = np.nanpercentile(expected, np.linspace(0,100,buckets+1))
        quantiles = np.unique(quantiles)
        if len(quantiles) <= 1:
            raise ValueError("Not enough quantile edges")
        exp_counts, _ = np.histogram(expected, bins=quantiles)
        act_counts, _ = np.histogram(actual, bins=quantiles)
    except Exception:
        exp_counts, _ = np.histogram(expected, bins=buckets)
        act_counts, _ = np.histogram(actual, bins=buckets)
    exp_prop = exp_counts / (exp_counts.sum() + 1e-8)
    act_prop = act_counts / (act_counts.sum() + 1e-8)
    exp_prop = np.clip(exp_prop, 1e-6, 1)
    act_prop = np.clip(act_prop, 1e-6, 1)
    return float(np.sum((exp_prop - act_prop) * np.log(exp_prop / act_prop)))

def ks_2sample(a, b):
    try:
        s,p = stats.ks_2samp(a,b)
        return float(s), float(p)
    except Exception:
        return None, None

def wasserstein(a,b):
    try:
        return float(stats.wasserstein_distance(a,b))
    except Exception:
        return None

# plotting helpers
def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(z=cm, x=labels if labels else list(range(cm.shape[1])),
                                    y=labels if labels else list(range(cm.shape[0])), colorscale="Blues"))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def plot_roc_curve(y_true, y_scores):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr,tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.3f}"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), showlegend=False))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    return fig

def plot_pr_curve(y_true, y_scores):
    from sklearn.metrics import precision_recall_curve, auc
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    area = auc(rec, prec)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"AUPR={area:.3f}"))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    return fig

# -------------------------
# UI: Centered uploads
# -------------------------
st.title("🚀 Auto MLOps Monitoring — Final")
st.markdown("Upload **models** (one or more) and a **training CSV** in the center. Prediction logs are optional. The app auto-detects problem type and runs evaluations + SHAP + drift checks.")

col_model, col_train, col_logs = st.columns([1,1,1])
with col_model:
    models_files = st.file_uploader("Upload model(s) (.pkl/.joblib) — multiple allowed", type=["pkl","pickle","joblib"], accept_multiple_files=True)
with col_train:
    train_file = st.file_uploader("Upload training CSV", type=["csv"])
with col_logs:
    logs_file = st.file_uploader("Upload prediction logs CSV (optional)", type=["csv"])

# Small control area (below uploads)
opt_col1, opt_col2, _ = st.columns([1,1,2])
with opt_col1:
    max_onehot = st.number_input("Max unique values -> One-Hot (auto)", min_value=2, max_value=200, value=30, step=1)
with opt_col2:
    shap_sample = st.number_input("SHAP sample (if enabled)", min_value=50, max_value=2000, value=200, step=50)

if not models_files or train_file is None:
    st.info("Upload at least one model and the training CSV to proceed.")
    st.stop()

# load train csv
try:
    train_file.seek(0)
    df_train = pd.read_csv(train_file)
    st.success(f"Training CSV loaded: {df_train.shape[0]} rows x {df_train.shape[1]} cols")
except Exception as e:
    st.error(f"Couldn't read training CSV: {e}")
    st.stop()

# optional logs
logs_df = None
if logs_file:
    try:
        logs_file.seek(0)
        logs_df = pd.read_csv(logs_file)
        st.success(f"Logs loaded: {logs_df.shape[0]} rows x {logs_df.shape[1]} cols")
    except Exception as e:
        st.warning(f"Couldn't read logs CSV: {e}")
        logs_df = None

# auto detect target
target_col = auto_detect_target_column(df_train)
st.write(f"**Auto-detected target:** `{target_col}`")
if target_col not in df_train.columns:
    st.error("Auto-detected target not found in training CSV columns. Aborting.")
    st.stop()

y_train = df_train[target_col]
X_train_raw = df_train.drop(columns=[target_col])

# encode training data
with st.spinner("Auto-encoding training data..."):
    X_train_enc, enc_info = auto_encode_df(X_train_raw, max_onehot=max_onehot)

# load models
loaded_models = []
for mf in models_files:
    try:
        mf.seek(0)
        obj = None
        try:
            obj = load_pickle_from_fileobj(mf)
        except Exception:
            import joblib
            mf.seek(0)
            obj = joblib.load(mf)
        loaded_models.append({"name": mf.name, "model": obj})
    except Exception as e:
        st.error(f"Failed loading model {getattr(mf,'name',str(mf))}: {e}")

if not loaded_models:
    st.error("No model loaded successfully.")
    st.stop()

# Evaluate each model
model_results = []
for item in loaded_models:
    name = item["name"]
    mod = item["model"]

    # Align encoded features to model
    try:
        X_aligned = align_features_to_model(X_train_enc, mod)
    except Exception as e:
        st.warning(f"Align failed for {name}: {e}; using encoded features as fallback.")
        X_aligned = X_train_enc.copy()

    X_aligned = X_aligned.fillna(0)

    # Predict
    y_pred = None
    y_proba = None
    pred_error = None
    try:
        y_pred = mod.predict(X_aligned)
        try:
            if hasattr(mod, "predict_proba"):
                y_proba = mod.predict_proba(X_aligned)
            elif hasattr(mod, "decision_function"):
                y_proba = mod.decision_function(X_aligned)
        except Exception:
            y_proba = None
    except Exception as e:
        pred_error = str(e)

    ptype = detect_problem_type(y_train)

    metrics = None
    if pred_error is None and y_pred is not None:
        if ptype == "classification":
            metrics = compute_classification_metrics(y_train.values, y_pred, y_proba)
        else:
            metrics = compute_regression_metrics(y_train.values, y_pred)
    else:
        metrics = {"error": pred_error}

    # feature importance
    fi_df = None
    try:
        if hasattr(mod, "feature_importances_"):
            arr = np.ravel(getattr(mod, "feature_importances_"))
            cols = list(X_aligned.columns)[:len(arr)]
            fi_df = pd.DataFrame({"feature": cols, "importance": arr[:len(cols)]}).sort_values("importance", ascending=False)
        elif hasattr(mod, "coef_"):
            arr = np.abs(np.ravel(getattr(mod, "coef_")))
            cols = list(X_aligned.columns)[:len(arr)]
            fi_df = pd.DataFrame({"feature": cols, "importance": arr[:len(cols)]}).sort_values("importance", ascending=False)
    except Exception:
        fi_df = None

    model_results.append({
        "name": name,
        "model": mod,
        "X_aligned": X_aligned,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
        "fi": fi_df,
        "pred_error": pred_error,
        "ptype": ptype
    })

# Comparison table
st.header("Model Comparison (evaluated on training CSV)")
rows = []
for r in model_results:
    m = r["metrics"] or {}
    rows.append({
        "model": r["name"],
        "error": r.get("pred_error"),
        "accuracy": m.get("accuracy"),
        "f1": m.get("f1_score"),
        "precision": m.get("precision"),
        "recall": m.get("recall"),
        "roc_auc": m.get("roc_auc"),
        "rmse": m.get("rmse"),
        "mae": m.get("mae"),
        "r2": m.get("r2")
    })
comp_df = pd.DataFrame(rows).set_index("model")
st.dataframe(comp_df)

# select model
names = [r["name"] for r in model_results]
sel = st.selectbox("Select model to inspect", names, index=0)
selected = next(r for r in model_results if r["name"] == sel)

st.subheader(f"Inspecting: {sel}")
if selected["pred_error"]:
    st.error(f"Prediction error: {selected['pred_error']}")
else:
    st.markdown("**Metrics**")
    st.json(selected["metrics"] if selected["metrics"] else {})

    # feature importance
    st.markdown("**Feature Importance (top 30)**")
    if selected["fi"] is not None and not selected["fi"].empty:
        fig = px.bar(selected["fi"].head(30), x="importance", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feature importance available.")

    # visuals
    st.markdown("**Performance Visuals**")
    if selected["ptype"] == "classification" and selected["y_pred"] is not None:
        # confusion
        try:
            labels = list(selected["model"].classes_) if hasattr(selected["model"], "classes_") else None
            cm_fig = plot_confusion_matrix(y_train, selected["y_pred"], labels=labels)
            st.plotly_chart(cm_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Confusion error: {e}")

        # ROC/PR
        if selected["y_proba"] is not None and len(np.unique(y_train)) == 2:
            try:
                arr = np.array(selected["y_proba"])
                scores = arr[:,1] if arr.ndim==2 and arr.shape[1]>=2 else arr.ravel()
                st.plotly_chart(plot_roc_curve(y_train, scores), use_container_width=True)
                st.plotly_chart(plot_pr_curve(y_train, scores), use_container_width=True)
            except Exception as e:
                st.warning(f"ROC/PR error: {e}")

        # error analysis: show false positives / false negatives (top 20)
        try:
            preds = np.array(selected["y_pred"])
            actual = np.array(y_train)
            df_errors = pd.DataFrame(X_train_raw).reset_index(drop=True)
            df_errors["actual"] = actual
            df_errors["pred"] = preds
            if len(np.unique(actual)) == 2:
                fp = df_errors[(df_errors["actual"]==0) & (df_errors["pred"]==1)]
                fn = df_errors[(df_errors["actual"]==1) & (df_errors["pred"]==0)]
                st.markdown("**False Positives (sample)**")
                st.dataframe(fp.head(20))
                st.markdown("**False Negatives (sample)**")
                st.dataframe(fn.head(20))
        except Exception as e:
            st.warning(f"Error analysis failed: {e}")

    elif selected["ptype"] == "regression" and selected["y_pred"] is not None:
        try:
            resid = np.array(y_train) - np.array(selected["y_pred"])
            st.plotly_chart(px.histogram(resid, nbins=50, title="Residuals (train)"), use_container_width=True)
            st.plotly_chart(px.scatter(x=np.array(selected["y_pred"]), y=np.array(y_train),
                                       labels={"x":"Predicted","y":"Actual"}, title="Predicted vs Actual"), use_container_width=True)
        except Exception as e:
            st.warning(f"Regression visuals error: {e}")

# SHAP integration
st.header("Explainability (SHAP) — optional")
if compute_shap_values is None:
    st.info("SHAP helpers not found or shap not installed. To enable, add shap_utils.py and install shap.")
else:
    if st.button(f"Compute SHAP for {sel} (sample={shap_sample})"):
        try:
            cache_key = f"shap_{sel}_{hash(tuple(selected['X_aligned'].columns))}_{shap_sample}"
            if cache_key in st.session_state:
                explainer, shap_vals, X_shap = st.session_state[cache_key]
            else:
                explainer, shap_vals, X_shap = compute_shap_values(selected["model"], selected["X_aligned"], nsamples=shap_sample)
                st.session_state[cache_key] = (explainer, shap_vals, X_shap)
            st.subheader("SHAP Summary")
            try:
                html = shap_summary_plot_html(shap_vals, X_shap)
                st.components.v1.html(html, height=420, scrolling=True)
            except Exception as e:
                st.warning(f"SHAP summary render failed: {e}")

            st.subheader("SHAP per-instance")
            idx = st.number_input("Index in SHAP sample", min_value=0, max_value=max(0, X_shap.shape[0]-1), value=0, step=1)
            try:
                htmlf = shap_force_plot_html(explainer, shap_vals, X_shap, idx=idx)
                st.components.v1.html(htmlf, height=420, scrolling=True)
            except Exception as e:
                st.warning(f"SHAP force render failed: {e}")
        except Exception as e:
            st.warning(f"SHAP compute error: {e}")

# Drift if logs provided
if logs_df is not None:
    st.header("Drift checks (logs vs training)")
    # detect preds and true in logs
    pred_col = next((c for c in ["prediction","pred","y_pred","yhat","model_prediction"] if c in logs_df.columns), None)
    true_col = next((c for c in ["true_label","true","label",target_col] if c in logs_df.columns), None)
    st.write(f"Detected in logs: prediction=`{pred_col}` ; true=`{true_col}`")
    # baseline features: use aligned features of selected model as baseline
    baseline_feats = list(selected["X_aligned"].columns)
    live_feats = [f for f in baseline_feats if f in logs_df.columns]
    drift_rows = []
    for f in live_feats:
        try:
            if pd.api.types.is_numeric_dtype(selected["X_aligned"][f]) and pd.api.types.is_numeric_dtype(logs_df[f]):
                exp = selected["X_aligned"][f].dropna().astype(float).values
                act = logs_df[f].dropna().astype(float).values
                if len(exp)>=2 and len(act)>=2:
                    p = psi(exp,act)
                    ks_s, ks_p = ks_2sample(exp,act)
                    w = wasserstein(exp,act)
                else:
                    p=ks_s=ks_p=w=None
                drift_rows.append({"feature":f,"type":"numeric","psi":p,"ks_stat":ks_s,"ks_p":ks_p,"wasserstein":w})
            else:
                exp_counts = selected["X_aligned"][f].astype(str).value_counts(normalize=True).to_dict()
                act_counts = logs_df[f].astype(str).value_counts(normalize=True).to_dict()
                keys = set(list(exp_counts.keys())+list(act_counts.keys()))
                maxdiff = max(abs(exp_counts.get(k,0)-act_counts.get(k,0)) for k in keys) if keys else None
                drift_rows.append({"feature":f,"type":"categorical","max_prop_diff":maxdiff,"exp":exp_counts,"act":act_counts})
        except Exception as e:
            drift_rows.append({"feature":f,"type":"error","error":str(e)})
    drift_df = pd.DataFrame(drift_rows)
    if not drift_df.empty:
        if "psi" in drift_df.columns:
            drift_df["psi"] = pd.to_numeric(drift_df["psi"], errors="coerce")
            st.dataframe(drift_df.sort_values("psi",ascending=False).fillna("N/A").reset_index(drop=True))
        else:
            st.dataframe(drift_df.fillna("N/A"))
    else:
        st.info("No overlapping features between selected model baseline and logs for drift checks.")
    # live metrics if true+pred present
    if true_col and pred_col:
        st.subheader("Live performance from logs")
        logs_ok = logs_df.dropna(subset=[true_col,pred_col])
        if len(logs_ok) >= 5:
            window = st.slider("Rolling window (rows)", min_value=5, max_value=min(500, max(20, len(logs_ok))), value=20)
            metrics_over = []
            for i in range(window, len(logs_ok)+1):
                s = logs_ok.iloc[i-window:i]
                try:
                    a = accuracy_score(s[true_col], s[pred_col])
                    f = f1_score(s[true_col], s[pred_col], zero_division=0)
                    p = precision_score(s[true_col], s[pred_col], zero_division=0)
                    r = recall_score(s[true_col], s[pred_col], zero_division=0)
                except Exception:
                    a=f=p=r=None
                metrics_over.append({"idx":i,"accuracy":a,"f1":f,"precision":p,"recall":r})
            mot = pd.DataFrame(metrics_over).set_index("idx")
            st.line_chart(mot[["accuracy","f1","precision","recall"]])
        else:
            st.info("Not enough rows in logs for rolling metrics.")
    else:
        st.info("Logs do not have both prediction & true label columns — live metrics disabled.")

# Quick download summary (JSON)
st.header("Quick Summary Download")
if st.button("Download evaluation summary (JSON)"):
    summ = {"generated_at": datetime.utcnow().isoformat()+"Z", "models":{}}
    for r in model_results:
        summ["models"][r["name"]] = {"metrics": r["metrics"], "pred_error": r["pred_error"]}
    b = io.BytesIO(json.dumps(summ, indent=2).encode("utf-8"))
    st.download_button("Download JSON", b, file_name="evaluation_summary.json", mime="application/json")

