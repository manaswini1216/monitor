# monitoring/dashboard/shap_utils.py
"""
SHAP helper utilities for Streamlit dashboard.
Provides:
 - compute_shap_values(model, X, nsamples=200)
 - shap_summary_plot_html(shap_values, X_sample)
 - shap_force_plot_html(shap_values, X_sample, idx)
The functions are robust: try TreeExplainer -> KernelExplainer fallback -> sampling.
Caching is used to avoid re-computation in Streamlit.
"""

import io
import json
import numpy as np
import pandas as pd

# Use lazy import for shap to avoid import error if not installed
def _try_import_shap():
    try:
        import shap
        return shap
    except Exception:
        return None

def _safe_sample_df(X, max_rows=500):
    """Return a sample of X (preserve columns)."""
    if X.shape[0] <= max_rows:
        return X.copy()
    return X.sample(n=max_rows, random_state=42).reset_index(drop=True)

def compute_shap_values(model, X, nsamples=200):
    """
    Compute SHAP values for model on dataframe X.
    Returns (explainer, shap_values, X_sample) or raises informative Exception.
    Uses a small sample for expensive methods.
    """
    shap = _try_import_shap()
    if shap is None:
        raise RuntimeError("shap library not installed. Install with `pip install shap`")

    # Use sample for heavy models
    X_sample = _safe_sample_df(X, max_rows=nsamples)

    # Try tree explainer first (fast for tree models)
    try:
        explainer = shap.Explainer(model, X_sample, feature_names=list(X.columns))
        shap_values = explainer(X_sample)
        return explainer, shap_values, X_sample
    except Exception:
        # Try TreeExplainer explicitly (older shap versions)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)  # older API
            # wrap to new style if possible
            return explainer, shap_values, X_sample
        except Exception:
            pass

    # Fallback: KernelExplainer (slow) — use small background sample
    try:
        # select a small background sample
        background = X_sample.sample(n=min(50, max(2, int(0.1 * max(1, X_sample.shape[0])))), random_state=42)
        explainer = shap.KernelExplainer(lambda v: model.predict_proba(v) if hasattr(model, "predict_proba") else model.predict(v), background)
        shap_values = explainer.shap_values(X_sample, nsamples=nsamples)
        return explainer, shap_values, X_sample
    except Exception as e:
        raise RuntimeError(f"All SHAP explainer attempts failed: {e}")

def shap_summary_plot_html(shap_values, X_sample):
    """
    Return HTML string for SHAP summary (beeswarm) - which can be rendered in st.components.v1.html
    Expects shap_values in shap.Explanation / array depending on explainer.
    """
    shap = _try_import_shap()
    if shap is None:
        raise RuntimeError("shap library not installed.")
    # try new plot API
    try:
        buf = io.BytesIO()
        # shap has its own JS that needs to be rendered; return HTML using shap's plotting
        # create matplotlib figure if possible
        try:
            import matplotlib.pyplot as plt
            plt.rcParams.update({'figure.dpi': 100})
            shap.plots.beeswarm(shap_values, show=False)
            plt.tight_layout()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            import base64
            return f"<img src='data:image/png;base64,{base64.b64encode(buf.read()).decode()}'/>"
        except Exception:
            # fallback: use shap.summary_plot to a file
            try:
                shap.summary_plot(shap_values.values if hasattr(shap_values, "values") else shap_values, X_sample, show=False)
                plt.tight_layout()
                plt.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                import base64
                return f"<img src='data:image/png;base64,{base64.b64encode(buf.read()).decode()}'/>"
            except Exception as e:
                raise RuntimeError("SHAP plotting failed: " + str(e))
    except Exception as e:
        raise RuntimeError("Could not produce SHAP summary HTML: " + str(e))

def shap_force_plot_html(explainer, shap_values, X_sample, idx=0):
    """
    Return HTML string for SHAP force plot for a single instance (index idx in X_sample).
    Works best with shap.Explanation objects (new API). For older arrays, try to convert.
    """
    shap = _try_import_shap()
    if shap is None:
        raise RuntimeError("shap library not installed.")
    try:
        # new style: shap.plots.force with matplotlib or JS
        try:
            html = shap.plots.waterfall(shap_values[idx], show=False)  # might return a matplotlib object
            # If it's a matplotlib figure, render to PNG
            import matplotlib.pyplot as plt, io, base64
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            return f"<img src='data:image/png;base64,{base64.b64encode(buf.read()).decode()}'/>"
        except Exception:
            # fallback to JS force_plot
            try:
                if hasattr(shap, "force_plot"):
                    # build JS force plot and return embeddable HTML
                    if hasattr(shap_values, "values") and hasattr(explainer, "expected_value"):
                        fp = shap.force_plot(explainer.expected_value, shap_values.values[idx], X_sample.iloc[idx], matplotlib=False)
                        # shap returns a JS object string via html()
                        html = f"<head>{shap.utils._html.UnicodeConverter.getjs()}</head>" if hasattr(shap.utils, "_html") else ""
                        html += shap.plots._force.html(fp)
                        return html
                    else:
                        # older API: explainer.expected_value + shap_values[idx]
                        fp = shap.force_plot(explainer.expected_value, shap_values[idx], X_sample.iloc[idx], matplotlib=False)
                        return shap.plots._force.html(fp)
            except Exception as e:
                raise RuntimeError("SHAP force plot failed: " + str(e))
    except Exception as e:
        raise RuntimeError("Could not produce SHAP force HTML: " + str(e))
