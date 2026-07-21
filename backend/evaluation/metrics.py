import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import plotly.graph_objects as go
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

def evaluate_model(model, test_ds, threshold=0.5):
    """
    Evaluates the model and computes comprehensive clinical metrics.
    """
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds)
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs).squeeze()
    
    # Threshold for binary classification
    y_pred = (y_pred_probs > threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    acc         = (tp + tn) / (tp + tn + fp + fn)
    f1          = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    metrics = {
        "accuracy": float(acc),
        "sensitivity": float(sensitivity), # Recall / Pneumonia Recall
        "specificity": float(specificity), # Normal Recall
        "precision": float(precision),     # PPV
        "npv": float(npv),                 # Negative Predictive Value
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm,
        "counts": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
    }
    
    return metrics, (fpr, tpr, roc_auc)


def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    """
    Plots and optionally saves ROC curve using Plotly.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:0.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random guess'))
    fig.update_layout(
        title='Receiver Operating Characteristic',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(save_path)
    
    return fig
