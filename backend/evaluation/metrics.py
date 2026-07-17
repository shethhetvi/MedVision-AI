import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import plotly.graph_objects as go
import os

def evaluate_model(model, test_ds):
    """
    Evaluates the model and computes key metrics.
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
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    metrics["roc_auc"] = roc_auc
    
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
