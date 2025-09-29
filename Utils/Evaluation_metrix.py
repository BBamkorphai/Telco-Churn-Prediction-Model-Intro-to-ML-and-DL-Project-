from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, r2_score, roc_curve
)
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_sklearn_model(model, X_test, y_test):
    """
    Evaluate sklearn-style models (Logistic Regression, Tree-based, etc.)
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Probabilities (if available)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred  # fallback (no probas available)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # Sensitivity
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])   # TNR
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    r2 = r2_score(y_test, y_pred)

    print("\nEvaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity (Recall): {recall:.4f}")
    print(f"Specificity (TNR): {specificity:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"R² score: {r2:.4f}")
    
    print(classification_report(y_test, y_pred))

    # --- PLOTS ---
    # Confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.title("ROC Curve")
    plt.show()

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "r2": r2,
        "confusion_matrix": cm
    }
