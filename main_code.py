import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix, classification_report)

np.random.seed(42)

# ---------------------------------------------------------------
# 1. Load data, drop seed_id, separate features/target
# ---------------------------------------------------------------
csv_path = r"C:\Users\amrit\updated_file.csv"
df = pd.read_csv(csv_path)
seed_ids = df["seed_id"]
X = df.drop(columns=["seed_id", "Actual output"])
y = df["Actual output"]

feature_names = X.columns.tolist()
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Class balance:\n{y.value_counts()}")

# ---------------------------------------------------------------
# 2. Train/test split (stratified, since classes are imbalanced 228:72)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------
# 3. Define candidate models
# ---------------------------------------------------------------
models = {
    "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "SVM (Linear)": SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42),
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}

results = {}
proba_store = {}
pred_store = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")

    results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "roc_auc": roc_auc,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()
    }
    proba_store[name] = y_proba
    pred_store[name] = y_pred

results_df = pd.DataFrame(results).T.sort_values("f1", ascending=False)
print("\n=== Model Comparison ===")
print(results_df.round(4))

best_model_name = results_df.index[0]
print(f"\nBest model by F1: {best_model_name}")

# ---------------------------------------------------------------
# 4. Save metrics table
# ---------------------------------------------------------------
results_df.round(4).to_csv("model_comparison_metrics.csv")

# ---------------------------------------------------------------
# 5. PLOTS
# ---------------------------------------------------------------
plt.style.use("default")
colors = plt.cm.tab10.colors

# --- Plot 1: Grouped bar chart of metrics across models ---
fig, ax = plt.subplots(figsize=(12, 6))
metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
x = np.arange(len(results_df.index))
width = 0.15
for i, metric in enumerate(metrics_to_plot):
    ax.bar(x + i * width, results_df[metric], width, label=metric.upper(), color=colors[i])
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df.index, rotation=20, ha="right")
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Accuracy, Precision, Recall, F1, ROC-AUC")
ax.legend(loc="lower right")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot1_metrics_comparison.png", dpi=150)
plt.close()

# --- Plot 2: ROC curves for all models ---
fig, ax = plt.subplots(figsize=(7, 7))
for i, name in enumerate(models.keys()):
    fpr, tpr, _ = roc_curve(y_test, proba_store[name])
    auc_val = results[name]["roc_auc"]
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=colors[i])
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot2_roc_curves.png", dpi=150)
plt.close()

# --- Plot 3: Confusion matrix for best model ---
cm = confusion_matrix(y_test, pred_store[best_model_name])
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted 0", "Predicted 1"])
ax.set_yticklabels(["Actual 0", "Actual 1"])
ax.set_title(f"Confusion Matrix — {best_model_name}")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("plot3_confusion_matrix.png", dpi=150)
plt.close()

# --- Plot 4: Predicted probability vs Actual class, with best-fit line ---

y_proba_best = proba_store[best_model_name]
y_test_arr = y_test.values.astype(float)


# jitter the actual 0/1 values slightly on x-axis just for visual separation; keep y = predicted prob
fig, ax = plt.subplots(figsize=(8, 6))
jitter = np.random.normal(0, 0.02, size=len(y_test_arr))
ax.scatter(y_test_arr + jitter, y_proba_best, alpha=0.6, s=40,
           c=["#d62728" if v == 0 else "#2ca02c" for v in y_test_arr], edgecolor="k", linewidth=0.3)

# best-fit straight line: predicted_proba = m * actual + c
m, c = np.polyfit(y_test_arr, y_proba_best, 1)
xs = np.array([-0.1, 1.1])
ax.plot(xs, m * xs + c, "b--", linewidth=2,
        label=f"Best-fit line: y = {m:.3f}x + {c:.3f}")
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="Decision threshold (0.5)")

ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Actual Output (0 = defective, 1 = good)")
ax.set_ylabel("Predicted Probability of Class 1")
ax.set_title(f"Predicted Probability vs Actual Output — {best_model_name}")
ax.legend(loc="center left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot4_predicted_vs_actual.png", dpi=150)
plt.close()

# --- Plot 5: Predicted class vs actual class scatter (cleaner version) with regression line ---
y_pred_best = pred_store[best_model_name]
fig, ax = plt.subplots(figsize=(7, 6))
jitter_x = np.random.normal(0, 0.05, size=len(y_test_arr))
jitter_y = np.random.normal(0, 0.05, size=len(y_pred_best))
correct = (y_test_arr == y_pred_best)
ax.scatter((y_test_arr + jitter_x)[correct], (y_pred_best + jitter_y)[correct],
           c="#2ca02c", label="Correct prediction", alpha=0.7, s=45, edgecolor="k", linewidth=0.3)
ax.scatter((y_test_arr + jitter_x)[~correct], (y_pred_best + jitter_y)[~correct],
           c="#d62728", label="Incorrect prediction", alpha=0.8, s=45, edgecolor="k", linewidth=0.3)
m2, c2 = np.polyfit(y_test_arr, y_pred_best.astype(float), 1)
xs = np.array([-0.2, 1.2])
ax.plot(xs, m2 * xs + c2, "b--", linewidth=2, label=f"Best-fit line: y={m2:.2f}x+{c2:.2f}")
ax.plot(xs, xs, "k:", alpha=0.4, label="Perfect prediction (y=x)")
ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.3, 1.3)
ax.set_xlabel("Actual Output")
ax.set_ylabel("Predicted Output")
ax.set_title(f"Predicted vs Actual Class — {best_model_name}")
ax.legend(loc="center")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot5_predicted_class_vs_actual.png", dpi=150)
plt.close()

print("\nAll plots saved.")
print(classification_report(y_test, pred_store[best_model_name]))