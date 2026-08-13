# -*- coding: utf-8 -*-
"""
SVR multi-output regression workflow for the ratio-based dataset.

The script loads the Excel file, prepares the feature and target matrices,
performs 5-fold cross-validation to tune SVR hyperparameters, fits the best
model, evaluates it on the held-out test set, and exports plots and results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openpyxl.utils import get_column_letter
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ---------------------------
# 1. Paths and data loading
# ---------------------------
script_dir = Path(__file__).resolve().parent
plots_dir = script_dir / "plots"
plots_dir.mkdir(exist_ok=True)
output_excel = script_dir / "svr_results.xlsx"
file_path = script_dir / "Used_Data_100rws - Enhanced.xlsx"

if not file_path.exists():
    raise FileNotFoundError(f"Input file not found: {file_path}")

df = pd.read_excel(file_path)

# ---------------------------
# 2. Feature and target setup
# ---------------------------
target_columns = ["Ker_L/W", "Ker_W/T", "Ker_L/T"]

# Keep the current feature set used for this workflow.
X = df.drop(df.columns[3:19], axis=1)
y = df[target_columns]

# ---------------------------
# 3. Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# ---------------------------
# 4. Hyperparameter tuning with 5-fold CV
# ---------------------------
kernel_values = ["linear", "rbf", "poly"]
C_values = [0.1, 1.0, 10.0]
epsilon_values = [0.01, 0.1, 0.2]
cv = KFold(n_splits=5, shuffle=True, random_state=42)

tuning_results = []

for kernel in kernel_values:
    for C in C_values:
        for epsilon in epsilon_values:
            fold_metrics = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train), start=1):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = MultiOutputRegressor(
                    make_pipeline(
                        StandardScaler(),
                        SVR(kernel=kernel, C=C, epsilon=epsilon),
                    )
                )
                model.fit(X_tr, y_tr)

                y_pred_train = model.predict(X_tr)
                y_pred_val = model.predict(X_val)

                train_r2 = r2_score(y_tr, y_pred_train)
                train_mse = mean_squared_error(y_tr, y_pred_train)
                val_r2 = r2_score(y_val, y_pred_val)
                val_mse = mean_squared_error(y_val, y_pred_val)

                fold_metrics.append(
                    {
                        "fold": fold_idx,
                        "train_r2": train_r2,
                        "train_mse": train_mse,
                        "val_r2": val_r2,
                        "val_mse": val_mse,
                    }
                )

            mean_train_r2 = sum(item["train_r2"] for item in fold_metrics) / len(fold_metrics)
            mean_train_mse = sum(item["train_mse"] for item in fold_metrics) / len(fold_metrics)
            mean_val_r2 = sum(item["val_r2"] for item in fold_metrics) / len(fold_metrics)
            mean_val_mse = sum(item["val_mse"] for item in fold_metrics) / len(fold_metrics)
            tuning_results.append(
                {
                    "kernel": kernel,
                    "C": C,
                    "epsilon": epsilon,
                    "cv_mean_train_r2": mean_train_r2,
                    "cv_mean_train_mse": mean_train_mse,
                    "cv_mean_test_val_r2": mean_val_r2,
                    "cv_mean_test_val_mse": mean_val_mse,
                    "train_r2": [item["train_r2"] for item in fold_metrics],
                    "train_mse": [item["train_mse"] for item in fold_metrics],
                    "val_r2": [item["val_r2"] for item in fold_metrics],
                    "val_mse": [item["val_mse"] for item in fold_metrics],
                    "fold_metrics": fold_metrics,
                }
            )

            train_r2_values = ", ".join(f"{item['train_r2']:.4f}" for item in fold_metrics)
            train_mse_values = ", ".join(f"{item['train_mse']:.4f}" for item in fold_metrics)
            val_r2_values = ", ".join(f"{item['val_r2']:.4f}" for item in fold_metrics)
            val_mse_values = ", ".join(f"{item['val_mse']:.4f}" for item in fold_metrics)

            print(
                f"kernel={kernel} | "
                f"C={C:.3f} | "
                f"epsilon={epsilon:.3f} | "
                f"mean train R2={mean_train_r2:.4f} | "
                f"mean train MSE={mean_train_mse:.4f} | "
                f"mean val R2={mean_val_r2:.4f} | "
                f"mean val MSE={mean_val_mse:.4f} | "
                f"train fold R2={train_r2_values} | "
                f"train fold MSE={train_mse_values} | "
                f"val fold R2={val_r2_values} | "
                f"val fold MSE={val_mse_values}"
            )

tuning_df = pd.DataFrame(tuning_results)
console_tuning_summary = tuning_df[
    [
        "kernel",
        "C",
        "epsilon",
        "cv_mean_train_r2",
        "cv_mean_train_mse",
        "cv_mean_test_val_r2",
        "cv_mean_test_val_mse",
        "train_r2",
        "train_mse",
        "val_r2",
        "val_mse",
    ]
].copy()
console_tuning_summary["train_r2"] = console_tuning_summary["train_r2"].apply(
    lambda scores: ", ".join(f"{x:.4f}" for x in scores)
)
console_tuning_summary["train_mse"] = console_tuning_summary["train_mse"].apply(
    lambda scores: ", ".join(f"{x:.4f}" for x in scores)
)
console_tuning_summary["val_r2"] = console_tuning_summary["val_r2"].apply(
    lambda scores: ", ".join(f"{x:.4f}" for x in scores)
)
console_tuning_summary["val_mse"] = console_tuning_summary["val_mse"].apply(
    lambda scores: ", ".join(f"{x:.4f}" for x in scores)
)
console_tuning_summary = console_tuning_summary.rename(
    columns={
        "cv_mean_train_r2": "cv_mean_train_r2",
        "cv_mean_train_mse": "cv_mean_train_mse",
        "cv_mean_test_val_r2": "cv_mean_test_val_r2",
        "cv_mean_test_val_mse": "cv_mean_test_val_mse",
        "train_r2": "train_fold_r2",
        "train_mse": "train_fold_mse",
        "val_r2": "test_fold_r2",
        "val_mse": "test_fold_mse",
    }
)

best_result = max(tuning_results, key=lambda item: item["cv_mean_test_val_r2"])
best_kernel = best_result["kernel"]
best_C = best_result["C"]
best_epsilon = best_result["epsilon"]

best_model = MultiOutputRegressor(
    make_pipeline(
        StandardScaler(),
        SVR(kernel=best_kernel, C=best_C, epsilon=best_epsilon),
    )
)

print(
    f"\nBest SVR settings based on 5-fold CV R2: "
    f"kernel={best_kernel}, C={best_C:.3f}, epsilon={best_epsilon:.3f}"
)

best_model.fit(X_train, y_train)

# ---------------------------
# 5. Final model evaluation
# ---------------------------
y_pred = best_model.predict(X_test)
y_pred_train = best_model.predict(X_train)

train_r2 = r2_score(y_train, y_pred_train)
train_mse = mean_squared_error(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)

print(f"Train R2: {train_r2:.4f}")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test R2: {test_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# ---------------------------
# 6. Prediction tables
# ---------------------------
results = pd.DataFrame(
    {
        "Actual_L/W": y_test["Ker_L/W"].values,
        "Pred_L/W": y_pred[:, 0],
        "Actual_W/T": y_test["Ker_W/T"].values,
        "Pred_W/T": y_pred[:, 1],
        "Actual_L/T": y_test["Ker_L/T"].values,
        "Pred_L/T": y_pred[:, 2],
    }
)

train_results = pd.DataFrame(
    {
        "Actual_L/W": y_train["Ker_L/W"].values,
        "Pred_L/W": y_pred_train[:, 0],
        "Actual_W/T": y_train["Ker_W/T"].values,
        "Pred_W/T": y_pred_train[:, 1],
        "Actual_L/T": y_train["Ker_L/T"].values,
        "Pred_L/T": y_pred_train[:, 2],
    }
)

# ---------------------------
# 7. Save visual diagnostics
# ---------------------------
for feature_col in X.columns:
    for target_col in target_columns:
        plt.figure(figsize=(6, 4))
        plt.scatter(X[feature_col], y[target_col], alpha=0.7)
        plt.xlabel(feature_col)
        plt.ylabel(target_col)
        plt.title(f"{feature_col} vs {target_col}")
        plt.grid(True, alpha=0.3)
        safe_name = f"{feature_col}_vs_{target_col}".replace("/", "_")
        plt.savefig(plots_dir / f"{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

for i, col in enumerate(target_columns):
    plt.figure()
    plt.scatter(y_test.iloc[:, i], y_pred[:, i])
    plt.xlabel(f"Actual {col}")
    plt.ylabel(f"Predicted {col}")
    plt.title(f"Actual vs Predicted - {col}")
    plt.plot(
        [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
        [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
    )
    safe_col = col.replace("/", "_")
    plt.savefig(plots_dir / f"actual_vs_predicted_{safe_col}.png", dpi=300, bbox_inches="tight")
    plt.close()

for i, col in enumerate(target_columns):
    residuals = y_test.iloc[:, i] - y_pred[:, i]

    plt.figure()
    plt.scatter(y_pred[:, i], residuals)
    plt.xlabel(f"Predicted {col}")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot - {col}")
    plt.axhline(y=0)
    safe_col = col.replace("/", "_")
    plt.savefig(plots_dir / f"residual_plot_{safe_col}.png", dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------
# 8. Feature importance using the fitted SVR models
# ---------------------------
feature_importance = {}
for i, col in enumerate(target_columns):
    estimator = best_model.estimators_[i]
    result = permutation_importance(
        estimator,
        X_train,
        y_train[col],
        scoring="r2",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    feature_importance[col] = pd.Series(result.importances_mean, index=X.columns)

importance_df = pd.DataFrame(feature_importance)

for col in target_columns:
    plt.figure()
    importance_df[col].sort_values().plot(kind="barh")
    plt.title(f"Feature Importance for {col} (SVR permutation importance)")
    plt.xlabel("Mean R2 decrease")
    safe_col = col.replace("/", "_")
    plt.savefig(plots_dir / f"feature_importance_{safe_col}.png", dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------
# 9. Export results to Excel
# ---------------------------
def auto_adjust_column_widths(writer, dataframe, sheet_name, max_width=50):
    worksheet = writer.book[sheet_name]
    for idx, column in enumerate(dataframe.columns, start=1):
        values = [str(column)] + [str(value) for value in dataframe[column].fillna("").astype(str)]
        max_length = max(len(value) for value in values)
        adjusted_width = min(max_length + 2, max_width)
        worksheet.column_dimensions[get_column_letter(idx)].width = adjusted_width


with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Input_Data", index=False)
    auto_adjust_column_widths(writer, df, "Input_Data")

    tuning_df.to_excel(writer, sheet_name="SVR_Tuning", index=False)
    auto_adjust_column_widths(writer, tuning_df, "SVR_Tuning")

    console_tuning_summary.to_excel(writer, sheet_name="Console_Tuning_Summary", index=False)
    auto_adjust_column_widths(writer, console_tuning_summary, "Console_Tuning_Summary")

    final_summary = pd.DataFrame(
        [
            {
                "best_kernel": best_kernel,
                "best_C": best_C,
                "best_epsilon": best_epsilon,
                "train_r2": train_r2,
                "train_mse": train_mse,
                "test_r2": test_r2,
                "test_mse": test_mse,
                "support_vectors_first_target": best_model.estimators_[0]
                .named_steps["svr"]
                .support_
                .shape[0],
            }
        ]
    )
    final_summary.to_excel(writer, sheet_name="Final_Summary", index=False)
    auto_adjust_column_widths(writer, final_summary, "Final_Summary")

    results.to_excel(writer, sheet_name="Test_Predictions", index=False)
    auto_adjust_column_widths(writer, results, "Test_Predictions")

    train_results.to_excel(writer, sheet_name="Train_Predictions", index=False)
    auto_adjust_column_widths(writer, train_results, "Train_Predictions")

    importance_df.to_excel(writer, sheet_name="Feature_Importance", index=True)
    auto_adjust_column_widths(writer, importance_df, "Feature_Importance")

    svr_summary = pd.DataFrame(
        [
            {
                "target_column": target_col,
                "support_vectors": best_model.estimators_[i]
                .named_steps["svr"]
                .support_
                .shape[0],
            }
            for i, target_col in enumerate(target_columns)
        ]
    )
    svr_summary.to_excel(writer, sheet_name="SVR_Summary", index=False)
    auto_adjust_column_widths(writer, svr_summary, "SVR_Summary")

print(f"Results saved to {output_excel}")
