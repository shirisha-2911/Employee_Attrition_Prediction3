# EMPLOYEE ATTRITION PREDICTION - HYPERPARAMETER TUNING (HPT)
# Adapted from the Iris Optuna + MLflow HPT script.
#
# Install (run once in your environment):
# pip install pandas numpy scikit-learn optuna mlflow joblib
#
# Update DATA_PATH to the location of your Employee Attrition.csv file.

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, make_scorer,
    classification_report, confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
DATA_PATH = "data/Employee Attrition.csv" # Change this path if required
TARGET = "Attrition"
POSITIVE_CLASS = "Left"              # Class we want to identify

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# 2. BASIC CLEANING
# ---------------------------------------------------------
df = df.drop_duplicates().copy()
df = df.dropna(subset=[TARGET])

# Match the target labels used in the employee attrition notebook.
# The target must contain the classes "Stayed" and "Left".
print("\nTarget distribution:")
print(df[TARGET].value_counts())

X = df.drop(columns=[TARGET, "Attrition_binary", "Employee ID"], errors="ignore")
y = df[TARGET]

if X.isnull().any().any():
    raise ValueError("Input features contain missing values. Add imputation before HPT.")

# ---------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 4. PREPROCESSING
#    Fit preprocessing inside each CV fold to avoid leakage.
# ---------------------------------------------------------
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

def make_preprocessor(scaler_type="standard"):
    scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
    return ColumnTransformer(
        transformers=[
            ("num", scaler, numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )

# ---------------------------------------------------------
# 5. OPTUNA OBJECTIVE FUNCTIONS
#    Optimize recall for employees who LEFT.
# ---------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
recall_left_scorer = make_scorer(recall_score, pos_label=POSITIVE_CLASS)

def evaluate_pipeline(trial, model, scaler_type):
    pipe = Pipeline([
        ("preprocessor", make_preprocessor(scaler_type)),
        ("model", model)
    ])
    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        scoring=recall_left_scorer,
        cv=cv,
        n_jobs=-1
    )
    return scores.mean()


def objective_knn(trial):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
    model = KNeighborsClassifier(
        n_neighbors=trial.suggest_int("n_neighbors", 3, 21, step=2),
        weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        p=trial.suggest_int("p", 1, 2)
    )
    return evaluate_pipeline(trial, model, scaler_type)


def objective_logistic(trial):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
    model = LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 100, log=True),
        solver=trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    )
    return evaluate_pipeline(trial, model, scaler_type)


def objective_dt(trial):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
    model = DecisionTreeClassifier(
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        max_depth=trial.suggest_int("max_depth", 2, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        class_weight="balanced",
        random_state=42
    )
    return evaluate_pipeline(trial, model, scaler_type)


def objective_svm(trial):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
    params = {
        "C": trial.suggest_float("C", 1e-3, 100, log=True),
        "kernel": kernel,
        "class_weight": "balanced"
    }
    if kernel in ["rbf", "poly"]:
        params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
    model = SVC(**params)
    return evaluate_pipeline(trial, model, scaler_type)


def objective_gnb(trial):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
    model = GaussianNB(
        var_smoothing=trial.suggest_float("var_smoothing", 1e-11, 1e-7, log=True)
    )
    return evaluate_pipeline(trial, model, scaler_type)


def objective_rf(trial):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        max_depth=trial.suggest_categorical("max_depth", [5, 10, 20, None]),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    return evaluate_pipeline(trial, model, scaler_type)


def objective_gb(trial):
    scaler_type = trial.suggest_categorical("scaler_type", ["standard", "minmax"])
    model = GradientBoostingClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        random_state=42
    )
    return evaluate_pipeline(trial, model, scaler_type)


# ---------------------------------------------------------
# 6. RUN OPTUNA HYPERPARAMETER OPTIMIZATION
# ---------------------------------------------------------
objectives = {
    "KNN": objective_knn,
    "LogisticRegression": objective_logistic,
    "DecisionTree": objective_dt,
    "SVM": objective_svm,
    "GaussianNB": objective_gnb,
    "RandomForest": objective_rf,
    "GradientBoosting": objective_gb
}

mlflow.set_experiment("EMPLOYEE_ATTRITION_HPT")

results = {}
best_pipelines = {}
N_TRIALS = 5

for model_name, objective in objectives.items():

    # Create a separate MLflow experiment for each model
    mlflow.set_experiment(experiment_name=model_name)

    print(f"\n----- Optimizing {model_name} -----")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"Attrition_{model_name}"
    )

    start_time = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    optimization_time = time.time() - start_time

    best_params = study.best_params
    best_cv_recall = study.best_value
    print("Best parameters:", best_params)
    print(f"Best CV recall for Left: {best_cv_recall:.4f}")

    # -----------------------------------------------------
    # 7. BUILD THE BEST MODEL USING OPTUNA PARAMETERS
    # -----------------------------------------------------
    scaler_type = best_params["scaler_type"]

    if model_name == "KNN":
        best_model = KNeighborsClassifier(
            n_neighbors=best_params["n_neighbors"],
            weights=best_params["weights"],
            p=best_params["p"]
        )
    elif model_name == "LogisticRegression":
        best_model = LogisticRegression(
            C=best_params["C"],
            solver=best_params["solver"],
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    elif model_name == "DecisionTree":
        best_model = DecisionTreeClassifier(
            criterion=best_params["criterion"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            class_weight="balanced",
            random_state=42
        )
    elif model_name == "SVM":
        svm_params = {
            "C": best_params["C"],
            "kernel": best_params["kernel"],
            "class_weight": "balanced"
        }
        if best_params["kernel"] in ["rbf", "poly"]:
            svm_params["gamma"] = best_params["gamma"]
        if best_params["kernel"] == "poly":
            svm_params["degree"] = best_params["degree"]
        best_model = SVC(**svm_params)
    elif model_name == "GaussianNB":
        best_model = GaussianNB(var_smoothing=best_params["var_smoothing"])
    elif model_name == "RandomForest":
        best_model = RandomForestClassifier(
            n_estimators=best_params["n_estimators"],
            criterion=best_params["criterion"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            max_features=best_params["max_features"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    else:  # GradientBoosting
        best_model = GradientBoostingClassifier(
            n_estimators=best_params["n_estimators"],
            learning_rate=best_params["learning_rate"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            subsample=best_params["subsample"],
            random_state=42
        )

    best_pipeline = Pipeline([
        ("preprocessor", make_preprocessor(scaler_type)),
        ("model", best_model)
    ])

    # Train on the full training split and evaluate once on held-out test data.
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    train_accuracy = best_pipeline.score(X_train, y_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=POSITIVE_CLASS, zero_division=0)

    best_pipelines[model_name] = best_pipeline
    results[model_name] = {
        "best_cv_recall": best_cv_recall,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "precision_left": precision,
        "recall_left": recall,
        "f1_left": f1,
        "optimization_time_sec": optimization_time,
        "best_params": best_params
    }

    # Log the run and model to MLflow.
    with mlflow.start_run(run_name=f"{model_name}_best"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("scaler_type", scaler_type)
        mlflow.log_params({k: str(v) for k, v in best_params.items()})
        mlflow.log_metric("best_cv_recall_left", best_cv_recall)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("precision_left", precision)
        mlflow.log_metric("recall_left", recall)
        mlflow.log_metric("f1_left", f1)
        mlflow.log_metric("optimization_time_sec", optimization_time)
        mlflow.sklearn.log_model(
    best_pipeline,
    artifact_path=f"{model_name}_pipeline",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy:  {test_accuracy:.4f}")
    print(f"Precision (Left): {precision:.4f}")
    print(f"Recall (Left):    {recall:.4f}")
    print(f"F1-score (Left):  {f1:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# 8. MODEL COMPARISON
# ---------------------------------------------------------
results_df = pd.DataFrame(results).T
print("\n----- HPT MODEL COMPARISON -----")
print(results_df[[
    "best_cv_recall", "train_accuracy", "test_accuracy",
    "precision_left", "recall_left", "f1_left",
    "optimization_time_sec"
]].sort_values("recall_left", ascending=False))

# Select the model with the highest cross-validation recall for Left.
# The held-out test set is used for final evaluation.
best_name = max(results, key=lambda name: results[name]["best_cv_recall"])
best_attrition_pipeline = best_pipelines[best_name]

print(f"\nBest model based on CV recall: {best_name}")
print("Classification report:")
print(classification_report(
    y_test,
    best_attrition_pipeline.predict(X_test),
    zero_division=0
))

# ---------------------------------------------------------
# 9. SAVE THE BEST PIPELINE AND COMPARISON RESULTS
# ---------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_attrition_pipeline, "models/best_attrition_hpt_pipeline.pkl")
results_df.to_csv("models/attrition_hpt_results.csv", index=True)

print("\nSaved:")
print("1. models/best_attrition_hpt_pipeline.pkl")
print("2. models/attrition_hpt_results.csv")
print("\nTo view MLflow runs, open a terminal in this folder and run:")
print("mlflow ui")
print("Then open http://127.0.0.1:5000")
