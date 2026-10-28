"""
Network Intrusion Detection System (NIDS) - Model Training Pipeline
====================================================================
Trains machine learning models on the NSL-KDD dataset to classify
network connections as normal or as one of four attack categories
(DoS, Probe, R2L, U2R).

Models trained (as required by the proposal):
    - Logistic Regression
    - Decision Tree
    - Random Forest

Outputs (saved to /models and /eda):
    - best_model.joblib        : the best performing trained pipeline
    - scaler.joblib            : fitted StandardScaler
    - encoders.joblib          : label encoders for categorical features
    - feature_info.joblib      : feature names, defaults, options for the web app
    - metrics.json             : performance of all models
    - several EDA plot images
"""

import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_DIR = "models"
EDA_DIR = "eda"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. COLUMN NAMES (NSL-KDD has 41 features + attack label + difficulty level)
# ---------------------------------------------------------------------------
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "attack", "level",
]

CATEGORICAL = ["protocol_type", "service", "flag"]

# Map each specific attack to one of the 5 standard NSL-KDD categories
ATTACK_MAP = {
    "normal": "normal",
    # DoS - Denial of Service
    "neptune": "DoS", "back": "DoS", "land": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "mailbomb": "DoS", "apache2": "DoS",
    "processtable": "DoS", "udpstorm": "DoS", "worm": "DoS",
    # Probe - Surveillance / scanning
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L - Remote to Local (unauthorized remote access)
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R - User to Root (privilege escalation)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def load_data():
    print("[1/6] Loading NSL-KDD dataset ...")
    train = pd.read_csv(os.path.join(DATA_DIR, "KDDTrain.txt"), names=COLUMNS)
    test = pd.read_csv(os.path.join(DATA_DIR, "KDDTest.txt"), names=COLUMNS)

    # drop the difficulty 'level' column (not a real feature)
    train = train.drop(columns=["level"])
    test = test.drop(columns=["level"])

    # map attacks to 5 categories
    train["category"] = train["attack"].map(ATTACK_MAP).fillna("normal")
    test["category"] = test["attack"].map(ATTACK_MAP).fillna("normal")
    train = train.drop(columns=["attack"])
    test = test.drop(columns=["attack"])

    print(f"      Train shape: {train.shape},  Test shape: {test.shape}")
    return train, test


def run_eda(train):
    print("[2/6] Running exploratory data analysis (saving plots) ...")
    sns.set_style("whitegrid")

    # (a) Attack category distribution
    plt.figure(figsize=(8, 5))
    order = train["category"].value_counts().index
    ax = sns.countplot(data=train, x="category", order=order, palette="viridis",
                       hue="category", legend=False)
    ax.set_title("Distribution of Traffic Categories (NSL-KDD Train)", fontweight="bold")
    ax.set_xlabel("Category"); ax.set_ylabel("Number of connections")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR, "01_category_distribution.png"), dpi=120)
    plt.close()

    # (b) Protocol type vs category
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=train, x="protocol_type", hue="category", palette="Set2")
    ax.set_title("Protocol Type by Traffic Category", fontweight="bold")
    ax.set_xlabel("Protocol"); ax.set_ylabel("Count")
    plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR, "02_protocol_vs_category.png"), dpi=120)
    plt.close()

    # (c) Correlation heatmap of numeric features
    plt.figure(figsize=(14, 11))
    num = train.select_dtypes(include=[np.number])
    corr = num.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.6}, xticklabels=True, yticklabels=True)
    plt.title("Correlation Heatmap of Numeric Features", fontweight="bold")
    plt.xticks(fontsize=6, rotation=90); plt.yticks(fontsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR, "03_correlation_heatmap.png"), dpi=120)
    plt.close()

    # (d) Top services involved in attacks
    plt.figure(figsize=(9, 5))
    attack_only = train[train["category"] != "normal"]
    top_services = attack_only["service"].value_counts().head(10)
    ax = sns.barplot(x=top_services.values, y=top_services.index, palette="rocket",
                     hue=top_services.index, legend=False)
    ax.set_title("Top 10 Services Targeted in Attacks", fontweight="bold")
    ax.set_xlabel("Number of attack connections"); ax.set_ylabel("Service")
    plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR, "04_top_attacked_services.png"), dpi=120)
    plt.close()

    print(f"      Saved 4 EDA plots to ./{EDA_DIR}/")


def preprocess(train, test):
    print("[3/6] Preprocessing (encoding + scaling) ...")

    # Encode categorical features. Fit on the union of train+test values so the
    # encoder never sees an unknown category at test time.
    encoders = {}
    for col in CATEGORICAL:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        encoders[col] = le

    # Encode the target
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(train["category"])
    y_test = target_encoder.transform(test["category"])

    X_train = train.drop(columns=["category"])
    X_test = test.drop(columns=["category"])
    feature_names = list(X_train.columns)

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    encoders["__target__"] = target_encoder
    return (X_train_s, X_test_s, y_train, y_test, scaler, encoders,
            feature_names, X_train, target_encoder)


def train_and_evaluate(X_train, X_test, y_train, y_test, target_encoder):
    print("[4/6] Training models ...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=25, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=30,
                                                 n_jobs=-1, random_state=42),
    }

    results = {}
    trained = {}
    class_names = list(target_encoder.classes_)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"      -> {name}")
        # 5-fold cross-validation on the training set (true learning capacity)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="accuracy", n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="weighted")
        results[name] = {
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(float(acc), 4),
            "test_weighted_f1": round(float(f1), 4),
            "report": classification_report(y_test, pred,
                                             target_names=class_names,
                                             output_dict=True, zero_division=0),
        }
        trained[name] = model
        print(f"         CV acc={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})  "
              f"test acc={acc:.4f}  test f1={f1:.4f}")

    # Pick the best model by cross-validated accuracy (robust selection)
    best_name = max(results, key=lambda k: results[k]["cv_accuracy_mean"])
    print(f"      BEST MODEL (by CV accuracy): {best_name}")

    # Confusion matrix for best model
    best_pred = trained[best_name].predict(X_test)
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {best_name}", fontweight="bold")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR, "05_confusion_matrix.png"), dpi=120)
    plt.close()

    # Model comparison bar chart
    plt.figure(figsize=(8, 5))
    names = list(results.keys())
    cv_accs = [results[n]["cv_accuracy_mean"] for n in names]
    test_accs = [results[n]["test_accuracy"] for n in names]
    x = np.arange(len(names)); w = 0.35
    plt.bar(x - w/2, cv_accs, w, label="CV Accuracy (train)", color="#4C72B0")
    plt.bar(x + w/2, test_accs, w, label="Test Accuracy (KDDTest+)", color="#DD8452")
    plt.xticks(x, names, rotation=10); plt.ylim(0, 1.05)
    plt.title("Model Performance Comparison", fontweight="bold")
    plt.ylabel("Accuracy"); plt.legend()
    for i, v in enumerate(cv_accs):
        plt.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(test_accs):
        plt.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(EDA_DIR, "06_model_comparison.png"), dpi=120)
    plt.close()

    return trained[best_name], best_name, results


def main():
    train, test = load_data()
    run_eda(train.copy())

    # Capture example rows with ORIGINAL string values BEFORE encoding
    raw_examples = {}
    for cat in sorted(train["category"].unique()):
        raw_examples[cat] = train[train["category"] == cat].iloc[0].to_dict()
    (X_train_s, X_test_s, y_train, y_test, scaler, encoders,
     feature_names, X_train_raw, target_encoder) = preprocess(train, test)

    best_model, best_name, results = train_and_evaluate(
        X_train_s, X_test_s, y_train, y_test, target_encoder)

    print("[5/6] Building feature metadata for the web app ...")
    # Sensible per-feature defaults (median of train) + min/max for sliders,
    # and the original category options for the categorical dropdowns.
    feature_info = {"features": [], "categorical_options": {}}
    medians = X_train_raw.median()
    for col in feature_names:
        info = {
            "name": col,
            "is_categorical": col in CATEGORICAL,
            "default": float(medians[col]),
            "min": float(X_train_raw[col].min()),
            "max": float(X_train_raw[col].max()),
        }
        feature_info["features"].append(info)

    # Human-readable categorical options (original string -> encoded int)
    for col in CATEGORICAL:
        le = encoders[col]
        feature_info["categorical_options"][col] = {
            cls: int(le.transform([cls])[0]) for cls in le.classes_
        }

    feature_info["target_classes"] = list(target_encoder.classes_)
    feature_info["best_model"] = best_name

    # A few real example rows users can load with one click in the web app
    examples = {}
    for cat, row in raw_examples.items():
        examples[cat] = {c: (str(row[c]) if c in CATEGORICAL else float(row[c]))
                         for c in feature_names}
    feature_info["examples"] = examples

    print("[6/6] Saving artifacts ...")
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    joblib.dump(feature_info, os.path.join(MODEL_DIR, "feature_info.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n==================== SUMMARY ====================")
    for name, r in results.items():
        print(f"{name:<22} CV_acc={r['cv_accuracy_mean']:.4f}  "
              f"test_acc={r['test_accuracy']:.4f}  test_f1={r['test_weighted_f1']:.4f}")
    print(f"Best model saved: {best_name}")
    print("=================================================")


if __name__ == "__main__":
    main()
