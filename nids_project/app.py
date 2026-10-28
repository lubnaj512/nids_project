"""
Network Intrusion Detection System - Web Application
=====================================================
A Flask web app that loads the trained Random Forest model and classifies
a network connection as normal traffic or one of four attack categories.

Run:
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

MODEL_DIR = "models"
CATEGORICAL = ["protocol_type", "service", "flag"]

# ---------------------------------------------------------------------------
# Load trained artifacts once at startup
# ---------------------------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
feature_info = joblib.load(os.path.join(MODEL_DIR, "feature_info.joblib"))

FEATURES = [f["name"] for f in feature_info["features"]]
TARGET_ENCODER = encoders["__target__"]
CLASS_NAMES = list(TARGET_ENCODER.classes_)

# Features surfaced as inputs in the UI (the most interpretable ones).
# Every other feature uses its median default unless an example is loaded.
KEY_FEATURES = [
    "protocol_type", "service", "flag", "duration", "src_bytes", "dst_bytes",
    "count", "srv_count", "serror_rate", "same_srv_rate", "dst_host_count",
    "dst_host_srv_count", "logged_in", "dst_host_same_srv_rate",
]

# Human-friendly descriptions of each attack category for the result panel
CATEGORY_INFO = {
    "normal": ("Normal Traffic", "No malicious activity detected in this connection."),
    "DoS": ("Denial of Service", "Attempt to overwhelm a host or service so legitimate users are denied access (e.g. neptune, smurf, teardrop)."),
    "Probe": ("Probe / Scan", "Surveillance and scanning to map the network and find vulnerabilities (e.g. satan, nmap, portsweep)."),
    "R2L": ("Remote to Local", "Unauthorized access from a remote machine, such as password guessing (e.g. guess_passwd, warezclient)."),
    "U2R": ("User to Root", "An attacker with a normal account exploits a vulnerability to gain root privileges (e.g. buffer_overflow, rootkit)."),
}

app = Flask(__name__)


def build_feature_vector(form_values):
    """Turn a dict of {feature: value} into a scaled model-ready row.
    Missing features fall back to their median default."""
    row = {}
    defaults = {f["name"]: f["default"] for f in feature_info["features"]}
    for name in FEATURES:
        if name in form_values and form_values[name] not in ("", None):
            val = form_values[name]
            if name in CATEGORICAL:
                # map original string -> encoded integer
                le = encoders[name]
                sval = str(val)
                if sval in le.classes_:
                    val = int(le.transform([sval])[0])
                else:
                    val = int(le.transform([le.classes_[0]])[0])
            else:
                val = float(val)
        else:
            val = defaults[name]
        row[name] = val
    # DataFrame with column names avoids sklearn feature-name warnings
    X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
    return scaler.transform(X)


@app.route("/")
def index():
    # Build the option lists for categorical dropdowns
    cat_options = {c: sorted(feature_info["categorical_options"][c].keys())
                   for c in CATEGORICAL}
    return render_template(
        "index.html",
        key_features=KEY_FEATURES,
        categorical=CATEGORICAL,
        cat_options=cat_options,
        examples=feature_info["examples"],
        defaults={f["name"]: f["default"] for f in feature_info["features"]},
        best_model=feature_info["best_model"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    X = build_feature_vector(data)

    pred_idx = int(model.predict(X)[0])
    pred_label = TARGET_ENCODER.inverse_transform([pred_idx])[0]

    # Probabilities per class
    proba = model.predict_proba(X)[0]
    probs = {TARGET_ENCODER.inverse_transform([i])[0]: round(float(p) * 100, 2)
             for i, p in enumerate(proba)}
    probs = dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))

    title, desc = CATEGORY_INFO.get(pred_label, (pred_label, ""))
    return jsonify({
        "prediction": pred_label,
        "is_attack": pred_label != "normal",
        "title": title,
        "description": desc,
        "confidence": probs[pred_label],
        "probabilities": probs,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
