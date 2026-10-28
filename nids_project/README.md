# Network Intrusion Detection System — ML Web App

A machine-learning web application that classifies a network connection as
**normal traffic** or one of four attack categories — **DoS, Probe, R2L, U2R** —
using the **NSL-KDD** benchmark dataset.

This is the implementation for the AI term project *"Deploy an ML Web App"*.

---

## 1. Project structure

```
nids_project/
├── data/
│   ├── KDDTrain.txt          # NSL-KDD training set (125,973 records)
│   └── KDDTest.txt           # NSL-KDD test set (22,544 records)
├── models/                   # created by train_model.py
│   ├── best_model.joblib     # trained Random Forest
│   ├── scaler.joblib         # fitted StandardScaler
│   ├── encoders.joblib       # label encoders (protocol, service, flag, target)
│   ├── feature_info.joblib   # feature metadata + example rows for the UI
│   └── metrics.json          # accuracy / F1 for all three models
├── eda/                      # EDA plots + confusion matrix + comparison chart
├── templates/
│   └── index.html            # web interface
├── train_model.py            # data prep + EDA + training pipeline
├── app.py                    # Flask web server
└── requirements.txt
```

## 2. Setup

```bash
# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## 3. Train the models (already done — re-run to reproduce)

```bash
python train_model.py
```

This loads NSL-KDD, maps the 39 specific attacks to 5 categories, runs EDA
(saves plots to `eda/`), trains **Logistic Regression**, **Decision Tree**, and
**Random Forest**, evaluates them with 5-fold cross-validation and on the
official test set, and saves the best model to `models/`.

## 4. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

- Click a **Quick Load** chip (DoS / Probe / R2L / U2R / normal) to load a real
  captured connection from the dataset, then it is analyzed automatically.
- Or edit the connection-feature fields manually and press **Analyze
  Connection**.
- The right panel shows the verdict (green = safe, red = threat) and the
  model's confidence for every class.

## 5. Results summary

| Model | CV accuracy (train) | Test accuracy (KDDTest+) | Test weighted-F1 |
|-------|--------------------:|-------------------------:|-----------------:|
| Logistic Regression | 0.966 | 0.762 | 0.714 |
| Decision Tree       | 0.998 | 0.753 | 0.709 |
| **Random Forest** (deployed) | **0.999** | 0.747 | 0.705 |

The high cross-validation accuracy shows the models learn the traffic patterns
very well. The lower score on the official test set is expected and
well-documented for NSL-KDD: the test set deliberately includes attack types
that never appear in training, which measures how the model generalizes to
**novel, unseen attacks** — a realistic challenge for any real-world IDS.

## 6. Deployment notes

`app.py` runs Flask's development server. To deploy publicly, use a production
WSGI server, e.g.:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

Free hosting options that work well for this app: **Render**, **Railway**, or
**PythonAnywhere** (push the repo, set the start command to the gunicorn line
above, and make sure the `models/` and `data/` folders are included).

---

*Dataset: NSL-KDD (Tavallaee et al., 2009), an improved version of the
KDD Cup 1999 dataset. For educational use.*
