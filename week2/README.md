# 🚀 MLflow Setup with Poetry

This guide provides a complete walkthrough for setting up [MLflow](https://mlflow.org/) using [Poetry](https://python-poetry.org/) as your Python dependency manager. It includes initializing a project, installing MLflow, configuring experiment tracking, and running the MLflow UI either with or without a persistent SQLite backend.

---

## 📦 Prerequisites

- Python 3.8 or higher
- Poetry installed ([Installation Guide](https://python-poetry.org/docs/#installation))
- Git (optional, but recommended)

---

## 🛠️ Step 1: Create a New Poetry Project

```bash
poetry new mlflow_project
cd mlflow_project
```

This creates a basic project structure with `pyproject.toml`.

---

## 📚 Step 2: Add Dependencies

Install MLflow and common ML packages:

```bash
poetry add mlflow scikit-learn pandas
```

---

## 🐍 Step 3: Activate the Poetry Environment

```bash
poetry shell
```

This ensures your environment is isolated.

---

## 💾 Step 4: Start MLflow Tracking Server

### ✅ Option 1: Persistent Backend with SQLite

```bash
mkdir mlruns
poetry run mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

### 🟡 Option 2: Ephemeral In-Memory Backend (No DB)

If you don’t provide a `--backend-store-uri`, MLflow uses a default local file-based backend and stores metadata in the `mlruns/` folder:

```bash
poetry run mlflow ui
```

This will use the default file path `./mlruns` for logging experiments and not persist metadata to a database.

The MLflow UI will be accessible at [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧪 Step 5: Log Your First Experiment

Create a file `train.py`:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

mlflow.sklearn.autolog()

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestRegressor()
model.fit(X_train, y_train)
```

Run it:

```bash
poetry run python train.py
```

Check the MLflow UI to see logged results.

---

## 🧱 Project Structure Example

```
mlflow_project/
├── mlflow_project/
│   └── __init__.py
├── train.py
├── mlruns/
├── mlflow.db (if SQLite used)
├── pyproject.toml
├── poetry.lock
└── README.md
```

---

