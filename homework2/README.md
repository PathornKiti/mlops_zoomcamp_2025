# MLflow Homework Solutions (MLOps Zoomcamp 2025)

## ✅ Q1. Install MLflow & Check Version

Install MLflow using Poetry:

```bash
poetry add mlflow
```

Check the installed version:

```bash
poetry run mlflow --version
```

📌 Example Output:

```bash
mlflow, version 2.22.0
```

👉 **Answer for Q1**: Version shown by the command (e.g., 2.22.0)

---

## ✅ Q2. Download & Preprocess Taxi Data

**Download Data:**

```bash
poetry run python homework2/load_data.py --color green --year 2023 --months 1 2 3
```

This will save files in:

```bash
homework2/data/
```

**Preprocess Data:**

```bash
poetry add scikit-learn pandas pyarrow
poetry run python homework2/preprocess_data.py --raw_data_path homework2/data --dest_path homework2/output
```

**Expected Files in output Folder:**

* train.pkl
* val.pkl
* test.pkl
* dv.pkl

👉 **Answer for Q2**: 4 files saved

---

## ✅ Q3. Train Model with MLflow Autologging

**Step 1: Start MLflow Tracking Server**

```bash
poetry run mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 127.0.0.1 \
    --port 5000
```

**Step 2: Edit train.py**

Add at the top of the file:

```python
import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("green-taxi-Jan-Mar-2023")
mlflow.sklearn.autolog()
```

Wrap the training code with:

```python
with mlflow.start_run():
    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Validation RMSE: {rmse:.2f}")
```

**Step 3: Run Training Script**

```bash
poetry run python homework2/train.py --data_path homework2/output
```

**Step 4: Launch MLflow UI**

```bash
poetry run mlflow ui
```

Open the UI: [http://127.0.0.1:5000](http://127.0.0.1:5000)

Check the experiment: green-taxi-Jan-Mar-2023

Look at the logged parameters.

👉 **Answer for Q3**: min\_samples\_split = 2

---

## ✅ Q4. Launch MLflow Tracking Server Locally

You need to pass both --backend-store-uri and --default-artifact-root.

**Correct Command:**

```bash
poetry run mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 127.0.0.1 \
    --port 5000
```

* **--backend-store-uri**: SQLite DB to store experiment metadata.
* **--default-artifact-root**: Folder to save artifacts (models, metrics, etc.).

👉 **Answer for Q4**: default-artifact-root

---

## ✅ Q5. Tune Model Hyperparameters

We tuned the hyperparameters of RandomForestRegressor using Hyperopt via `hpo.py`. The following adjustments were made:

* Wrapped the objective function with `mlflow.start_run()`.
* Logged hyperparameters with `mlflow.log_params(params)`.
* Logged validation RMSE with `mlflow.log_metric("rmse", rmse)`.

The optimization was executed with:

```bash
poetry run python homework2/hpo.py --data_path homework2/output
```

Results were explored in MLflow UI at [http://127.0.0.1:5000](http://127.0.0.1:5000) under the experiment `random-forest-hyperopt`.

After sorting by RMSE in ascending order, the **best validation RMSE** obtained was:

👉 **Answer for Q5**: 5.335

---

## ✅ Q6. Promote the Best Model to the Model Registry

We used `register_model.py` to evaluate the top 5 models from the hyperparameter optimization on the test set (March 2023 data).

### Adjustments made:

* Retrieved top 5 runs using `search_runs` from `MlflowClient`.
* Evaluated each model and logged `val_rmse` and `test_rmse`.
* Selected the best run with the lowest `test_rmse` using `search_runs` ordered by `metrics.test_rmse ASC`.
* Registered the best model using `mlflow.register_model()`.

### Code Addition:

```python
# Select best model
best_run = client.search_runs(
    experiment_ids=experiment.experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=1,
    order_by=["metrics.test_rmse ASC"]
)[0]

# Register best model
model_uri = f"runs:/{best_run.info.run_id}/model"
mlflow.register_model(model_uri=model_uri, name="random-forest-best-model")
```

### Run Command:

```bash
poetry run python homework2/register_model.py --data_path homework2/output --top_n 5
```

After this, we explored the new experiment `random-forest-best-models` and found the **best test RMSE** was:

👉 **Answer for Q6**: 5.567

---

## ✅ Final Summary

| Question | Answer                            |
| -------- | --------------------------------- |
| Q1       | Version shown by mlflow --version |
| Q2       | 4                                 |
| Q3       | min\_samples\_split = 2           |
| Q4       | default-artifact-root             |
| Q5       | 5.335                             |
| Q6       | 5.567                             |

---

