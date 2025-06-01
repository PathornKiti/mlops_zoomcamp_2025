from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import requests
import os
import logging
import pickle
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor


URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
LOCAL_FILE = "/tmp/yellow_tripdata_2023-03.parquet"


def task_download_data():
    if not os.path.exists(LOCAL_FILE):
        response = requests.get(URL)
        with open(LOCAL_FILE, 'wb') as f:
            f.write(response.content)

    df = pd.read_parquet(LOCAL_FILE)
    logging.info(f"✅ Loaded {len(df):,} records")


def task_prepare_data():
    df = pd.read_parquet(LOCAL_FILE)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df.to_parquet("/tmp/processed.parquet", index=False)
    logging.info(f"✅ Prepared dataframe with {len(df):,} records")

    try:
        os.remove(LOCAL_FILE)
        logging.info(f"🧹 Removed raw file: {LOCAL_FILE}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to delete {LOCAL_FILE}: {e}")


def task_train_model():
    logging.info("🚀 task_train_model started")

    try:
        df = pd.read_parquet("/tmp/processed.parquet")
        logging.info(f"✅ Loaded processed parquet file with shape: {df.shape}")
    except Exception as e:
        logging.error("❌ Failed while reading parquet file", exc_info=e)
        return

    try:
        # Sample ~10% randomly
        df_sampled = df.sample(frac=0.1, random_state=42)
        logging.info(f"✅ Random sampling complete, new shape: {df_sampled.shape}")

        categorical = ['PULocationID', 'DOLocationID']
        train_dicts = df_sampled[categorical].to_dict(orient='records')
        y_train = df_sampled['duration']
        logging.info("✅ Successfully converted to dicts and extracted y")
    except Exception as e:
        logging.error("❌ Failed during sampling or dict conversion", exc_info=e)
        return

    try:
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)
        logging.info(f"✅ DictVectorizer fit complete, X_train shape: {X_train.shape}")
    except Exception as e:
        logging.error("❌ Failed during DictVectorizer.fit_transform", exc_info=e)
        return

    try:
        model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
        model.fit(X_train, y_train)
        logging.info(f"✅ Model training complete. Intercept: {model.intercept_[0]:.2f}")
    except Exception as e:
        logging.error("❌ Failed during model.fit", exc_info=e)
        return

    try:
        with open("/tmp/dv.pkl", "wb") as f_out:
            pickle.dump(dv, f_out)
        with open("/tmp/model.pkl", "wb") as f_out:
            pickle.dump(model, f_out)
        logging.info("✅ Model and vectorizer saved to disk")
    except Exception as e:
        logging.error("❌ Failed during model/vectorizer serialization", exc_info=e)




def task_register_model():
    mlflow.set_tracking_uri("http://mlflow:5050")
    mlflow.set_experiment("yellow-taxi")

    with mlflow.start_run():
        with open("/tmp/dv.pkl", "rb") as f_in:
            dv = pickle.load(f_in)
        with open("/tmp/model.pkl", "rb") as f_in:
            model = pickle.load(f_in)

        mlflow.sklearn.log_model(model, "model")
        model_path = mlflow.get_artifact_uri("model") + "/MLmodel"

        size = os.path.getsize(model_path)
        logging.info(f"✅ Logged model MLModel file size: {size:,} bytes")


with DAG(
    dag_id="yellow_taxi_pipeline_sgd_reg",
    start_date=datetime(2023, 3, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "yellow-taxi"],
) as dag:

    t1 = PythonOperator(
        task_id="load_data",
        python_callable=task_download_data,
    )

    t2 = PythonOperator(
        task_id="prepare_data",
        python_callable=task_prepare_data,
    )

    t3 = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    t4 = PythonOperator(
        task_id="register_model",
        python_callable=task_register_model,
    )

    t1 >> t2 >> t3 >> t4
