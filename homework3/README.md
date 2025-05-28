# 📣 MLflow + Apache Airflow 3.0.1 — Dockerized Dev Environment

This repository sets up **Apache Airflow 3.0.1** and **MLflow** in isolated containers using Docker Compose. It enables you to run ETL workflows and experiment tracking in a unified local setup.

---

## 📆 Project Structure

```
.
├── Dockerfile.airflow        # Custom Airflow Dockerfile
├── Dockerfile.mlflow         # Custom MLflow Dockerfile
├── docker-compose.yml        # Compose file to spin up services
├── dags/                     # Airflow DAGs live here
├── mlruns/                   # MLflow artifact storage
├── requirements.txt          # Shared Python requirements
└── README.md                 # You're here!
```

---

## 🚀 Getting Started

### 1. 📅 Prerequisites

* [Docker](https://www.docker.com/products/docker-desktop)
* [Docker Compose](https://docs.docker.com/compose/)

---

### 2. 🛠️ Setup Steps

#### 🔧 2.1 Add Your Python Libraries

Edit `requirements.txt` to include any additional dependencies needed by either MLflow or Airflow.

#### 📁 2.2 Create Required Folders

```bash
mkdir -p dags mlruns
```

---

### 3. 💪 Build and Start Services

```bash
docker compose up --build
```

---

## 🌐 Services Overview

| Service    | Port   | Description                    |
| ---------- | ------ | ------------------------------ |
| Airflow UI | `8080` | DAG orchestration & scheduling |
| MLflow UI  | `5050` | Experiment tracking dashboard  |

---

## 💾 Persistent Volumes

* Airflow DB volume: `airflow_db`
* MLflow artifacts: `mlruns/` (mounted)
* MLflow DB volume: `mlflow_db`

These ensure your data persists across container restarts.

---

## 🪟 Stopping & Cleaning Up

To stop services:

```bash
docker compose down
```

To remove all volumes as well:

```bash
docker compose down -v
```

---

## 📌 Notes

* Airflow uses SQLite with `SequentialExecutor` by default.
* MLflow uses SQLite for the backend store and `mlruns/` for artifact storage.
* This setup is intended for **local development** only.

---

## 🧠 Useful Links

* [Airflow Docs](https://airflow.apache.org/docs/)
* [MLflow Docs](https://mlflow.org/docs/latest/index.html)
