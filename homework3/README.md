# Airflow + MLflow Setup Guide

This guide walks you through setting up **Apache Airflow** with **MLflow** using Docker Compose.

---

## 📁 Project Structure

```
.
├── config/
├── dags/
├── logs/
├── mlflow/
├── plugins/
├── docker-compose.yaml
├── Dockerfile.airflow
├── Dockerfile.mlflow
├── README.md
├── requirements.txt
└── Makefile
```

---

## 🚀 Step-by-Step Setup

### 1. ✅ Build the Docker Images

```bash
make build
```

### 2. 📆 Start the Airflow + MLflow Services

```bash
make up
```

This will spin up the following services:

* `airflow-apiserver` (UI available at [http://localhost:8080](http://localhost:8080))
* `airflow-scheduler`
* `airflow-dag-processor`
* `postgres` (Airflow metadata DB)
* `mlflow` (UI available at [http://localhost:5050](http://localhost:5050))

### 3. 🪄 Initialize Airflow (first time only)

```bash
make airflow-init
```

> ⚠️ Make sure `airflow-init` finishes successfully before running DAGs.

---

## 🛠️ Creating and Running DAGs

1. Place your DAG Python files inside the `dags/` directory.

2. Access the Airflow UI at [http://localhost:8080](http://localhost:8080)

   * Username: `airflow`
   * Password: `airflow`

3. Trigger your DAG manually or wait for it to be scheduled.

---

## 📡 Tracking Models in MLflow

1. After the model is trained in the DAG, it will be logged to MLflow.
2. Open MLflow UI: [http://localhost:5050](http://localhost:5050)
3. Navigate through experiments, runs, and models.

---

## 📄 Makefile Commands

```Makefile
# Build all docker images
docker-compose build

# Start all containers
docker-compose up -d

# Initialize Airflow
docker-compose run --rm airflow-init

# Stop all services
docker-compose down

# Remove all volumes and services
docker-compose down -v

# View logs
docker-compose logs -f

# Open Airflow URL
echo "http://localhost:8080"

# Open MLflow URL
echo "http://localhost:5050"
```

Wrap these commands in a `Makefile` like so:

```make
build:
	docker compose build

up:
	docker compose up -d

airflow-init:
	docker compose run --rm airflow-init

down:
	docker compose down

clean:
	docker compose down -v

deploy: build up airflow-init

logs:
	docker compose logs -f

open-airflow:
	echo "http://localhost:8080"

open-mlflow:
	echo "http://localhost:5050"
```

---

## 🛀 Clean-Up Tips

If you encounter memory issues (especially on 8GB machines):

* Lower memory limits in `docker-compose.yaml` (`mem_limit` fields)
* Process fewer rows in your training DAG
* Restart Docker after stopping services

---

## ✅ Quick Checks

```bash
docker compose ps             # Check if all services are healthy
curl localhost:8080           # Check if Airflow UI is up
curl localhost:5050/health    # Check if MLflow is healthy
```
