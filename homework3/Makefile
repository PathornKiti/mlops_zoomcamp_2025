# Makefile for running Airflow + MLflow project

# Variables
PROJECT_NAME=homework3
COMPOSE=docker compose

# Build all custom images
default: build

.PHONY: build
build:
	$(COMPOSE) build

.PHONY: up
up:
	$(COMPOSE) up -d

.PHONY: down
down:
	$(COMPOSE) down

.PHONY: logs
logs:
	$(COMPOSE) logs -f

.PHONY: ps
ps:
	$(COMPOSE) ps

.PHONY: restart
restart:
	$(MAKE) down
	$(MAKE) up

.PHONY: airflow-init
airflow-init:
	$(COMPOSE) run --rm airflow-init

.PHONY: clean
clean:
	$(COMPOSE) down -v --remove-orphans
	rm -rf logs/* mlflow/*

.PHONY: reset
reset:
	$(MAKE) clean
	$(MAKE) build
	$(MAKE) up

.PHONY: open-airflow
open-airflow:
	echo "Airflow: http://localhost:8080 (user: airflow / airflow)"

.PHONY: open-mlflow
open-mlflow:
	echo "MLflow: http://localhost:5050"
