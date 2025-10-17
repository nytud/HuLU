import os
import json
import random
import pathlib
from datetime import datetime

import torch


def set_seeds(args) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def read_file(file_path, readlines: bool = False):
    file_path = pathlib.Path(file_path)
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return file.readlines() if readlines else file.read()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e


def read_json(file_path):
    file_path = pathlib.Path(file_path)
    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file: {file_path}") from e


def save_json(data, dir_path, file_name: str) -> None:
    dir_path = pathlib.Path(dir_path)
    file_path = dir_path / file_name
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4, default=str)
    except OSError as e:
        raise OSError(f"Could not save file: {file_path}") from e


def setup_reporting(args) -> None:
    print("Setting up reporting...")

    if args.report_to.lower() == 'wandb':
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb is not installed. Please install it to use wandb reporting.")
        raise ValueError(f"Currently unsupported reporting destination: {args.report_to}")

    elif args.report_to.lower() == 'mlflow':

        try:
            import mlflow
        except ImportError:
            raise ImportError("mlflow is not installed. Please install it to use mlflow reporting.")

        run_name = args.run_name or os.getenv('MLFLOW_RUN_NAME')
        experiment_name = args.experiment_name or os.getenv('MLFLOW_EXPERIMENT_NAME')
        report_uri = args.report_uri or os.getenv('MLFLOW_TRACKING_URI')
        mlflow_username = os.getenv('MLFLOW_TRACKING_USERNAME')
        mlflow_password = os.getenv('MLFLOW_TRACKING_PASSWORD')

        assert report_uri is not None, "MLflow reporting requires a report URI to be set."
        assert run_name is not None, "MLflow reporting requires a run name to be set."
        assert mlflow_username is not None, "MLflow reporting requires a username env var to be set."
        assert mlflow_password is not None, "MLflow reporting requires a password env var to be set."

        mlflow.set_tracking_uri(report_uri)
        mlflow.set_experiment(experiment_name)
        args.reporting = True

    else:
        args.reporting = False


def save_results(args, results) -> None:

    if not args.save_results_path:
        return None

    dir_path = pathlib.Path(args.save_results_path)
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    eval_name = args.model_name.replace('/', '-') # e.g. google/bert-base-uncased -> google-bert-base-uncased
    file_name = f"hulu-{eval_name}-results-{current_time}.json"

    save_json(results, dir_path, file_name)
