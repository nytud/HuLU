from typing import Optional

import time
import logging

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef
)
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForMultipleChoice, AutoModelForSequenceClassification

from src import helper
from src import constants
from src.preprocess import preprocess


def evaluate(args) -> None:

    print("Evaluation started for tasks: " + ", ".join(args.tasks))
    score_results = {}
    eval_start_time = time.time()

    if args.report_to:
        helper.setup_reporting(args)

    for task_name in args.tasks:

        print(f"Started evaluation for {task_name}.")
        task_start_time = time.time()

        dataset, pad_token_id = preprocess(args, task_name)
        results = fine_tune(args, dataset, task_name, pad_token_id)
        score_results[task_name] = results

        print(f"Task {task_name} took {time.time() - task_start_time:.3f} seconds.")
    print(f"Evaluation took {time.time() - eval_start_time:.3f} seconds for tasks: {', '.join(args.tasks)}.")

    helper.save_results(args, score_results)


def compute_metrics(eval_pred):

    logits = torch.tensor(eval_pred.predictions)
    preds = torch.argmax(logits, dim=1)
    labels = torch.tensor(eval_pred.label_ids)

    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    accuracy = accuracy_score(labels_np, preds_np)
    balanced_acc = balanced_accuracy_score(labels_np, preds_np)
    precision = precision_score(labels_np, preds_np, average="macro", zero_division=0)
    recall = recall_score(labels_np, preds_np, average="macro", zero_division=0)
    f1 = f1_score(labels_np, preds_np, average="macro")
    mcc = matthews_corrcoef(labels_np, preds_np)

    # specificity (macro)
    cm = confusion_matrix(labels_np, preds_np, labels=np.unique(labels_np))
    total = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - (tp + fp + fn)
    specificity = float(np.mean(tn / (tn + fp + 1e-8)))

    probs = torch.softmax(logits, dim=1).numpy()

    if len(np.unique(labels_np)) == 2:
        # binary classification, shape: (n_samples,), probs[:, 1] is positive class
        auc = roc_auc_score(labels_np, probs[:, 1])
    else: # multi-class classification, shape (n_samples, n_classes)
        auc = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "mcc": mcc,
        "auc": auc
    }


LORA_TRIAL_PARAM_MAP = {
    "lora_r": "r",
    "lora_alpha": "lora_alpha",
    "lora_dropout": "lora_dropout",
    "lora_target_modules": "target_modules",
}


def lora_hp_space(trial) -> dict:
    return {
        "r": trial.suggest_categorical("lora_r", [4, 8, 16, 32]),
        "lora_alpha": trial.suggest_categorical("lora_alpha", [8, 16, 32, 64]),
        "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.3),
        "target_modules": trial.suggest_categorical("lora_target_modules", [None, "all-linear"]),
    }


def resolve_torch_dtype(parameters: dict) -> Optional[torch.dtype]:
    if parameters.get("bf16"):
        return torch.bfloat16
    elif parameters.get("fp16"):
        return torch.float16
    elif parameters.get("fp32"):
        return torch.float32
    return None


def build_model_init(
    args, task_name: str, lora_base_params: Optional[dict], torch_dtype: Optional[torch.dtype], pad_token_id: Optional[int]
):

    def model_init(trial=None):
        if task_name == constants.COPA:
            model = AutoModelForMultipleChoice.from_pretrained(
                args.model_name, ignore_mismatched_sizes=True, device_map="auto", torch_dtype=torch_dtype
            )
        else:
            model_kwargs = {"num_labels": 3} if task_name in [constants.CB, constants.SST] else {"num_labels": 2}
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name, device_map="auto", torch_dtype=torch_dtype, **model_kwargs
            )
            model.config.pad_token_id = pad_token_id

        if lora_base_params is not None:
            lora_params = dict(lora_base_params)
            if trial is not None:
                lora_params.update(lora_hp_space(trial))
            config = LoraConfig(**lora_params)
            model = get_peft_model(model, config)
            print(f"LoRA parameters added to the model. Parameters: {lora_params}")

        return model

    return model_init


def hp_space(trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 6),
    }


def run_hyperparameter_search(args, trainer, task_name: str, parameters: dict, lora_base_params: Optional[dict]) -> dict:
    try:
        import optuna  # noqa: F401
    except ImportError as e:
        raise ImportError("optuna is required for hyperparameter search. Install it with `pip install optuna`.") from e

    metric_key = trainer.args.metric_for_best_model
    direction = "maximize" if trainer.args.greater_is_better else "minimize"

    logging.info(f"Starting hyperparameter search for {task_name}: {args.hp_trials} trials, optimizing {metric_key}.")

    best_run = trainer.hyperparameter_search(
        direction=direction,
        backend="optuna",
        hp_space=hp_space,
        n_trials=args.hp_trials,
        compute_objective=lambda metrics: metrics[metric_key],
    )
    print(f"Best trial for {task_name}: {best_run.hyperparameters} -> {metric_key}={best_run.objective}")

    best_parameters = dict(parameters)
    best_parameters.update({
        key: value for key, value in best_run.hyperparameters.items()
        if key not in LORA_TRIAL_PARAM_MAP
    })

    if lora_base_params is not None:
        lora_best = dict(lora_base_params)
        for trial_key, lora_key in LORA_TRIAL_PARAM_MAP.items():
            if trial_key in best_run.hyperparameters:
                lora_best[lora_key] = best_run.hyperparameters[trial_key]
        best_parameters["lora"] = lora_best

    helper.save_json(best_parameters, args.save_results_path, f"{task_name}-best-parameters.json")

    return {
        "metric": metric_key,
        "best_objective": best_run.objective,
        "best_hyperparameters": best_run.hyperparameters,
    }


def fine_tune(args, dataset, task_name: str, pad_token_id: Optional[int]) -> Optional[dict]:

    parameters = helper.read_json(args.parameters_path)
    lora_base_params = parameters.pop("lora") if isinstance(parameters.get("lora"), dict) else None
    torch_dtype = resolve_torch_dtype(parameters)

    model_init = build_model_init(args, task_name, lora_base_params, torch_dtype, pad_token_id)

    training_args = TrainingArguments(
        output_dir=f"{args.save_results_path}{task_name}/{args.run_name if args.run_name else ''}",
        logging_dir=f"{args.save_results_path}{task_name}/logs",
        run_name=f"{task_name}-{args.run_name if args.run_name else ''}",
        **parameters
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics
    )

    if args.hp_search:
        return run_hyperparameter_search(args, trainer, task_name, parameters, lora_base_params)

    results = trainer.train()

    if args.eval_test:
        # TODO trainer.evaluate() runs a newly created mlflow run for test set, fix later
        results = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
        return results

    results = trainer.predict(dataset["test"])

    logits = results.predictions
    predictions = np.argmax(logits, axis=1).tolist()

    helper.save_results_for_submission(args, task_name, predictions)

    return None
