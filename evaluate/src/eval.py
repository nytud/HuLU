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
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForMultipleChoice, AutoModelForSequenceClassification

import helper
import constants
from preprocess import preprocess


def evaluate(args) -> None:

    logging.info("Evaluation started for tasks: " + ", ".join(args.tasks))
    score_results = {}
    eval_start_time = time.time()

    if args.report_to:
        helper.setup_reporting(args)

    for task_name in args.tasks:

        logging.info(f"Started evaluation for {task_name}.")
        task_start_time = time.time()

        dataset, tokenizer = preprocess(args, task_name)
        results = fine_tune(args, dataset, tokenizer, task_name)
        score_results[task_name] = results

        logging.info(f"Task {task_name} took {time.time() - task_start_time:.3f} seconds.")

    logging.info(f"Evaluation took {time.time() - eval_start_time:.3f} seconds for tasks: {', '.join(args.tasks)}.")

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


def fine_tune(args, dataset, tokenizer, task_name: str) -> Optional[dict]:

    if task_name == constants.COPA:
        model = AutoModelForMultipleChoice.from_pretrained(args.model_name, ignore_mismatched_sizes=True)
    else:
        model_kwargs = {"num_labels": 3} if task_name in [constants.CB, constants.SST] else {"num_labels": 2}
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_kwargs)

    parameters = helper.read_json(args.parameters_path)

    training_args = TrainingArguments(
        output_dir=f"{args.save_results_path}{task_name}/{args.run_name if args.run_name else ''}",
        logging_dir=f"{args.save_results_path}{task_name}/logs",
        run_name=f"{task_name}-{args.run_name if args.run_name else ''}",
        **parameters
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics
    )
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

