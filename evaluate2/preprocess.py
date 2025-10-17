import os
import pathlib
import logging
from functools import partial

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer

import constants


def preprocess(args, task_name: str):

    dataset = prepare_datasets(args, task_name)

    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer = load_tokenizer(args)
    tokenizer_params = constants.TOKENIZER_PARAMETERS[task_name]

    process_fn = partial(
        preprocess_fn, tokenizer=tokenizer, tokenizer_params=tokenizer_params, task_name=task_name
    )
    remove_columns = constants.IRRELEVANT_COLUMNS.get(task_name)

    dataset = dataset.map(process_fn, remove_columns=remove_columns)
    logging.info(dataset)

    return dataset, tokenizer


def load_tokenizer(args):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file = args.tokenizer_name,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    logging.info(f"Loaded tokenizer from {args.tokenizer_name}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024 # TODO make configurable
    return tokenizer


def preprocess_fn(examples, tokenizer, tokenizer_params: dict, task_name: str):

    label = constants.LABELS.get(examples.get("label"), examples.get("label"))

    if task_name == constants.COPA:
        tokenized = preprocess_fn_copa(examples, tokenizer, tokenizer_params, task_name)
        return {**tokenized, "label": label}

    inputs = tuple(examples[col_name] for col_name in constants.RELEVANT_COLUMNS[task_name]),
    tokenized = tokenizer(*inputs, **tokenizer_params)
    return {**tokenized, "label": label}


def preprocess_fn_copa(examples, tokenizer, tokenizer_params: dict, task_name: str):

    premise = examples.get("premise")
    choice1 = examples.get("choice1")
    choice2 = examples.get("choice2")
    question = examples.get("question")

    choices = [choice1, choice2]
    input_strings = [
        f"{premise} {constants.CONJUNCTIONS[question]} {choice}" for choice in choices
    ]

    tokenized_choices = [
        tokenizer(
            input_str,
            truncation=tokenizer_params["truncation"],
            max_length=tokenizer_params["max_length"],
            padding=tokenizer_params["padding"],
        )
        for input_str in input_strings
    ]

    input_ids = torch.tensor([x["input_ids"] for x in tokenized_choices])
    attention_mask = torch.tensor([x["attention_mask"] for x in tokenized_choices])

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def prepare_datasets(args, task_name: str):

    dataset_name = constants.HULU_DATASETS[task_name]

    datasets_path = os.getenv('DATASETS')
    if args.eval_test and datasets_path is None:
        raise ValueError(
            "Local 'DATASETS' env var must be set to evaluate on test set. "
            "Please use Huggingface hub datasets if you don't have the test set and"
            "submit the results through https://hulu.nytud.hu/"
        )
    if datasets_path is None:
        logging.info(
            f"Loading dataset {dataset_name} from Huggingface hub. Test set not available."
            "Please use Huggingface hub datasets and submit the results through https://hulu.nytud.hu/"
        )
        dataset = load_dataset(dataset_name)
    else:
        logging.info(f"Loading dataset {task_name} from local path: {datasets_path}")
        datasets_path = pathlib.Path(datasets_path)
        task_dataset_path = datasets_path / task_name
        assert task_dataset_path.exists(), f"Dataset path {task_dataset_path} does not exist."
        dataset = load_dataset("json", data_files={
            "train": str(task_dataset_path / "train.json"),
            "validation": str(task_dataset_path / "val.json"),
            "test": str(task_dataset_path / "test.json")
        })
        logging.info(f"Loaded dataset files from {task_dataset_path}")

    return dataset