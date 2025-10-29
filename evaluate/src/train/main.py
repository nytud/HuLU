import logging

import torch
import torch.multiprocessing as mp

from train.arguments import Arguments
from train.fsdp import FSdpPipeline, run_worker
from train.preprocess import PreprocessPipeline
from train.train import TrainPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run_task_standard(task: str, args: Arguments) -> None:
    """
    Runs the training pipeline in single-process (non-distributed) mode.
    """
    dataset = PreprocessPipeline().preprocess_dataset(args, task)

    pipeline = TrainPipeline(
        hulu_args=args, current_task=task, tokenizer_name=args.tokenizer_name
    )
    pipeline.set_tokenized_datasets(
        train_dataset=dataset["train"],
        dev_dataset=dataset["validation"],
        test_dataset=dataset["test"],
    )

    trained_model = pipeline.training()
    pipeline.create_submission(trained_model)


def run_task_fsdp(task: str, args: Arguments) -> None:
    """
    Runs the training pipeline using FSDP (distributed training).
    This spawns one process per available CUDA device.
    """
    dataset = PreprocessPipeline().preprocess_dataset(args, task)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("FSDP selected but no CUDA devices found.")

    mp.spawn(
        run_worker,
        args=(world_size, args, task, args.tokenizer_name, dataset),
        nprocs=world_size,
        join=True,
    )


def benchmark(args: Arguments) -> None:
    """
    For each task specified in the arguments, run the appropriate training pipeline.
    """
    task_runner = run_task_fsdp if args.use_fsdp else run_task_standard

    for task in args.tasks:
        logging.info(
            "######### Started evaluating %s on task %s", args.model_name, task
        )
        task_runner(task, args)
