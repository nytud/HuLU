# HuLU evaluate

`hulu_evaluate` is a library for evaluating and training language models on Hungarian tasks within the [HuLU benchmark](https://hulu.nytud.hu/). It includes support for fine-tuning with LoRA, official evaluation scripts, and provides a leaderboard for benchmarking.

## Features

- **Training on multiple tasks**: Supports tasks like "CoLA", "RTE", "WNLI", "CB", "SST", and "COPA".
- **LoRA (Low-Rank Adaptation)**: Fine-tune models with reduced compute requirements.
- **Official training scripts**: Provided for training and evaluation.
- **HuLU Leaderboard**: Submit your results to the HuLU leaderboard.
- **Fully Sharded Data Parallel**: Shard and paralellize training of models

## Installation

Installation via pypi

```bash
pip install hulu-evaluate
```

To install the `hulu_evaluate` library for testing and development, clone the repository and install dependencies:

```bash
pip install git+https://git.nlp.nytud.hu/DLT/HuLU-evaluate.git
```

## Usage

Command-Line Interface (CLI)
The CLI provides commands for configuration, training, and submission.

Setting Up the CLI
The CLI entry point can be called with:

```bash
hulu-evaluate <command> [<args>]
```

### Commands include

## Train a Model: Train on HuLU tasks using the train command

```bash
hulu-evaluate --model-name <MODEL_NAME> --output-dir <OUTPUT_DIR> --train-epochs 6 --train-batch 8
```

You can submit your results on the hulu.nytud.hu webpage. The results are created by default in the "HuluFinetune" directory.

## Programmatic Usage

You can also integrate hulu_evaluate functionality directly in Python scripts.

Training

```python
from hulu_evaluate.train.main import Main
from hulu_evaluate.hulu_arguments.train_arguments import Arguments

# Define training arguments
arguments = Arguments(
    model_name="my_model",
    train_epochs=6,
    tasks=["cola", "rte"],
)

# Train the model
Main(arguments)
```

## Command Reference

### Training Command

train - Trains a model on HuLU tasks with specified parameters.

```bash
hulu-evaluate --model-name <MODEL_NAME> --output-dir <OUTPUT_DIR> --train-epochs <EPOCHS>
--model-name: Name of the model to train.
--train-epochs: Number of epochs.
--use-lora: Enable LoRA fine-tuning (optional).
--config-file: Path to config JSON file
--output-dir: Output directory
--model-name: Model name
--tokenizer-name: Tokenizer name (defaults to model_name)
--train-epochs:  default=6 Number of training epochs
--train-batch: default=8 Training batch size
--train-lr: default=2e-05 Learning rate
--train-warmup: default=0 Warmup steps
--train-maxlen: default=256 Max sequence length
--train-seed: default=42 Random seed
--precision: default="fp32" Precision (e.g., fp16 or fp32)
--use-lora: default=False Use LoRA for training
--lora-r: default=8 LoRA r parameter
--lora-alpha: default=16 LoRA alpha parameter
--lora-dropout: default=0.1 LoRA dropout rate
--tasks default=["cola", "rte", "wnli", "cb", "sst"] List of tasks to train on
--use-fsdp: default=False Using FSDP
--gradient_accumulation_steps: default=1 Set steps for gradient accumulation
```

## The HuLU Benchmark

HuLU (Hungarian Language Understanding Benchmark Kit) was created on the basis of the GLUE and SuperGLUE benchmark databases. The main purpose of HuLU is to enable a standardized evaluation of neural language models in a simple way while also enabling multi-perspective analysis. It also compares the performance of various LMs on various tasks and shares the results with LT professionals in tabular and graphical formats.
**The tasks and their precise presentation is available on the official [HuLU page](https://hulu.nytud.hu/tasks)**

## Submitting the Results to the HuLU Leaderboard

The official [HuLU page](https://hulu.nytud.hu/tasks) offers an opportunity to validate your results from the training procedure.
The path selected with the `--output-path` or it will default to the same folder the process is ran.
The created folder is named `finetune_results` and will contain the predictions of the model for the given task e.g.: `cola_predicted_labels.json`. To submit your results you will need this file.

Navigate to the webpage https://hulu.nytud.hu/. Sign up at the top right corner - use google authentication. After successful authorization select `Submission` fill the form and upload the newly created predictions for the given task - one at a time.

After submission you will be informed when your results are presented on the site. Please be patient as all submissions are processed manually.

## Citing

```bibtex
@inproceedings{hatvani2024hulu,
  author    = {Péter Hatvani, Kristóf Varga and Zijian Győző Yang},
  title     = {Evaluation Library for the Hungarian Language Understanding Benchmark (HuLU)},
  booktitle = {Proceedings of the 21th Hungarian Computational Linguistics Conference},
  year      = {2024},
  address   = {Hungary},
  publisher = {Szegedi Tudományegyetem TTIK, Informatikai Intézet},
  note      = {Affiliations: PPKE Doctoral School of Linguistics, HUN-REN Hungarian Research Center for Linguistics},
  url       = {https://hatvanipeter.hu/},
  email     = {hatvani9823@gmail.com, varga.kristof@nytud.hun-ren.hu, yang.zijian.gyozo@nytud.hun-ren.hu}
}
```

### Contributing

Contributions are welcome! Please submit issues or pull requests to improve hulu_evaluate.

### License

This project is licensed under the Apache License.
