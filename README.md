# HuLU
[Hungarian Language Understanding Benchmark Kit](https://hulu.nytud.hu/)


This repository contains the databases included in HuLU, the Hungarian Language Understanding Benchmark Kit developed, maintained and updated in the Language Technology Research Group of the Hungarian Research Centre for Linguistics.

Currently (11/07/2024) six corpora are available to download and to test the models on.

- **HuCOLA** (Hungarian Corpus of Linguistic Acceptability) contains 9 076 Hungarian sentences labeled for their acceptability/grammaticality (0/1). The sentences were collected by two human annotators from three linguistic books. Each sentence was annotated by four human annotators. The final label of the sentence is the one assigned by the majority of the annotators. The proportion of train, validation and test sets is 80% (7 276 sentences), 10% (900 sentences) and 10% (900 sentences), respectively.
- **HuCoPa** (Hungarian Choice of Plausible Alternatives Corpus) contains 1,000 instances. Each instance is composed of a premise and two alternatives. The task is to select the alternative that describes a situation standing in causal relation to the situation described by the premise. The corpus was created by translating and re-annotating the original English CoPA corpus. The train, validation, and test sets contain 400, 100 and 500 instances, respectively.
- **HuRC** (Hungarian Reading Comprehension Dataset) contains 80,621 instances. Each instance is composed of a passage and a cloze-style query with a masked entity. The task is to select the named entity that is being masked in the query. The data was collected from the online news of Népszabadság online (nol.hu).
- **HuSST** (Hungarian version of the Stanford Sentiment Treebank) contains 11 683 sentences. Each sentence is annotated for its sentiment on a three-point scale. The corpus was created by translating and re-annotating the full sentences of the SST. The train, validation, and test sets contain 9 347, 1 168, and 1 168 sentences, respectively.
- **HuWNLI** is a Hungarian dataset of anaphora resolution, designed as a sentence pair classification task of natural language inference. Its base, the HuWS corpus was created by translating and manually curating the original English Winograd schemata. The NLI format was created by replacing the ambiguous pronoun with each possible referent in the schemata. We extended the set of sentence pairs derived from the schemata by the translation of the sentence pairs that build up the WNLI dataset of GLUE. The data is distributed in three splits: training set (562), development set (59), and test set (134).
- **HuCB** (Hungarian CommitmentBank) consists of short text fragments in which at least one sentence contains a subordinating clause, which is syntactically subordinated to a logical inference-canceling operator. In the database, the premise is the complete text fragment and the hypothesis is the embedded tag clause. In the inference task, it is necessary to decide to what extent the author of the text is committed to the truth of the subordinate clause. The corpus consists of a training, a validation, and a test set (of 250, 103, and 250 examples, respectively).


## Evaluation

An evaluation library / script is provided for fine-tuning and benchmarking language models on Hungarian tasks within the [HuLU benchmark](https://hulu.nytud.hu/).

It provides a unified CLI tool for running experiments across all six tasks, with support for Low-Rank Adaptation (LoRA), submission-ready outputs for the HuLU leaderboard, and multi-task training (i.e. training on more than one task). The library is configurable via parameters in a JSON file (see `parameters.json`) and compatible with HuggingFace's `transformers` library. For each task, the following evaluation metrics are computed: accuracy, balanced accuracy, precision, recall, F1 score, ROC-AUC, specificity and Matthews correlation coefficient (MCC).

### Installation

Installation via pypi

```bash
pip install hulu
```

To install the library for testing and development, clone the repository and install dependencies:

```bash
git clone git@github.com:nytud/HuLU.git
cd evaluate/
pip install .
```

### Usage

The CLI provides a unified interface for configuring and running HuLU model evaluation and fine-tuning.


```bash
hulu <command> [<args>]
```

Below are the supported command-line arguments.

| Option | Description |
|--------|-------------|
| `--model-name <STR>` | **(Required)** Name or path of the pretrained model to load. |
| `--parameters-path <PATH>` | **(Required)** Path to the JSON configuration file containing training hyperparameters (e.g., `parameters.json`). |
| `--tokenizer-name <STR>` | Name or path of the tokenizer. Defaults to the model’s tokenizer if not specified. |
| `--tasks <LIST>` | List of HuLU tasks to run. Choices: `hucola`, `hurte`, `huwnli`, `hucommitmentbank`, `husst`, `hucopa`. |
| `--eval-test <true/false>` | Whether to evaluate on the test set after training. Default: `false`. |
| `--report-to <STR>` | Reporting backend (e.g., `wandb`, `mlflow`). |
| `--report-uri <STR>` | URI for the selected reporting backend. |
| `--experiment-name <STR>` | Experiment/project name used for reporting. Default: `hulu-finetune`. |
| `--run-name <STR>` | Custom run name used for logging and reporting. |
| `--save-results-path <PATH>` | Directory to save prediction outputs and logs. Default: `./results/`. |

---

#### Example Usage

```bash
hulu --model-name distilbert/distilbert-base-uncased --tasks hucola husst --parameters ./parameters.json --experiment-name test-experiment --eval-test false
```

### Submitting the Results to the HuLU Leaderboard

The official [HuLU page](https://hulu.nytud.hu/tasks) allows you to validate the results of your training procedure. The output path is determined by the `--save-results-path` option; if it is not provided, a `results` directory is created in the working directory. This folder contains the model’s predictions for each task, and these files are required for submission.

Navigate to the webpage https://hulu.nytud.hu/ and sign up. After successful authorization, select `Submission` fill the form and upload the newly created predictions for the given task - one at a time.

## Citation

If you use these resources or any part of its documentation, please refer to:

Noémi Ligeti-Nagy, Gergő Ferenczi, Enikő Héja, László János Laki, Noémi Vadász, Zijian Győző Yang, and Tamás Váradi. 2024. HuLU: Hungarian Language Understanding Benchmark Kit. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 8360–8371, Torino, Italia. ELRA and ICCL.

```bibtex
@inproceedings{ligeti-nagy-etal-2024-hulu-hungarian,
    title = "{H}u{LU}: {H}ungarian Language Understanding Benchmark Kit",
    author = "Ligeti-Nagy, Noémi  and
      Ferenczi, Gergő  and
      Héja, Enikő  and
      Laki, László János  and
      Vadász Noémi  and
      Yang, Zijian Győző  and
      Váradi Tamás",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.733",
    pages = "8360--8371",
}
```

and to any other references listed in the readme files of the individual corpora.

If you use the evaluation scripts / library, please refer to:

```bibtex
@inproceedings{hatvani2024hulu,
  author    = {Péter Hatvani, Kristóf Varga and Zijian Győző Yang},
  title     = {Evaluation Library for the Hungarian Language Understanding Benchmark (HuLU)},
  booktitle = {Proceedings of the 21th Hungarian Computational Linguistics Conference},
  year      = {2024},
  address   = {Hungary},
  publisher = {Szegedi Tudományegyetem TTIK, Informatikai Intézet},
  note      = {Affiliations: PPKE Doctoral School of Linguistics, HUN-REN Hungarian Research Center for Linguistics},
  email     = {hatvani9823@gmail.com, varga.kristof@nytud.hun-ren.hu, yang.zijian.gyozo@nytud.hun-ren.hu}
}
```

This evaluation scripts (only!) are licensed under the Apache License.