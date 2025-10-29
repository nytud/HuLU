CONJUNCTIONS = {
    "cause": "mert",
    "effect": "ezért",
}

SST_LABELS = {
    "positive": 1,
    "negative": 2,
    "neutral": 0,
}

CB_LABELS = {
    "entailment": 1,
    "contradiction": 2,
    "neutral": 0,
}

LABELS = {}
LABELS.update(SST_LABELS)
LABELS.update(CB_LABELS)

HULU_DATASETS = {
    "sst": "NYTK/HuSST",
    "rte": "NYTK/HuRTE",
    "wnli": "NYTK/HuWNLI",
    "cola": "NYTK/HuCOLA",
    "copa": "NYTK/HuCoPA",
    "cb": "NYTK/HuCommitmentBank",
}

IRRELEVANT_COLUMNS = {
    "cola": ["id", "sentence"],
    "sst": ["id", "sentence"],
    "wnli": ["id", "sentence1", "sentence2", "orig_id"],
    "rte": ["id", "premise", "hypothesis"],
    "cb": ["id", "premise", "hypothesis"],
    "copa": ["id", "choice1", "choice2", "question"],
}

RELEVANT_COLUMNS = {
    "cola": ["sentence"],
    "sst": ["sentence"],
    "wnli": ["sentence1", "sentence1"],
    "rte": ["premise", "hypothesis"],
    "cb": ["premise", "hypothesis"],
    "copa": ["premise", "choice1", "choice2", "question"],
}

TOKENIZER_PARAMETERS = {
    "cola": {
        "truncation": True,
        "padding": "max_length",
        "max_length": -1,
        "add_special_tokens": True,
    },
    "sst": {
        "truncation": True,
        "padding": "max_length",
        "max_length": -1,
        "add_special_tokens": True,
    },
    "wnli": {
        "truncation": True,
        "padding": "max_length",
        "max_length": -1,
    },
    "rte": {
        "truncation": True,
        "padding": "max_length",
        "max_length": -1,
    },
    "cb": {"truncation": True, "padding": "max_length", "max_length": 512},
    "copa": {"truncation": True, "padding": "max_length", "max_length": 256},
}
