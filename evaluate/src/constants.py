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
    "husst": "NYTK/HuSST",
    "hurte": "NYTK/HuRTE",
    "huwnli": "NYTK/HuWNLI",
    "hucola": "NYTK/HuCOLA",
    "hucopa": "NYTK/HuCoPA",
    "hucommitmentbank": "NYTK/HuCommitmentBank",
}

IRRELEVANT_COLUMNS = {
    "hucola": ["id", "sentence"],
    "husst": ["id", "sentence"],
    "huwnli": ["id", "sentence1", "sentence2", "orig_id"],
    "hurte": ["id", "premise", "hypothesis"],
    "hucommitmentbank": ["id", "premise", "hypothesis"],
    "hucopa": ["id", "choice1", "choice2", "question"],
}

RELEVANT_COLUMNS = {
    "hucola": ["sentence"],
    "husst": ["sentence"],
    "huwnli": ["sentence1", "sentence1"],
    "hurte": ["premise", "hypothesis"],
    "hucommitmentbank": ["premise", "hypothesis"],
    "hucopa": ["premise", "choice1", "choice2", "question"],
}

TOKENIZER_PARAMETERS = {
    "huwnli": { "truncation": True, "padding": "max_length" },
    "hurte":  { "truncation": True, "padding": "max_length" },
    "hucommitmentbank":   { "truncation": True, "padding": "max_length", "max_length": 512 },
    "hucopa": { "truncation": True, "padding": "max_length", "max_length": 256 },
    "hucola": { "truncation": True, "padding": "max_length", "add_special_tokens": True },
    "husst":  { "truncation": True, "padding": "max_length", "add_special_tokens": True },
}

CB = "hucommitmentbank"
SST = "husst"
COPA = "hucopa"