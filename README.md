# HuLU
Hungarian Language Understanding Benchmark Kit


This repository contains the databases included in HuLU, the Hungarian Language Understanding Benchmark Kit developed, maintained and updated in the Language Technology Research Group of the Hungarian Research Centre for Linguistics.

Currently (3/11/2022) four corpora are available to download and to test the models on.

- **HuCOLA** (Hungarian Corpus of Linguistic Acceptability) contains 9 076 Hungarian sentences labelled for their acceptability/grammaticality (0/1). The sentences were collected by two human annotators from three linguistic books. Each sentence was annotated by four human annotators. The final label of the sentence is the one assigned by the majority of the annotators. The proportion of train, validation and test sets is 80% (7 276 sentences), 10% (900 sentences) and 10% (900 sentences), respectively.  
- **HuCoPa** (Hungarian Choice of Plausible Alternatives Corpus) contains 1 000 instances. Each instance is composed of a premise and two alternatives. The task is to select the alternative that describes a situation standing in causal relation to the situation described by the premise. The corpus was created by translating and re-annotating the original English CoPA corpus. The train, validation and test sets contain 400, 100 and 500 instances, respectively.  
- **HuRC** (Hungarian Reading Comprehension Dataset) contains 80 621 instances. Each instance is composed of a passage and a cloze-style query with a masked entity. The task is to select the named entity that is being masked in the query. The data was collected from the online news of Népszabadság online (nol.hu).  
- **HuSST** (Hungarian version of the Stanford Sentiment Treebank) contains 11 683 sentences. Each sentence is annotated for its sentiment on a three-point scale. The corpus was created by translating and re-annotating the full sentences of the SST. The train, validation and test sets contain 9 347, 1 168 and 1 168 sentences, respectively.  
- **HuWNLI** is a Hungarian dataset of anaphora resolution, designed as a sentence pair classification task of natural language inference. Its base, the HuWS corpus was created by translating and manually curating the original English Winograd schemata. The NLI format created by replacing the ambiguous pronoun with each possible referent in the schemata. We extended the set of sentence pairs derived from the schemata by the translation of the sentence pairs that build up the WNLI dataset of GLUE. The data is distributed in three splits: training set (562), development set (59) and test set (134).  

## Citation

If you use these resources or any part of its documentation, please refer to:

Ligeti-Nagy, N., Ferenczi, G., Héja, E., Jelencsik-Mátyus, K., Laki, L. J., Vadász, N., Yang, Z. Gy. and Váradi, T. (2022) HuLU: magyar nyelvű benchmark adatbázis kiépítése a neurális nyelvmodellek kiértékelése céljából [HuLU: Hungarian benchmark dataset to evaluate neural language models]. In: Berend, G., Gosztolya, G. and Vincze, V. (eds), XVIII. Magyar Számítógépes Nyelvészeti Konferencia. Szeged, Szegedi Tudományegyetem, Informatikai Intézet. 431–446.

```

@inproceedings{ligetinagy2022hulu,
  title={HuLU: magyar nyelvű benchmark adatbázis kiépítése a neurális nyelvmodellek kiértékelése céljából},
  author={Ligeti-Nagy, N. and Ferenczi, G. and Héja, E. and Jelencsik-Mátyus, K. and Laki, L. J. and Vadász, N. and Yang, Z. Gy. and Váradi, T.},
  booktitle={XVIII. Magyar Számítógépes Nyelvészeti Konferencia},
  year={2022},
  pages = {431--446},
  editors = {Berend, G. and Gosztolya, G. and Vincze, V.},
  address = {Szeged},
  publisher = {Szegedi Tudományegyetem, Informatikai Intézet}
}
```

and to any other the references listed in the readme files of the individual corpora. 
