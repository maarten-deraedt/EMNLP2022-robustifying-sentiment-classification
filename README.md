# Robustifying Sentiment Classification by Maximally Exploiting Few Counterfactuals

This repository contains the implementation of the EMNLP 2022 paper
["Robustifying Sentiment Classification by Maximally Exploiting Few Counterfactuals"]( https://arxiv.org/abs/2210.11805) by Maarten De Raedt, Fréderic Godin, Chris Develder and Thomas Demeester.

For any questions about the paper or code contact the first author at [maarten.deraedt@ugent.be](mailto:maarten.deraedt@ugent.be).

## Table of Contents
- [Installation](#installation)
- [Experiments](#experiment)


### Installation
Install the requirements and create the directory to which the results will be written to.
```bash
$ pip3 install -r requirements.txt
$ mkdir -p results/metrics
```
Download [encodings.zip](https://drive.google.com/file/d/1ZivfY2OWX3fGJBA7nkDsqIcTk_D_7jQg/view?usp=sharing), unzip it, and place it in under the root of the project.
The directory and file structure should match the structure below.
```
EMNLP2022_robustiyfing_sentiment_classification
└─── datasets/   
│   └──aaai-2021-counterfactuals/
│   └──IMDb/
│   └──OOD/
└───encodings
│   └──all-distilroberta-v1/
│   └──all-mpnet-base-v2/
│   └──all-roberta-large-v1/
│   └──unsup-simcse-bert-base-uncased/
│   └──unsup-simcse-bert-large-uncased/
│   └──unsup-simcse-roberta-large/
└───results
│    └──metrics/
│    │   all-roberta-large-v1.json
│    │   unsup-simcse-roberta-large.json
│    │   ...
│   evaluate.py
│   evaluators.py
│   featurizers.py
│   models.py
│   README.md
│   requirements.txt
```

### Experiments
Run the command below to reproduce the main results for SRoBERTa-large.
```bash
python3 evaluate.py --name "all-roberta-large-v1"
```
And for SimCSE-RoBERTa-large:
```bash
python3 evaluate.py --name "unsup-simcse-roberta-large"
```
The results will be written to `results/metrics/{name}.json`. 
Note that for each value of k (16, 32, 64, 128), 50 different: k/2 negative and
k/2 positive counterfactuals are randomly sampled. As such, the results may
slightly differ from those reported in the paper but the main results and findings will stay consistent.
Depending on the CPU, running the experiments for a single encoder may take between 1 to 2 hours.


