# tmynNLP
A unified framework for NLP tasks, using Machine Learning models, inspired by [AllenNLP](https://github.com/allenai/allennlp)

# Setup

It is recommended to use a conda environment for this project.

Follow the steps below to get started.

**Create a conda environment**

```shell
conda create -n tmynnlp_env python=3.8
```

**Install the package with the requirements**

```shell
pip install -e .
```

**To make the CLI executable from any location: set this alias in your .zshrc or .bash_profile**

```shell
alias tmynnlp='[ABSOLUTE_PATH_TO_THE_DIR]/tmynnlp/__main__.py'
```

# CLI usage

## Example: Running the Experiments with specific configurations:

**The trainner**
```shell
tmynnlp train runs/runs.json --include_package document_classification
```

**Sample runs.json file**
```json
[
    {
        "type": "experiment1",
        "num_epochs": 10,
        "batch_size": 32,
        "dataset_reader": {
            "type": "ted_multi",
            "train_data_path": "train",
            "valid_data_path": "validation",
            "mock_samples_num": 500,
            "preprocessor": {
                "type": "drop_nan"
            }
        },
        "tokenizer": {
            "type": "huggingface_tokenizer",
            "pretrained_model": "bert-base-cased"
        },
        "model": {
            "type": "huggingface_sequence_classifier",
            "arch": "xlm-roberta-base",
            "num_labels": 60
        },
        "tracker": {
            "type": "aim",
            "repo_path": ".tmp_aim"
        },
        "metrics": [
            {
                "type": "accuracy"
            },
            {
                "type": "f1",
                "average": "weighted"
            }
        ],
        "criterion": {
            "type": "CrossEntropyLoss"
        },
        "optimizer": {
            "type": "SGD",
            "lr": 0.0005
        },
        "lr_scheduler": {
            "type": "StepLR",
            "step_size": 1.0,
            "gamma": 0.1
        }
    }
]
```
