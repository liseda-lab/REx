# REx
Rewarding Explainability in Drug Repurposing with Knowledge Graphs

- This repo provides the implementation described in [paper](https://liseda-lab.github.io/assets/pdf/2025IJCAI_RewardingExplainability.pdf) as well as the Supplementary Material.

## Overview
REx is an approach designed to validate scientific hypothesis.  This README will guide you through building a Docker image, configuring the project, and running the necessary commands.

## Prerequisites
- Docker installed on your machine

## Building the Docker Image
To build the Docker image, use the provided `Dockerfile`. Run the following command in the root directory of the project:

```sh
docker build -t rex-image .
```
Start a container from the built image:

```sh
docker run --gpus all -d --name rl_workspace -v $(pwd):/REx rex-image tail -f /dev/null

```

Create an interactive shell in the container to run commands:

```sh
docker exec -it rl_workspace bash
```

## Running REx
Once inside the container, to run REx, you will need to run the following command:

```sh
uv run bash run.sh configs/{dataset}
```

Where `{dataset}` is the name of the dataset you would like to run the approach on. 

## Datasets 
Datasets should have the following files:
```
dataset
    ├── graph.txt
    ├── dev.txt
    ├── test.txt
    ├── train.txt
    └── clustered_IC_classes_edgeType.json
    └── vocab
        └── entity_vocab.json
        └── relation_vocab.json
```

Where:
- `graph.txt` contains all triples of the KG except for `dev.txt`, `test.txt`.
- `dev.txt` contains all validation triples.
- `test.txt` contains all test triples.
- `train.txt` contains all train triples.
- `clustered_IC_classes_edgeType.json` contains the IC scores for each edge types of the graph. It is a dictionary where the keys are the edge types and the values are dictionaries with the IC scores for each class.
- `vocab/entity_vocab.json` contains the vocabulary for the entities.
- `vocab/relation_vocab.json` contains the vocabulary for the relations.
- The vocab files are created by using the `create_vocab.py` file.


**Note**: The existing datasets have the graph.txt file divided into one or more files. To use the existing datasets, you need to merge the files into one file. Just run the following command in the dataset directory:

```sh
cat graph_part*.txt > graph.txt

```



