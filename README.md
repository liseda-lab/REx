# REx
Rewarding Explainability in Drug Repurposing with Knowledge Graphs

- This repo provides the implementation and Supplementary Material described in this [paper](https://liseda-lab.github.io/assets/pdf/2025IJCAI_RewardingExplainability.pdf).


REx is a method designed to validate scientific hypotheses via **explanatory paths in biomedical knowledge graphs**. It uses a **reinforcement learning** framework guided by a **multi-objective reward function** that prioritizes:

- **Fidelity** – ensures the path connects the subject and object of the hypothesis.
- **Relevance** – prioritizes explanations that are specific and informative.
- **Simplicity** – promotes concise and interpretable paths.

These paths are enriched with ontology-based classes to ensure **completeness** and **coherence** with biomedical knowledge.

<p align="center">
  <img src="Overview.png" alt="REx Overview" width="800"/>
</p>



---

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

## Parse results
To parse the results of the REx approach, you can use the `path_parser.py` script. This script will read the testbeam files generated by REx and create a json file with a dictionary with the results. The script will also create a file with the results in the same directory.
- required files:
  - `testbeam.txt`: the file generated by REx with the results of the test set.
  - `vocab/entity_vocab.json`: the vocabulary for the entities.
  - `vocab/relation_vocab.json`: the vocabulary for the relations.

## Ontology Enrichment
- `lca_finder.py` contains the lowest common ancestor (LCA) implementation for the entities in a given dataset. It outputs the LCA for each pair provided in the input list.
- required files:
  - `dataset_ontology_DAG.pkl`: a pickle file with a DAG with the subclasses of the ontology, the dataset and the mapping of the entities to the ontology classes.
  - `onto_labels.json`: a json with a dictionary mapping the entity IDs from the ontologies to their labels.
  - `graph_labels.tsv`: a tsv file with the labels of the entities in the graph. The first column is the entity ID and the second column is the label.

## Authors
- __Susana Nunes__
- __Samy Badreddine__
- __Catia Pesquita__

For any comments or help needed with this implementation, please send an email to: scnunes@ciencias.ulisboa.pt

## Acknowledgments
This work was supported by FCT through the fellowship 2023.00653.BD, and the LASIGE Research Unit, ref. UID/00408/2025. It was also partially supported by the KATY project (European Union Horizon 2020 grant No. 101017453), and project 41, HfPT: Health from Portugal, funded by the Portuguese Plano de Recuperação e Resiliência. We thank Sony AI, where the first author conducted part of this work during an internship. We also thank Pedro Cotovio for his input on reinforcement learning fundamentals. 

## Citation
If you use this code in your research, please cite the following paper:

```
If you use this code, please cite our paper
@inproceedings{Nunes2025REx,
  title={Rewarding Explainability in Drug Repurposing with Knowledge Graphs},
  author={Nunes, Susana and Badreddine, Samy and Pesquita, Catia},
  booktitle={IJCAI},
  year={2025}
}
```
