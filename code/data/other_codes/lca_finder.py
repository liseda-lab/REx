#!/usr/bin/env python3
"""
lca_finder.py

Compute and annotate lowest common ancestors (LCAs) for node pairs
in NCIT and CHEBI DAGs, with clean modular structure and type hints.
"""

import pickle
import json
from itertools import combinations
from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict

import networkx as nx
from rdflib import URIRef

Graph = nx.DiGraph
Pair = Tuple[URIRef, URIRef]
Ancestors = Dict[Pair, Set[URIRef]]
Labels = Dict[str, str]


def load_pickle(path: str) -> Any:
    """Load and return a Python object from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_labels(path: str, prefix: str = "http://onto/") -> Labels:
    """
    Load node labels from a TSV file.
    Expects lines: <id>\\t<label>
    Returns mapping from full URI (prefix+id) to label.
    """
    labels: Labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            uri, label = line.strip().split('\t')
            labels[f"{prefix}{uri}"] = label
    return labels


def load_json(path: str) -> Dict[str, Any]:
    """Load and return JSON data from a file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_graphs(
    ncit_path: str,
    chebi_path: str,
    label_path: str,
    ontology_labels_path: str
) -> Tuple[Graph, Graph, Labels, Labels]:
    """
    Load NCIT and CHEBI DAGs and their label mappings.
    Returns: (dag_ncit, dag_chebi, graph_labels, ontology_labels)
    """
    dag_ncit = load_pickle(ncit_path)
    dag_chebi = load_pickle(chebi_path)
    graph_labels = load_labels(label_path)
    ontology_labels = load_json(ontology_labels_path)
    return dag_ncit, dag_chebi, graph_labels, ontology_labels


def compute_pairs(nodes: List[URIRef]) -> List[Tuple[URIRef, URIRef]]:
    """Generate unique unordered node pairs from a list of nodes."""
    return list(combinations(nodes, 2))


def get_common_ancestors(dag: Graph, pairs: List[Pair]) -> Ancestors:
    """
    Compute the lowest common ancestors for each pair in the DAG.
    Returns a mapping from pair -> set([ancestor]).
    """
    raw = nx.all_pairs_lowest_common_ancestor(dag, pairs)
    return {pair: {anc} for pair, anc in raw.items() if anc is not None}


def filter_general(ancestors: Ancestors, skip_uris: Set[str]) -> Ancestors:
    """
    Remove too-general ancestors based on a set of URIs to skip.
    Discards any pair whose ancestor set becomes empty.
    """
    filtered: Ancestors = {}
    for pair, anc_set in ancestors.items():
        valid = {anc for anc in anc_set if str(anc) not in skip_uris}
        if valid:
            filtered[pair] = valid
    return filtered


def annotate_ancestors(
    ancestors: Ancestors,
    graph_labels: Labels,
    ontology_labels: Labels
) -> Dict[Tuple[str, str], List[str]]:
    """
    Convert ancestor URIs and node URIs to human-readable labels.
    Returns mapping from (node_label1, node_label2) -> [ancestor_labels...].
    """
    annotated: Dict[Tuple[str, str], List[str]] = {}
    for (n1, n2), anc_set in ancestors.items():
        key = (
            graph_labels.get(str(n1), str(n1)),
            graph_labels.get(str(n2), str(n2))
        )
        annotated[key] = [
            ontology_labels.get(str(anc), str(anc))
            for anc in anc_set
        ]
    return annotated


def process_paths(
    paths: List[List[URIRef]],
    dag_ncit: Graph,
    dag_chebi: Graph,
    graph_labels: Labels,
    ontology_labels: Labels,
) -> Tuple[Ancestors, Dict[Tuple[str, str], List[str]]]:
    """
    Main pipeline: 
      - Flatten unique nodes
      - Compute NCIT & CHEBI pairs
      - Find LCAs
      - Merge, filter, and annotate
    """
    # Define too-general URIs to skip
    too_general = {
        'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C7057',
        'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C1909',
        'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C1908'
    }

    all_nodes = {node for path in paths for node in path}
    nodes_ncit = [n for n in all_nodes if n in dag_ncit]
    nodes_chebi = [n for n in all_nodes if n in dag_chebi]

    print(f"Nodes in NCIT: {len(nodes_ncit)}, in CHEBI: {len(nodes_chebi)}")

    pairs_ncit = compute_pairs(nodes_ncit)
    pairs_chebi = compute_pairs(nodes_chebi)
    print(f"Pair combos NCIT: {len(pairs_ncit)}, CHEBI: {len(pairs_chebi)}")

    anc_ncit = get_common_ancestors(dag_ncit, pairs_ncit)
    anc_chebi = get_common_ancestors(dag_chebi, pairs_chebi)

    merged = defaultdict(set)
    for anc_map in (anc_ncit, anc_chebi):
        for pair, anc_set in anc_map.items():
            merged[pair].update(anc_set)

    filtered = filter_general(merged, too_general)
    annotated = annotate_ancestors(filtered, graph_labels, ontology_labels)
    return filtered, annotated


def main(label_file, ontology_label_file, ncit_pickle, chebi_pickle):
    """
    Main entry point to load graphs, process paths, and output results.
    """

    # Example input: replace with real URIRef paths
    example_paths: List[List[URIRef]] = [
        # [URIRef("http://onto/C1"), URIRef("http://onto/C2"), ...],
    ]

    # Load resources
    dag_ncit, dag_chebi, graph_labels, ontology_labels = load_graphs(
        ncit_pickle, chebi_pickle, label_file, ontology_label_file
    )


    # Process and output
    raw_ancestors, labeled_ancestors = process_paths(
        example_paths,
        dag_ncit, dag_chebi,
        graph_labels, ontology_labels,
    )

    print("Common Ancestors (raw):")
    for pair, anc in raw_ancestors.items():
        print(f"  {pair}: {anc}")

    print("\nCommon Ancestors (labeled):")
    for pair, labels in labeled_ancestors.items():
        print(f"  {pair}: {labels}")


if __name__ == "__main__":
    main('graph_labels.tsv', 'onto_labels.json', 'NCIT_HETIONET_DAG.pkl', 'CHEBI_HETIONET_DAG.pkl')
    #Note
    # The DAG files are the dataset with mappings for an ontology in DAG form with only subclass relations.
    # graph_labels.tsv contains the labels for the dataset
    # onto_labels.json contains the labels for the ontologies 