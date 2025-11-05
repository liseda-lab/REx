# -*- coding: utf-8 -*-
"""
Author: Susana Nunes
Description:
    This script processes the embeddings of nodes in an RDF graph
    and divides them by edge type. It reads the embeddings from a JSON file,
    extracts the edges from the graph, and organizes the embeddings
    based on the edge types. The output is a JSON file where each key
    corresponds to an edge type and the value is a dictionary of node
    embeddings associated with that edge type.
"""

import json
from pathlib import Path
import pickle
import numpy as np
from rdflib import OWL, RDF, RDFS, BNode, Graph, Namespace, URIRef
from tqdm import tqdm

class EmbsByEdge:
    """
    Class to process the embeddings of the nodes in the graph and
    divide by edge type.

    Args:
        graph (rdflib.graph.Graph): The graph object.
        embs_path (str): embeddings in JSON file.
        edges_labels (lixst): The list of edges labels in the graph.
        namespace (str): The namespace of the nodes in the graph.

    returns:
        new_embs (dict): A dictionary to store the processed embeddings.
        This dictionary is divided by edge type. The keys are the
        edge labels and the values are dictionaries with the
        embeddings of the nodes connected by the edge.

    """

    def __init__(
        self,
        graph: Graph,
        embs_path: Path,
        edges_labels_path: list,
        new_embs_path: Path,
    ):
        self.graph = graph
        self.embs_path = embs_path
        self.edges_labels_path = edges_labels_path
        self.new_embs_path = new_embs_path
        self.embs: dict = {}
        self.all_edges: set = set()
        self.new_embs: dict = {}

    def load_embeddings(self):
        with open(self.embs_path, "r") as f:
            self.embs = json.load(f)

    def load_edges(self):
        with open(self.edges_labels_path, "r") as f:
            edges = f.readlines()
            for edge in edges:
                rel, _ = edge.strip().split("\t")
                rel_ = URIRef(rel)
                self.all_edges.add(rel_)

    def save_embs(self):
        with open(self.new_embs_path, "w") as f:
            json.dump(self.new_embs, f, indent=4)

    def get_edges_node(self, entity) -> set:
        """
        Get the in and outer edges of an entity.
        """

        edges = set()
        entity_edges = set(
            p for _, p, _ in self.graph.triples((entity, None, None))
        )
        entity_edges.update(
            p for _, p, _ in self.graph.triples((None, None, entity))
        )

        # compare sets edges_labels and entity_edges
        edges = entity_edges.intersection(self.all_edges)

        return edges

    def assign_edges(self) -> None:
        """
        Get nodes list and assign nodes to each edge.
        """
        for entity, emb in self.embs.items():
            edges = self.get_edges_node(URIRef(entity))
            if edges:
                for edge in edges:
                    if edge not in self.new_embs:
                        self.new_embs[edge] = {}
                    self.new_embs[edge][entity] = emb

    def process_embs(self):
        self.load_embeddings()
        self.load_edges()
        self.assign_edges()
        self.save_embs()



if __name__ == "__main__":
    #EMBEDDINGS FILE FROM OWL2VEC* or other format
    #example embedding_format:
    # {
    #     "http://entity1_uri": ["0.16179", "-0.03999", "0.14161", ...],
    #     "http://entity2_uri": ["0.26419", "-0.10048", "0.11824", ...],
    #     "http://entity3_uri": ["0.06358", "-0.25567", "0.31913", ...]
    # }
    # # Each entity URI maps to its embedding vector (as string floats)
    
    embs_path = 'embeddings.json'
    
    #THIS IS IN RDF format (not necessarily owl) as long as RDFLIB can parse
    graph_path = 'rdf_graph.owl'

    graph = Graph()
    graph.parse(graph_path)

    
    #TSV FILE WITH EDGE\TLABEL from your graph  
    edges_labels = 'edge_labels.tsv'

    #OUTPUT
    new_embs_path = 'embeddings_edge_type.json'
    
    process = EmbsByEdge(graph, embs_path, edges_labels, new_embs_path)
    process.process_embs()
