# -*- coding: utf-8 -*-
"""
Author: Susana Nunes
Description:
    This script calculates the node degree for clustered nodes
    and computes the Information Content (IC) based on the node degree.
    It reads the RDF graph and clustering results, counts the node degrees,
    and saves the IC values in JSON files.
"""


import math
from rdflib import Graph, URIRef, RDF, RDFS, OWL
import json
from tqdm import tqdm
from pathlib import Path
import pickle

class NodesClusteredDegree:
    def __init__(
        self,
        graph: Graph,
        owl_classes: list,
        clustering: dict,
        nodesDict_path: Path,
    ):
        self.graph = graph
        self.nodes_list = owl_classes
        self.nodesDict_path = nodesDict_path
        self.count_dic: dict = {}

        # Create a reverse lookup dictionary from self.clustering
        self.clustering = {
            entity: cluster
            for cluster, entities in clustering.items()
            for entity in entities
        }

    def write_nodes(self) -> dict:
        # Write the JSON string to a file
        with open(self.nodesDict_path, "w") as f:
            f.write(json.dumps(self.count_dic, indent=4))

        return self.count_dic

    def count_nodes(self):
        for entity in tqdm(self.nodes_list):
            # check if the entity is a class,
            # we don't count instances node degrees
            # but with metamodelling there are individuals that are classes
            if (URIRef(entity), RDF.type, OWL.Class) in self.graph:
                # get all out triples with the entity target as subject
                # (OUT NODES)
                out_triples = [
                    o
                    for s, p, o in self.graph.triples(
                        (URIRef(entity), None, None)
                    )
                    if not (
                        o == OWL.Class
                        or o == OWL.NamedIndividual
                        or o
                        in self.graph.subjects(RDF.type, OWL.NamedIndividual)
                        or p == RDFS.label
                        or p == OWL.annotatedTarget
                        or p == OWL.annotatedSource
                        or p
                        == URIRef(
                            "http://purl.obolibrary.org/obo/hasAnnotation"
                        )
                        or p
                        in self.graph.subjects(
                            RDF.type, OWL.AnnotationProperty
                        )
                        or p
                        in self.graph.subjects(RDF.type, OWL.DatatypeProperty)
                    )
                ]
                # get all out triples with
                # the entity target as object (IN NODES)
                in_triples = [
                    s
                    for s, p, o in self.graph.triples(
                        (None, None, URIRef(entity))
                    )
                    if not (
                        s in self.graph.subjects(RDF.type, OWL.NamedIndividual)
                        or p in {OWL.annotatedTarget, OWL.annotatedSource}
                        or p
                        in self.graph.subjects(
                            RDF.type, OWL.AnnotationProperty
                        )
                        or p
                        in self.graph.subjects(RDF.type, OWL.DatatypeProperty)
                        or p
                        == URIRef(
                            "http://purl.obolibrary.org/obo/hasAnnotation"
                        )
                    )
                ]

                entities = out_triples + in_triples

                # Merge triples into one if they are in the same cluster
                clusters = {}
                for entity in entities:
                    entity = str(entity)
                    cluster = self.clustering.get(entity, None) #COME BACK TO THIS
                    if cluster not in clusters:
                        clusters[cluster] = []
                    clusters[cluster].append(entity)

                counter = len(
                    clusters
                )  # Count the number of clusters, not entities
                self.count_dic[entity] = counter

            # if the entity is not a class
            else:
                pass

        self.write_nodes()

        return self.count_dic


class NodesClusteredDegreeEdgeType:
    def __init__(
        self,
        graph: Graph,
        owl_classes: list,
        clustering: dict,
        nodesDict_path: Path,
    ):
        self.graph = graph
        self.nodes_list = owl_classes
        self.nodesDict_path = nodesDict_path
        self.count_dic: dict = {}
        self.clustering = clustering

    def write_nodes(self) -> dict:
        with open(self.nodesDict_path, "w") as f:
            f.write(json.dumps(self.count_dic, indent=4))

        return self.count_dic

    def count_nodes_by_edge(self):
        for main_entity in tqdm(self.nodes_list):
            # check if the entity is a class,
            # we don't count instances node degrees
            # but with metamodelling there are individuals that are classes
            if (URIRef(main_entity), RDF.type, OWL.Class) in self.graph:
                # get all out triples with the entity target as subject
                # (OUT NODES)
                out_triples = [
                    (p, o)
                    for s, p, o in self.graph.triples(
                        (URIRef(main_entity), None, None)
                    )
                    if not (
                        o == OWL.Class
                        or o == OWL.NamedIndividual
                        or o
                        in self.graph.subjects(RDF.type, OWL.NamedIndividual)
                        or p == RDFS.label
                        or p == OWL.annotatedTarget
                        or p == OWL.annotatedSource
                        or p
                        == URIRef(
                            "http://purl.obolibrary.org/obo/hasAnnotation"
                        )
                        or p
                        in self.graph.subjects(
                            RDF.type, OWL.AnnotationProperty
                        )
                        or p
                        in self.graph.subjects(RDF.type, OWL.DatatypeProperty)
                    )
                ]
                # get all out triples with
                # the entity target as object (IN NODES)
                in_triples = [
                    (p, s)
                    for s, p, o in self.graph.triples(
                        (None, None, URIRef(main_entity))
                    )
                    if not (
                        s in self.graph.subjects(RDF.type, OWL.NamedIndividual)
                        or p in {OWL.annotatedTarget, OWL.annotatedSource}
                        or p
                        in self.graph.subjects(
                            RDF.type, OWL.AnnotationProperty
                        )
                        or p
                        in self.graph.subjects(RDF.type, OWL.DatatypeProperty)
                        or p
                        == URIRef(
                            "http://purl.obolibrary.org/obo/hasAnnotation"
                        )
                    )
                ]

                # list of tuples (edge, entity)
                edge_neighbors = out_triples + in_triples

                # turn to dictionary with edge as key and
                # list of entities as value
                edges_neighbors_dict = {}
                for edge, neighbor in edge_neighbors:
                    if edge not in edges_neighbors_dict:
                        edges_neighbors_dict[edge] = []
                    edges_neighbors_dict[edge].append(neighbor)

                # Merge triples into one if they are in the same cluster
                for edge, neighbors in edges_neighbors_dict.items():
                    # reverse lookup dictionary to get
                    # the cluster of the entity
                    cluster = self.clustering[str(edge)]
                    edge_clustering = {
                        entityy: cluster
                        for cluster, entitiess in cluster.items()
                        for entityy in entitiess
                    }

                    # Merge triples into one if they are in the same cluster
                    clusters = {}
                    for neighb in neighbors:
                        neighb = str(neighb)
                        cluster = edge_clustering.get(neighb, None) #COME BACK TO THIS 
                        if cluster not in clusters:
                            clusters[cluster] = []
                        clusters[cluster].append(neighb)

                    counter = len(
                        clusters
                    )  # Count the number of clusters, not entities
                    if edge not in self.count_dic:
                        self.count_dic[edge] = {}
                    self.count_dic[edge][main_entity] = counter

            # if the entity is not a class
            else:
                pass

        self.write_nodes()

        return self.count_dic

class NodeDegree_IC:
    """
    Given a node degree computes the normalized
    IC for each class in the ontology.
    Args:
        nodeDegreeDic_path (str): Path to the file
        with the node degree of each class.
        icNodesmaps_path (str): Path to save the IC map.

    Returns:
        icMap (dict): A dictionary containing
        the computed IC values for classes.
    """

    def __init__(self, nodeDegreeDic: dict, icNodesmaps_path: Path):
        self.IC_nodes: dict = {}
        self.nodesDegree = nodeDegreeDic
        self.ICmap_path = icNodesmaps_path

    def highestND(self) -> int:
        """
        Get the highest node degree of a node (owl class) in the whole graph.
        Returns:
            int: highest node degree.
        """

        Hkey = max(self.nodesDegree, key=self.nodesDegree.get)  # type: ignore

        Hvalue = self.nodesDegree[Hkey]
        # print(Hkey, self.nodesDegree[Hkey])

        return Hvalue

    def calculate_normalized_NodeIC(self) -> dict:
        """
        Calculate the normalized IC of each node in the graph.
        Returns:
            dict: dictionary with the normalized IC of each node.
        """
        Hvalue = self.highestND()
        # print('highest ND:', Hvalue)
        max_IC = -math.log2(
            1 / Hvalue
        )  # Calculate the maximum possible IC value
        # (when node degree is 1)
        for node, node_degree in self.nodesDegree.items():
            if node_degree != 0:
                IC = -math.log2(node_degree / Hvalue)
                normalized_IC = (
                    IC / max_IC
                )  # Normalize IC value, a min value is zero
                # so I didnt put it
                self.IC_nodes[node] = normalized_IC
            else:
                self.IC_nodes[node] = 0

        return self.IC_nodes

    def get_icNodes(self) -> str:
        """
        Writes the IC given a node degree of each class in a JSON file.
        Returns:
            icMap (dict): A dictionary containing
              the computed IC values for classes.
        """
        icMap = self.calculate_normalized_NodeIC()

        ic = json.dumps(icMap, indent=4)  # convert to JSON format

        with open(self.ICmap_path, "w") as f:
            f.write(ic)  # save the IC map into a json file

        return ic


class NodeDegree_IC_ByEdgeType:
    """
    Given a node degree computes the normalized
    IC for each class in the ontology bye edge type.
    Args:
        nodeDegreeDic_path (str): Path to the file
        with the node degree of each class by edge type.
        icNodesmaps_path (str): Path to save the IC map
        by edge type.

    Returns:
        icMap (dict): A dictionary with dictionaries containing
        the edge type and computed IC values for classes .
    """

    def __init__(self, nodeDegreeDic: dict, icNodesmaps_path: Path):
        self.IC_nodes: dict = {}
        self.nodesDegree = (
            nodeDegreeDic  # dict with all the node degrees by edge type
        )
        self.ICmap_path = icNodesmaps_path

    def highest_node_degree_edge(self, nodes_degree_edge) -> int:
        """
        Get the highest node degree of a node (owl class) in
        the edge type subgraph.
        Returns:
            int: highest node degree.
        """

        Hkey = max(
            nodes_degree_edge, key=nodes_degree_edge.get
        )  # type: ignore

        Hvalue = nodes_degree_edge[Hkey]

        return Hvalue

    def calculate_normalized_NodeIC_edge(self, nodes_degree_edge) -> dict:
        """
        Calculate the normalized IC of each node in the subgraph of
        an edge type.
        Returns:
            dict: dictionary with the normalized IC of each node.
        """

        IC_nodes_edge_dict: dict = {}

        Hvalue = self.highest_node_degree_edge(nodes_degree_edge)
        # print("highest ND:", Hvalue)

        if Hvalue == 1:
            # If Hvalue is 1, all nodes have the same degree,
            # and their normalized IC should be 0.
            for node in nodes_degree_edge.keys():
                IC_nodes_edge_dict[node] = 0
        else:
            max_IC = -math.log2(
                1 / Hvalue
            )  # Calculate the maximum possible IC value
            # (when node degree is 1)
            for node, node_degree in nodes_degree_edge.items():
                if node_degree != 0:
                    IC = -math.log2(node_degree / Hvalue)
                    normalized_IC = (
                        IC / max_IC
                    )  # Normalize IC value, a min value is zero
                    # so I didnt put it
                    IC_nodes_edge_dict[node] = normalized_IC
                else:
                    IC_nodes_edge_dict[node] = 0

        return IC_nodes_edge_dict

    def get_icNodes_by_edge(self) -> None:
        """
        Writes the IC given a node degree of each class in a JSON file.
        Returns:
            icMap (dict): A dictionary containing
              the computed IC values for classes.
        """

        for i in self.nodesDegree:
            nodes_degree_edge = self.nodesDegree[i]

            IC_nodes_edge = self.calculate_normalized_NodeIC_edge(
                nodes_degree_edge
            )

            self.IC_nodes[i] = IC_nodes_edge

        with open(self.ICmap_path, "w") as f:
            json.dump(self.IC_nodes, f, indent=4)



if __name__ == "__main__":
    print("Starting the script...")

    # Input paths
    #file with nodes to calculate IC (one per line) (e.g. http://onto/Gene::10318 - your graph ents need to always have the url)
    owl_classes_path = "owlClasses.txt" #nodes from the dataset to calculate IC for
    #file with dataset (in OWL format)
    rdf_path =  "graph.owl"

    #file with edge-type clustering
    clustering_edgeType_path = "clustering_default_10percent_edgeType.json"

    # Output folder
    main_output = Path("storage/IC_results")
    main_output.mkdir(exist_ok=True)

    # Load OWL classes
    print(f"Loading OWL classes from: {owl_classes_path}")
    with open(owl_classes_path, "r") as file:
        owl_classes = [line.strip() for line in file.readlines()]
    print(f"Loaded {len(owl_classes)} OWL classes.")

    # Load RDF graph with ontologies -> if you're using the pkl file from the script that creates the graph, change this to load the pkl file instead
    print(f"Loading RDF graph from: {rdf_path}")
    graph = Graph()
    graph.parse(rdf_path) 
    print("RDF graph loaded.")

    # Load edge-type clustering
    print(f"Loading edge-type clustering from: {clustering_edgeType_path}")
    with open(clustering_edgeType_path, 'r') as f:
        clustering_edge = json.load(f)
    print("Edge-type clustering loaded.")

    # Clustered node degree by edge type
    node_clustered_edgeType_degree_path = main_output / "clustered_nodeDegree_edgeType.json"
    print("Calculating node degree by edge type...")
    node_degree_edge = NodesClusteredDegreeEdgeType(
        graph,
        owl_classes,
        clustering_edge,
        str(node_clustered_edgeType_degree_path),
    )
    clustered_nodeDegree_edgeType = node_degree_edge.count_nodes_by_edge()
    print("Node degree by edge type calculated.")

    # IC nodes by edge type
    clustered_ic_edgeType_nodes_path = main_output / "clustered_IC_classes_edgeType.json"
    print("Calculating IC nodes by edge type...")
    ic_node_degree_edge = NodeDegree_IC_ByEdgeType(
        clustered_nodeDegree_edgeType,
        str(clustered_ic_edgeType_nodes_path),
    )
    ic_node_degree_edge.get_icNodes_by_edge()
    print("IC nodes by edge type calculated.")


    print("Script finished successfully.")
