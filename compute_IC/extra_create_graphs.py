from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL, Namespace
import pickle

#THIS FUNCTION IS NEEDED IF YOU WANT TO CREATE THE GRAPH FROM RAW FILES
# ITS IS USED TO RUN THE SCRIPTS 1 AND 3 IN THE compute_IC FOLDER
def create_graph_in_OWL(graph_path, graph_labels_path, edges_labels_path, output_base='graph.owl'):
        """Creates the graph from given input files.
        This graph can be used to then compute mappings between our dataset and an ontology.
        It is also used for calculating the IC of the clusters.

        Note: For a better performance of ontology matching algorithms, open the serialized file through Protégé and save it.
        
        Returns:
        RDFlibGraph: The graph created from the input files.
        
        """
        namespace = Namespace('http://onto/')
        graph = Graph()

        with open(graph_path, "r") as f:
            for line in f:
                ent1, rel, ent2 = line.strip("\n").split("\t")

                rel = rel.replace(">", "_")
                rel = rel.replace(" ", "_")
                rel_ = namespace[rel]

                ent1_ = namespace[ent1.replace(" ", "_")]
                ent2_ = namespace[ent2.replace(" ", "_")]

                graph.add((URIRef(ent1_), URIRef(rel_), URIRef(ent2_)))
                graph.add((URIRef(ent1_), RDF.type, OWL.Class))
                graph.add((URIRef(ent2_), RDF.type, OWL.Class))

        with open(graph_labels_path, "r") as f:
            for line in f:
                ent, name, type = line.strip("\n").split("\t")
                ent_ = namespace[ent.replace(" ", "_")]

                graph.add((URIRef(ent_), RDFS.label, Literal(name)))

        with open(edges_labels_path, "r") as f:
            for line in f:
                rel, label = line.strip("\n").split("\t")
                rel_ = namespace[rel.replace(">", "_")]
                graph.add((URIRef(rel_), RDFS.label, Literal(label)))
                graph.add((URIRef(rel_), RDF.type, OWL.ObjectProperty))


        graph.serialize(destination=f"{output_base}.owl", format="xml")


#USAGE EXAMPLE
#graph_path = "hetionet.txt" #FILE WITH THREE COLUMNS: ENTITY1    RELATION    ENTITY2
#graph_labels_path = "graph_labels.tsv" #TSV FILE WITH THREE COLUMNS: ENTITY    NAME    TYPE
#edges_labels_path = "edges_labels.tsv" #TSV FILE WITH TWO COLUMNS: RELATION    LABEL
#output= 'hetionet_graph'
#create_graph_in_OWL(graph_path, graph_labels_path, edges_labels_path, output)

##### ---- ###### 

#THIS FUNCTION IS TO CREATE A EXPANDED GRAPH WITH ONTLOGY ALIGNMENTS AND OUR GRAPH 
##THIS GRAPH IS USED TO GENERATE RICHER EMBEDDINGS FROM OWL2VEC* FOR INSTANCE
def build_expanded_graph(
    chebi_mappings_path: str,
    ncit_mappings_path: str,
    base_graph_path: str,
    chebi_owl_path: str,
    ncit_owl_path: str, 
    output_base: str
) -> Graph:
    """
    Build an expanded RDF graph by aligning Hetionet with ontology mappings (ChEBI, NCIT).

    Inputs:
      - chebi_mappings_path: TSV file with columns "ONTOLOGY_ID<TAB>HETIONET_ID"
      - ncit_mappings_path:  TSV file with columns "ONTOLOGY_ID<TAB>HETIONET_ID"
      - base_graph_path:     Path to the base Hetionet graph (OWL/RDF)
      - chebi_owl_path:      Path to ChEBI ontology file (OWL/RDF)
      - ncit_owl_path:       Path to NCIT ontology file (OWL/RDF)
      - output_base:         Base path (without extension) for outputs

    Outputs:
      - Writes {output_base}.owl (RDF/XML) and {output_base}.pkl (pickle of rdflib.Graph)
      - Returns the in-memory rdflib.Graph
    """
    # ---- read mappings (deduplicated) ----
    chebi_mappings = set()
    with open(chebi_mappings_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                onto, target = line.split("\t", 1)
            except ValueError:
                continue  # skip malformed lines
            chebi_mappings.add((target, onto))  # (HETIONET_ID, ONTOLOGY_ID)

    ncit_mappings = set()
    with open(ncit_mappings_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                onto, target = line.split("\t", 1)
            except ValueError:
                continue
            ncit_mappings.add((target, onto))   # (HETIONET_ID, ONTOLOGY_ID)

    # ---- parse graphs (base + ontologies) ----
    g = Graph()
    g.parse(base_graph_path)
    g.parse(chebi_owl_path)
    g.parse(ncit_owl_path)

    # ---- add owl:sameAs links from mappings ----
    for target, onto in chebi_mappings:
        g.add((URIRef(target), OWL.sameAs, URIRef(onto)))

    for target, onto in ncit_mappings:
        g.add((URIRef(target), OWL.sameAs, URIRef(onto)))

    # ---- serialize outputs ----
    g.serialize(destination=f"{output_base}.owl", format="xml")

    # ---- save in pkl format ----
    with open(f"{output_base}.pkl", "wb") as fh:
        pickle.dump(g, fh)

    return g


# Example:
# build_expanded_graph(
#     chebi_mappings_path="chebi_hetionet_mappings.tsv",
#     ncit_mappings_path="ncit_hetionet_mappings.tsv",
#     base_graph_path="hetionet_graph.owl",
#     chebi_owl_path="chebi.owl",
#     ncit_owl_path="NCIT.owl",
#     output_base="hetionet_chebi_ncit"
# )
