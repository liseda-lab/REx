#!/usr/bin/env python3
"""
rex_for_visualization.py
Parser for simpler beam output with IC-based filtering and metapaths. 
This is to save a json indicated for visualization tools .
It includes LCA support.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from itertools import combinations
import networkx as nx
from rdflib import URIRef


class SimpleBeamParser:
    """Parser for simple beam output with IC-based metapath filtering."""
    
    def __init__(self, 
                 nodes_tsv: Path,
                 edges_tsv: Path,
                 clustered_ic_json: Path,
                 enable_lca: bool = False,
                 ncit_pickle: Optional[Path] = None,
                 chebi_pickle: Optional[Path] = None,
                 ontology_labels: Optional[Path] = None,
                 debug: bool = False):
        """Initialize parser with label mappings and IC data."""
        self.debug = debug
        self.node_labels, self.node_types = self._load_tsv_with_types(nodes_tsv)
        self.edge_labels = self._load_tsv_labels(edges_tsv)
        self.clustered_ic = self._load_clustered_ic(clustered_ic_json)
        
        self.enable_lca = enable_lca
        if enable_lca:
            if not all([ncit_pickle, chebi_pickle, ontology_labels]):
                raise ValueError("LCA requires ncit_pickle, chebi_pickle, and ontology_labels paths")
            self.lca_finder = LCAFinder(ncit_pickle, chebi_pickle, nodes_tsv, ontology_labels)
        else:
            self.lca_finder = None
    
    def _load_tsv_labels(self, tsv_path: Path) -> Dict[str, str]:
        """Load TSV of <code>\t<label> into a dict."""
        mapping = {}
        with tsv_path.open(encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
        return mapping
    
    def _load_tsv_with_types(self, tsv_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Load TSV of <id>\t<label>\t<type> into two dicts."""
        labels = {}
        types = {}
        with tsv_path.open(encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    node_id, label, node_type = parts[0], parts[1], parts[2]
                    labels[node_id] = label
                    types[node_id] = node_type
                elif len(parts) >= 2:
                    # Fallback for files without type column
                    labels[parts[0]] = parts[1]
                    # Infer type from prefix as fallback
                    if '::' in parts[0]:
                        types[parts[0]] = parts[0].split('::')[0]
                    else:
                        types[parts[0]] = "Unknown"
        return labels, types
    
    def _load_clustered_ic(self, json_path: Path) -> Dict[str, Dict[str, float]]:
        """Load clustered IC by edge type from JSON."""
        with json_path.open('r') as f:
            return json.load(f)
    
    def _calculate_path_ic(self, nodes: List[str], edges: List[str]) -> float:
        """Calculate average IC for a path using clustered IC by edge type."""
        total_ic = 0.0
        edge_count = 0
        
        for i, edge_type in enumerate(edges):
            if edge_type and edge_type != 'NO_OP':
                if i < len(nodes) - 1:
                    node1 = nodes[i]
                    node2 = nodes[i + 1]
                    
                    # Try both with and without prefix for edge type
                    edge_key = edge_type
                    if edge_key not in self.clustered_ic:
                        edge_key = f"http://onto/{edge_type}"
                        if edge_key not in self.clustered_ic:
                            if self.debug:
                                print(f"    Warning: Edge type '{edge_type}' not found in clustered IC")
                            continue
                    
                    # Try both formats for nodes
                    node1_key = node1
                    node2_key = node2
                    
                    # Check if we need to add prefix to nodes
                    edge_dict = self.clustered_ic.get(edge_key, {})
                    if node1 not in edge_dict and f"http://onto/{node1}" in edge_dict:
                        node1_key = f"http://onto/{node1}"
                    if node2 not in edge_dict and f"http://onto/{node2}" in edge_dict:
                        node2_key = f"http://onto/{node2}"
                    
                    # Get IC values
                    ic1 = edge_dict.get(node1_key, 0.0)
                    ic2 = edge_dict.get(node2_key, 0.0)
                    
                    if self.debug and (ic1 == 0.0 or ic2 == 0.0):
                        print(f"    Warning: Zero IC for {node1}={ic1:.3f}, {node2}={ic2:.3f} on edge {edge_type}")
                    
                    edge_ic = (ic1 + ic2) / 2.0
                    total_ic += edge_ic
                    edge_count += 1
        
        return total_ic / edge_count if edge_count > 0 else 0.0
    def _get_metapath(self, nodes: List[str], edges: List[str]) -> str:
        """Generate metapath signature from nodes and edges."""
        # Create pattern from node types and edge types
        node_types = [self._infer_node_type(node) for node in nodes]
        metapath_parts = []
        
        for i in range(len(nodes)):
            metapath_parts.append(node_types[i])
            if i < len(edges) and edges[i] and edges[i] != 'NO_OP':
                metapath_parts.append(edges[i])
        
        return '->'.join(metapath_parts)
    
    def _infer_node_type(self, node_code: str) -> str:
        """Get node type from loaded types dictionary."""
        return self.node_types.get(node_code, "Unknown")
    
    def _parse_block(self, block: str) -> Optional[Tuple[str, Dict]]:
        """
        Parse a single block from the simple beam output.
        Returns: (pair_id, path_obj) or None
        """
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        
        if len(lines) < 6:
            if self.debug:
                print(f"Skipping block with {len(lines)} lines (need 6)")
            return None
        
        # Parse the simple format
        pair_id = lines[0]  # Compound::XX	Disease::YY
        reward_header = lines[1]  # Reward:1 (pair-level, not used for filtering)
        route_line = lines[2]  # Node path
        edges_line = lines[3]  # Edge types
        label_line = lines[4]  # Reward value (1 or -1) - this is what we filter on
        negative_score = lines[5]  # Negative score (ignored)
        
        # Parse label (reward) - this is what determines if we process the path
        try:
            label = int(label_line)
        except ValueError:
            if self.debug:
                print(f"Could not parse reward value: {label_line}")
            return None
        
        # Only process paths with positive rewards (not the header, the actual path reward)
        if label != 1:
            if self.debug:
                print(f"Skipping path with reward {label}")
            return None
        
        # Parse nodes and edges
        raw_nodes = route_line.split('\t')
        raw_edges = edges_line.split('\t')
        
        # Calculate IC for this path
        path_ic = self._calculate_path_ic(raw_nodes, raw_edges)
        
        # Generate metapath
        metapath = self._get_metapath(raw_nodes, raw_edges)
        
        if self.debug:
            print(f"Found path with IC={path_ic:.4f}, metapath={metapath}")
        
        # Create nodes and edges for JSON output
        nodes = []
        for node_code in raw_nodes:
            nodes.append({
                "id": self.node_labels.get(node_code, node_code),
                "type": self._infer_node_type(node_code)
            })
        
        edges = []
        for i in range(len(raw_edges)):
            if i < len(raw_nodes) - 1 and raw_edges[i] and raw_edges[i] != 'NO_OP':
                edges.append({
                    "source": self.node_labels.get(raw_nodes[i], raw_nodes[i]),
                    "target": self.node_labels.get(raw_nodes[i+1], raw_nodes[i+1]),
                    "label": self.edge_labels.get(raw_edges[i], raw_edges[i])
                })
        
        path_obj = {
            "score": {
                "scientific_validity": None,
                "completeness": None,
                "relevance": None,
                "final_score": path_ic
            },
            "reasoning": "Path extracted with average IC",
            "nodes": nodes,
            "edges": edges,
            "metapath": metapath,
            "raw_nodes": raw_nodes  # Keep for LCA computation
        }
        
        return pair_id, path_obj
    
    def parse_file(self, 
                input_path: Path,
                chunk_separator: str = "___") -> Dict[str, Any]:
        """Parse the entire file with pairs and paths."""
        raw_content = input_path.read_text(encoding='utf-8').strip()
        
        # First split by double newlines to find pairs
        # Each pair starts with "Compound::XXX\tDisease::YYY\nReward:1"
        import re
        
        # Pattern to find ANY pair headers (regardless of Reward value)
        pair_pattern = r'(\S+::\S+\t+\S+::\S+)\nReward:(-?\d+)\n'
        
        # Find all pairs with their content
        pairs_data = []
        matches = list(re.finditer(pair_pattern, raw_content))
        
        for i, match in enumerate(matches):
            pair_id = match.group(1)
            start = match.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(raw_content)
            pair_content = raw_content[start:end].strip()
            
            if self.debug:
                print(f"Found pair: {pair_id}")
            
            # Split paths within this pair
            path_chunks = pair_content.split(chunk_separator)
            
            paths = []
            for chunk in path_chunks:
                if not chunk.strip():
                    continue
                
                lines = [ln.strip() for ln in chunk.strip().splitlines() if ln.strip()]
                
                if len(lines) != 4:
                    if self.debug:
                        print(f"  Skipping path with {len(lines)} lines (need 4)")
                    continue
                
                # Parse the 4-line path format
                route_line = lines[0]  # Node path
                edges_line = lines[1]  # Edge types
                reward_line = lines[2]  # Reward value (1 or -1)
                score_line = lines[3]   # Negative score
                
                # Parse reward
                try:
                    reward = int(reward_line)
                except ValueError:
                    if self.debug:
                        print(f"  Could not parse reward: {reward_line}")
                    continue
                
                # Only process paths with reward=1
                if reward != 1:
                    if self.debug:
                        print(f"  Skipping path with reward={reward}")
                    continue
                
                # Parse nodes and edges
                raw_nodes = route_line.split('\t')
                raw_edges = edges_line.split('\t')
                
                # Calculate IC
                path_ic = self._calculate_path_ic(raw_nodes, raw_edges)
                
                # Generate metapath
                metapath = self._get_metapath(raw_nodes, raw_edges)
                
                if self.debug:
                    print(f"  Found valid path with IC={path_ic:.4f}, metapath={metapath}")
                
                # Create path object
                nodes = []
                for node_code in raw_nodes:
                    nodes.append({
                        "id": self.node_labels.get(node_code, node_code),
                        "type": self._infer_node_type(node_code)
                    })
                
                edges = []
                for i in range(len(raw_edges)):
                    if i < len(raw_nodes) - 1 and raw_edges[i] and raw_edges[i] != 'NO_OP':
                        edges.append({
                            "source": self.node_labels.get(raw_nodes[i], raw_nodes[i]),
                            "target": self.node_labels.get(raw_nodes[i+1], raw_nodes[i+1]),
                            "label": self.edge_labels.get(raw_edges[i], raw_edges[i])
                        })
                
                path_obj = {
                    "score": {
                        "scientific_validity": None,
                        "completeness": None,
                        "relevance": None,
                        "final_score": path_ic
                    },
                    "reasoning": "Path extracted with average IC",
                    "nodes": nodes,
                    "edges": edges,
                    "metapath": metapath,
                    "raw_nodes": raw_nodes
                }
                
                paths.append((metapath, path_ic, path_obj))
            
            # Keep only highest IC path per metapath
            best_by_metapath = {}
            for metapath, ic, path_obj in paths:
                if metapath not in best_by_metapath or ic > best_by_metapath[metapath][0]:
                    best_by_metapath[metapath] = (ic, path_obj)
            
            if best_by_metapath:
                pairs_data.append((pair_id, best_by_metapath))
        
        # Build final results
        results = {"pairs": []}
        path_id_counter = 1
        
        for pair_id, metapath_dict in pairs_data:
            selected_paths = []
            
            for metapath, (ic, path_obj) in metapath_dict.items():
                # Add LCA if enabled
                if self.enable_lca and self.lca_finder:
                    lcas = self.lca_finder.compute_lcas_for_path(path_obj['raw_nodes'])
                    if lcas:
                        path_obj["lowest_common_ancestors"] = lcas
                
                # Clean up and add ID
                path_obj.pop('raw_nodes', None)
                path_obj.pop('metapath', None)
                path_obj["id"] = f"path_{path_id_counter}"
                path_id_counter += 1
                
                selected_paths.append(path_obj)
            
            if selected_paths:
                results["pairs"].append({
                    "id": pair_id,
                    "paths": selected_paths
                })
        
        return results
class LCAFinder:
    """Compute lowest common ancestors for node pairs in ontology DAGs."""
    
    def __init__(self, ncit_pickle: Path, chebi_pickle: Path, 
                 nodes_tsv: Path, ontology_labels: Path):
        self.dag_ncit = self._load_pickle(ncit_pickle)
        self.dag_chebi = self._load_pickle(chebi_pickle)
        self.node_labels = self._load_tsv_labels(nodes_tsv)
        self.onto_labels = self._load_json(ontology_labels)
        
        self.too_general = {
            'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C7057',
            'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C1909',
            'http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C1908'
        }
    
    def _load_pickle(self, path: Path) -> nx.DiGraph:
        with path.open('rb') as f:
            return pickle.load(f)
    
    def _load_json(self, path: Path) -> Dict[str, str]:
        with path.open('r') as f:
            return json.load(f)
    
    def _load_tsv_labels(self, tsv_path: Path) -> Dict[str, str]:
        labels = {}
        prefix = "http://onto/"
        with tsv_path.open('r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    uri, label = parts[0], parts[1]
                    labels[f"{prefix}{uri}"] = label
        return labels
    
    def compute_lcas_for_path(self, node_codes: List[str]) -> Dict[str, List[str]]:
        """Compute LCAs for all pairs in a path."""
        nodes = [URIRef(f"http://onto/{code}") for code in node_codes]
        nodes_ncit = [n for n in nodes if n in self.dag_ncit]
        nodes_chebi = [n for n in nodes if n in self.dag_chebi]
        
        pairs_ncit = list(combinations(nodes_ncit, 2))
        pairs_chebi = list(combinations(nodes_chebi, 2))
        
        lcas = defaultdict(set)
        
        if pairs_ncit:
            ncit_lcas = dict(nx.all_pairs_lowest_common_ancestor(self.dag_ncit, pairs_ncit))
            for pair, anc in ncit_lcas.items():
                if anc and str(anc) not in self.too_general:
                    lcas[pair].add(anc)
        
        if pairs_chebi:
            chebi_lcas = dict(nx.all_pairs_lowest_common_ancestor(self.dag_chebi, pairs_chebi))
            for pair, anc in chebi_lcas.items():
                if anc and str(anc) not in self.too_general:
                    lcas[pair].add(anc)
        
        result = {}
        for (n1, n2), anc_set in lcas.items():
            n1_label = self.node_labels.get(str(n1), str(n1))
            n2_label = self.node_labels.get(str(n2), str(n2))
            key = f"{n1_label},{n2_label}"
            result[key] = [self.onto_labels.get(str(anc), str(anc)) for anc in anc_set]
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse simple beam output files")
    parser.add_argument("input", type=Path, help="Input beam file")
    parser.add_argument("--nodes-tsv", type=Path, required=True)
    parser.add_argument("--edges-tsv", type=Path, required=True)
    parser.add_argument("--clustered-ic", type=Path, required=True, help="Clustered IC JSON file")
    parser.add_argument("--enable-lca", action="store_true")
    parser.add_argument("--ncit-pickle", type=Path)
    parser.add_argument("--chebi-pickle", type=Path)
    parser.add_argument("--onto-labels", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    beam_parser = SimpleBeamParser(
        nodes_tsv=args.nodes_tsv,
        edges_tsv=args.edges_tsv,
        clustered_ic_json=args.clustered_ic,
        enable_lca=args.enable_lca,
        ncit_pickle=args.ncit_pickle,
        chebi_pickle=args.chebi_pickle,
        ontology_labels=args.onto_labels,
        debug=args.debug
    )
    
    print(f"Parsing {args.input}...")
    results = beam_parser.parse_file(args.input)
    
    # Statistics
    total_paths = sum(len(p['paths']) for p in results['pairs'])
    print(f"\nParsing complete:")
    print(f"  Total pairs: {len(results['pairs'])}")
    print(f"  Total paths (after filtering): {total_paths}")
    
    output_path = args.output or args.input.with_suffix('.json')
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()

    # python rex_for_visualization.py paths_CtD \
    # --nodes-tsv graph_labels.tsv \
    # --edges-tsv edges_labels.tsv \
    # --clustered-ic clustered_IC_classes_edgeType.json \
    # --output output.json \
    # --debug #OPTIONAL 

    #FOR ALSO FINDING LCAs
    # python rex_for_visualization.py hetionet_dt/paths_CbG \
    #     --nodes-tsv graph_labels.tsv \
    #     --edges-tsv edges_labels.tsv \
    #     --clustered-ic hetionet_dt/clustered_IC_classes_edgeType.json \
    #     --enable-lca \
    #     --ncit-pickle NCIT_HETIONET_DAG.pkl \
    #     --chebi-pickle CHEBI_HETIONET_DAG.pkl \
    #     --onto-labels onto_labels.json \
    #     --output output_with_lca.json