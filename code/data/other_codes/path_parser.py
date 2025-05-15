#!/usr/bin/env python3
"""
path_parser.py

Parse REx output files to extract positive paths,
mapping both node URIs and edge codes to labels,
and producing a full 'labels' sequence per path.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Type aliases
NodePath    = List[str]
EdgePath    = List[str]
LabelMap    = Dict[str, str]
ParsedPaths = Dict[str, List[Dict[str, List[str]]]]


def load_tsv_labels(tsv_path: Path) -> LabelMap:
    """
    Load a TSV of <code>\\t<label> into a dict.
    """
    mapping: LabelMap = {}
    with tsv_path.open(encoding='utf-8') as f:
        for line in f:
            code, label = line.strip().split('\t', 1)
            mapping[code] = label
    return mapping


def parse_block(
    block: str,
    node_map: LabelMap,
    edge_map: LabelMap
) -> Optional[Tuple[NodePath, EdgePath, List[str]]]:
    """
    Parse a single REx block. If label==1, return:
      (raw_nodes, raw_edges, full_label_path)
    where full_label_path alternates node-label, arrow, edge-label, arrow, node-label...
    """
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if len(lines) < 4:
        return None

    route_line, edges_line, label_line, _score = lines[:4]
    try:
        label = int(label_line)
    except ValueError:
        return None
    if label != 1:
        return None

    # Raw splits
    raw_nodes = route_line.split('\t')
    raw_edges = [e for e in edges_line.split('\t') if e != 'NO_OP']

    # Map
    labeled_nodes = [ node_map.get(n, n) for n in raw_nodes ]
    labeled_edges = [ edge_map.get(e, e) for e in raw_edges ]

    # Build full alternating label path:
    full_labels: List[str] = []
    for i, node_lbl in enumerate(labeled_nodes):
        full_labels.append(node_lbl)
        if i < len(labeled_edges):
            full_labels.append(f"→ {labeled_edges[i]}")

    return raw_nodes, raw_edges, full_labels


def parse_paths_file(
    paths_file: Path,
    edges_tsv: Path,
    nodes_tsv: Path,
    separator: str = '#####################',
    chunk_sep: str   = '___'
) -> ParsedPaths:
    """
    Parse the entire REx output and return:
       { pair_key: [
           {'route': [...], 'edges': [...], 'labels': [...]},
           ...
         ]
       }
    """
    node_map = load_tsv_labels(nodes_tsv)
    edge_map = load_tsv_labels(edges_tsv)

    raw = paths_file.read_text(encoding='utf-8').strip()
    blocks = [b for b in raw.split(separator) if b.strip()]

    results: ParsedPaths = {}
    for block in blocks:
        lines = block.splitlines()
        pair_key = lines[0].strip()

        entries: List[Dict[str, List[str]]] = []
        for chunk in block.split(chunk_sep):
            parsed = parse_block(chunk, node_map, edge_map)
            if parsed:
                route, edges, labels = parsed
                entries.append({
                    'route': route,
                    'edges': edges,
                    'labels': labels
                })
        if entries:
            results[pair_key] = entries

    return results


def main(paths_file, edges_tsv, nodes_tsv):
    """
    Main function to parse paths file and print the results.
    """
    parsed = parse_paths_file(paths_file, edges_tsv, nodes_tsv)

    # Print to console
    for pair, entries in parsed.items():
        print(f"\nPair: {pair}")
        for e in entries:
            print(f"  Route : {e['route']}")
            print(f"  Edges : {e['edges']}")
            print(f"  Labels: {' '.join(e['labels'])}")

    # Dump JSON
    out = paths_file.with_suffix('.json')
    out.write_text(json.dumps(parsed, indent=2, ensure_ascii=False))
    print(f"\nParsed output saved to {out}")


if __name__ == "__main__":
    main(
        paths_file=Path("paths_CtD"),
        edges_tsv=Path("edges_labels.tsv"),
        nodes_tsv=Path("graph_labels.tsv")
    )
