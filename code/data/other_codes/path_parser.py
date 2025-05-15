#!/usr/bin/env python3
"""
path_parser.py

Parse REx output files to extract positive paths and their edge sequences.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Type aliases
NodePath = List[str]
EdgePath = List[str]
LabelMap = Dict[str, str]
ParsedPaths = Dict[str, List[Dict[str, EdgePath]]]


def load_edge_labels(tsv_path: Path) -> LabelMap:
    """
    Load edge label mappings from a TSV file.
    Each line should be: <edge_code>\\t<label>
    """
    mapping: LabelMap = {}
    with tsv_path.open(encoding='utf-8') as f:
        for line in f:
            code, label = line.strip().split('\t', 1)
            mapping[code] = label
    return mapping


def parse_block(block: str, edge_map: LabelMap) -> Optional[Tuple[NodePath, EdgePath]]:
    """
    Parse a single block of REx output. Returns (route, mapped_edges)
    if the block contains a positive path (label == 1), else None.
    """
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if len(lines) < 4:
        return None

    route_line, edges_line, label_line, _score_line = lines[:4]
    try:
        label = int(label_line)
    except ValueError:
        return None

    # Only keep positively labeled paths
    if label != 1:
        return None

    # Split route (nodes) and edges
    node_route = route_line.split('\t')
    edge_codes = [e for e in edges_line.split('\t') if e != 'NO_OP']

    # Map edge codes to labels, skipping missing ones
    mapped_edges = [edge_map.get(code, code) for code in edge_codes]
    return node_route, mapped_edges


def parse_paths_file(
    file_path: Path,
    edges_tsv: Path,
    separator: str = '#####################',
    chunk_sep: str = '___'
) -> ParsedPaths:
    """
    Parse an entire REx output file and return a dict mapping each query pair
    to a list of dicts with 'route' and 'edges' for each positive path.
    """
    edge_map = load_edge_labels(edges_tsv)
    raw_text = file_path.read_text(encoding='utf-8').strip()
    blocks = [blk for blk in raw_text.split(separator) if blk.strip()]

    results: ParsedPaths = {}

    for block in blocks:
        lines = block.splitlines()
        pair_key = lines[0].strip()

        positive_paths: List[Dict[str, EdgePath]] = []
        for chunk in block.split(chunk_sep):
            parsed = parse_block(chunk, edge_map)
            if parsed:
                route, edges = parsed
                positive_paths.append({'route': route, 'edges': edges})

        if positive_paths:
            results[pair_key] = positive_paths

    return results


def main(paths_file, edges_tsv):

    parsed = parse_paths_file(paths_file, edges_tsv)

    # Print results
    for pair, entries in parsed.items():
        print(f"\nPair: {pair}")
        for entry in entries:
            print(f"  Route: {' -> '.join(entry['route'])}")
            print(f"  Edges: {entry['edges']}")

    # Optionally, save to JSON for easy downstream use
    out_json = paths_file.with_suffix('.json')
    out_json.write_text(json.dumps(parsed, indent=2, ensure_ascii=False))
    print(f"\nParsed output saved to {out_json}")


if __name__ == "__main__":
    main("paths_CtD", "edges_labels.tsv")
    #Note
    # paths_CtD should be the path to your REx output file
    # edges_labels.tsv should be the path to your edge label dataset