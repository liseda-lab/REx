# -*- coding: utf-8 -*-
"""
Author: Susana Nunes
Description:
    This script performs KMeans clustering on node embeddings
    divided by edge types using the FAISS library. It reads the
    embeddings from a JSON file, clusters them, and saves the
    clustering results in another JSON file.
"""

import faiss
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class ClusteringNodes:
    def __init__(self, embs_path: Path, cluster_path: Path):
        self.embs_path = embs_path
        self.cluster_path = cluster_path
        self.clustered_nodes: dict = {}
        self.embs: dict = self.load_embeddings()
        self.centroid_mapping: dict = {}

    def load_embeddings(self):
        print("🔹 [Checkpoint 1] Loading embeddings...", flush=True)
        with open(self.embs_path, "r") as f:
            embs = json.load(f)
        print(f"Loaded {len(embs)} embeddings.", flush=True)
        return embs

    def create_tensor(self, embs):
        print("🔹 [Checkpoint 2] Converting embeddings to NumPy array...", flush=True)
        keys = list(embs.keys())
        values = [embs[key] for key in keys]
        x = np.array(values)
        print(f"Shape of embedding matrix: {x.shape}", flush=True)
        return x, keys

    def train_kmeans(self, input_data, ncentroids, niter=20, verbose=True):
        print("🔹 [Checkpoint 3] Preparing and running FAISS KMeans clustering...", flush=True)
        d = input_data.shape[1]
        print(f"Clustering {input_data.shape[0]} nodes into {ncentroids} centroids.", flush=True)
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(input_data)
        print("KMeans clustering complete.", flush=True)
        return kmeans

    def compute_centroid_mapping(self, kmeans, input_data, keys, plot=False):
        print("🔹 [Checkpoint 4] Mapping vectors to centroids...", flush=True)
        _, I = kmeans.index.search(input_data, 1)
        print("Mapping complete.", flush=True)

        if plot:
            print("🔹 [Checkpoint 5] Plotting and saving histogram...", flush=True)
            plt.hist(I, bins=kmeans.niter)
            plt.savefig('clusters.png')
            print("Histogram saved as clusters.png", flush=True)

        print("🔹 [Checkpoint 6] Building centroid mapping...", flush=True)
        centroid_mapping = {}
        for idx, name in enumerate(keys):
            centroid_index = I[idx][0]
            centroid_mapping.setdefault(int(centroid_index), []).append(name)
        centroid_mapping = dict(sorted(centroid_mapping.items()))
        print("Centroid mapping built.", flush=True)
        return centroid_mapping

    def save_mapping(self, centroid_mapping):
        print("🔹 [Checkpoint 7] Saving clustering results to JSON...", flush=True)
        with open(self.cluster_path, "w") as f:
            json.dump(centroid_mapping, f, indent=4)
        print(f"Clustering results saved to {self.cluster_path}", flush=True)

    def cluster_nodes(self, embs=None, save=True, plot=False, ncentroids=None):
        print("[Checkpoint] Clustering nodes...", flush=True)
        if embs is None:
            embs = self.embs
        tensor, keys = self.create_tensor(embs)
        node_number = len(embs)
        if ncentroids is None:
            ncentroids = int(0.1 * node_number)
        kmeans = self.train_kmeans(tensor, ncentroids)
        cluster = self.compute_centroid_mapping(kmeans, tensor, keys, plot)

        if save:
            self.save_mapping(cluster)

        print("All steps completed successfully!", flush=True)
        return cluster

    def cluster_by_edge(self):
        print("[Checkpoint] Clustering nodes by edge types...", flush=True)
        edge_embs = {}
        for edge in self.embs:
            print(f"\n🔸 Clustering for edge type: {edge}", flush=True)
            cluster = self.cluster_nodes(self.embs[edge], save=False, plot=False)
            edge_embs[edge] = cluster

        self.save_mapping(edge_embs)
        print("All edge-type clustering completed successfully!", flush=True)
        return edge_embs


if __name__ == "__main__":
    embs_path = Path('embeddings_edge_type.json')
    output_path = Path("clustering_default_10percent_edgeType.json")

    clustering = ClusteringNodes(embs_path, output_path)
    clustering.cluster_by_edge()
