"""
Hierarchical Clustering for Mario Scenes

This module performs hierarchical clustering on scene annotation features to group
scenes by gameplay similarity. Supports multiple cluster counts and generates
summary statistics for each clustering solution.

Main Functions:
    - generate_clusters(): Compute hierarchical clustering for multiple cluster counts
    - summary_clusters(): Generate per-cluster statistics and feature profiles

Output:
    Pickle file with cluster assignments and summaries for each requested cluster count.
"""

import argparse
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from mario_scenes.load_data import load_annotation_data, load_reduced_data
import pickle
import os
import os.path as op


def generate_clusters(list_n_clusters=[10]):
    """
    Perform hierarchical clustering on scene annotations.

    Uses Ward linkage with Euclidean distance to cluster scenes based on their
    27-dimensional feature vectors. Cuts the dendrogram at multiple heights to
    generate solutions with varying numbers of clusters.

    Parameters
    ----------
    list_n_clusters : list of int, default=[10]
        List of cluster counts to generate (e.g., [5, 10, 15, 20]).
        Each value produces a different clustering solution by cutting
        the hierarchy at a different height.

    Returns
    -------
    dict
        Nested dictionary with structure:
        {
            0: {
                'index': np.ndarray,  # Cluster assignments for each scene
                'n_clusters': int,    # Number of clusters
                'summary': dict       # Output from summary_clusters()
            },
            1: {...},
            ...
        }
        Keys are enumeration indices (not cluster counts).

    Examples
    --------
    >>> clusters = generate_clusters([10, 20, 30])
    >>> print(clusters[0]['n_clusters'])
    10
    >>> print(clusters[0]['summary'][0])
    {'n_scenes': 23, 'labels': ..., 'homogeneity': ...}

    Notes
    -----
    Uses scipy.cluster.hierarchy for clustering. The dendrogram leaf order
    is preserved but not currently used in output.
    """
    X = load_annotation_data()
    hier = linkage(X, method='ward', metric='euclidean')  # scipy's hierarchical clustering
    res = dendrogram(hier, labels=X.index, get_leaves=True)  # Generate a dendrogram from the hierarchy
    order = res.get('leaves')  # Extract the order on papers from the dendrogram
    part = {}
    for ind, n_clusters in enumerate(list_n_clusters):
        # Cut the hierarchy and turn the parcellation into a dataframe
        clusters = np.squeeze(cut_tree(hier, n_clusters=n_clusters))
        part[ind] = {
            'index': clusters,
            'n_clusters': n_clusters,
            'summary': summary_clusters(clusters)
        }
    return part

def summary_clusters(part, threshold=0.05):
    """
    Generate summary statistics for each cluster.

    Computes per-cluster feature profiles by averaging annotation values within
    each cluster. Identifies dominant features (those exceeding threshold) and
    calculates homogeneity scores.

    Parameters
    ----------
    part : np.ndarray
        1D array of cluster assignments, same length as number of scenes.
        Values are cluster IDs (e.g., 0, 1, 2, ..., n_clusters-1).
    threshold : float, default=0.05
        Minimum average feature value to include in cluster 'labels'.
        Features below this threshold are considered absent in the cluster.

    Returns
    -------
    dict
        Nested dictionary keyed by cluster ID:
        {
            0: {
                'n_scenes': int,                # Number of scenes in cluster
                'labels': pd.Series,            # Features above threshold, sorted descending
                'homogeneity': pd.Series        # Average value for all 27 features
            },
            1: {...},
            ...
        }

    Examples
    --------
    >>> X = load_annotation_data()
    >>> X['clusters'] = cluster_assignments  # From generate_clusters()
    >>> summary = summary_clusters(cluster_assignments, threshold=0.1)
    >>> print(summary[0]['labels'])
    Enemy       0.85
    Gap         0.62
    Roof        0.15
    dtype: float64

    Notes
    -----
    'labels' contains only features exceeding the threshold, sorted by frequency.
    'homogeneity' contains all 27 features regardless of threshold.
    """
    X = load_annotation_data()
    X['clusters'] = part
    all_cluster = X.groupby('clusters').mean()
    summary = {}
    for index, row in all_cluster.iterrows():
        # Filter, sort in descending order
        summary[index] = {
            'n_scenes': len(X[X['clusters'] == index]),
            'labels': row[row > threshold].sort_values(ascending=False),
            'homogeneity': row
        }
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hierarchical clusters and save results as a pickle file.")
    parser.add_argument("--n_clusters", type=int, nargs='+', default=[10], help="List of numbers of clusters to generate.")

    args = parser.parse_args()

    # Generate clusters
    clusters = generate_clusters(args.n_clusters)

    # Create output directory and save results
    BASE_DIR = op.dirname(op.dirname(op.dirname(op.dirname(op.abspath(__file__)))))
    OUTPUT_DIR = op.join(BASE_DIR, 'outputs', 'cluster_scenes')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = op.join(OUTPUT_DIR, "hierarchical_clusters.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(clusters, f)

    print(f"Clustering completed. Results saved to {output_file}")
