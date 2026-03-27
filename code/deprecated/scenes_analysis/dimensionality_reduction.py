from mario_scenes.load_data import load_annotation_data
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import os.path as op
import os
import logging


def dimensionality_reduction(
    df_features,
    method="pca",
    n_components=2,
    random_state=42,
    dr_params=None
):
    """
    Apply a dimensionality reduction technique to the given feature DataFrame.

    Parameters
    ----------
    df_features : pd.DataFrame
        A DataFrame containing only the feature columns (no IDs or labels).
        All features should be numeric (or pre-encoded as numeric).

    method : str, optional
        The dimensionality reduction method to use. Options:
          - "none" : return the original data without changes
          - "pca"  : Principal Component Analysis
          - "umap" : Uniform Manifold Approximation and Projection
          - "tsne" : t-distributed Stochastic Neighbor Embedding
        Default is "pca".

    n_components : int, optional
        Number of components (dimensions) to project down to.
        Typically 2 or 3 if you're plotting. Default is 2.

    random_state : int, optional
        Random seed for reproducibility in methods that support it
        (PCA, UMAP, t-SNE). Default is 42.

    dr_params : dict, optional
        A dictionary of additional parameters for the DR method.
        Examples:
          - For PCA:  {"svd_solver": "full"}
          - For UMAP: {"n_neighbors": 15, "min_dist": 0.1}
          - For t-SNE: {"perplexity": 30, "learning_rate": 200}
        Default is None (no extra parameters).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the transformed features, whose columns are named
        DR_1, DR_2, ..., DR_{n_components}.
        If method="none", returns a copy of df_features.
    """
    if dr_params is None:
        dr_params = {}

    method = method.lower()

    # If no DR is needed, just return the original data.
    if method == "none":
        return df_features.copy()

    X = df_features.values  # Convert DF to numpy array

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state, **dr_params)
        embedding = reducer.fit_transform(X)

    elif method == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, **dr_params)
        embedding = reducer.fit_transform(X)

    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state, **dr_params)
        embedding = reducer.fit_transform(X)

    else:
        raise ValueError("method must be one of ['none', 'pca', 'umap', 'tsne'].")

    # Build a new DataFrame of the results
    dr_columns = [f"DR_{i+1}" for i in range(n_components)]
    df_reduced = pd.DataFrame(embedding, columns=dr_columns, index=df_features.index)

    return df_reduced


def main():
    # Prepare outputs
    OUTPUT_DIR = op.join("outputs", "dimensionality_reduction")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('Outputs folder created')

    # Load inputs
    df_features = load_annotation_data()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Verbose level
    verbose = 1

    if verbose > 0:
        logger.info("Starting dimensionality reduction on scene annotations")

    # 1) PCA
    if verbose > 0:
        logger.info("Applying PCA...")
    df_pca = dimensionality_reduction(
        df_features,
        method="pca",
        n_components=2,
        random_state=42,
        dr_params={"svd_solver": "full"}
    )
    df_pca.to_csv(op.join(OUTPUT_DIR, "pca.csv"))
    if verbose > 0:
        logger.info("PCA completed and saved to pca.csv")

    # 2) UMAP
    if verbose > 0:
        logger.info("Applying UMAP...")
    df_umap = dimensionality_reduction(
        df_features,
        method="umap",
        n_components=2,
        random_state=42,
        dr_params={"n_neighbors": 15, "min_dist": 0.1}
    )
    df_umap.to_csv(op.join(OUTPUT_DIR, "umap.csv"))
    if verbose > 0:
        logger.info("UMAP completed and saved to umap.csv")

    # 3) t-SNE
    if verbose > 0:
        logger.info("Applying t-SNE...")
    df_tsne = dimensionality_reduction(
        df_features,
        method="tsne",
        n_components=2,
        random_state=42,
        dr_params={"perplexity": 30, "learning_rate": 200}
    )
    df_tsne.to_csv(op.join(OUTPUT_DIR, "tsne.csv"))
    if verbose > 0:
        logger.info("t-SNE completed and saved to tsne.csv")

    if verbose > 0:
        logger.info("Dimensionality reduction process completed.")

if __name__ == "__main__":
    main()