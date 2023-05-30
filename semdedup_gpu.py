import faiss
import numpy as np
from datasets import load_dataset
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from cuml.preprocessing import normalize

# helpers

def sort_by_centroid_distance(embeddings, centroid, descending=True):
    distances = cdist(embeddings, centroid.reshape(1, -1), "euclidean")
    sorted_indices = np.argsort(distances, axis=0)
    if descending:
        sorted_indices = sorted_indices[::-1]
    return embeddings[sorted_indices.flatten()]


# Load the dataset
dataset = load_dataset("openwebtext_emb", split="train")

# get the embeddings for clustering
embeddings = dataset["embedding"]

# Normalize the embeddings
embeddings = normalize(embeddings)

# perform clustering with FAISS
num_clusters = 11000
niter = 20
epsilon = 0.09  # Define the similarity threshold
verbose = True
d = embeddings.shape[1]  # dimension

# Initialize the clustering
kmeans = faiss.Kmeans(d, num_clusters, niter=niter, verbose=verbose, spherical=True, gpu=True)
kmeans.train(embeddings)

D, I = kmeans.index.search(embeddings, 1)
cluster_labels = I.reshape(-1)
cluster_centers = kmeans.centroids

points_to_keep = []

for i in range(num_clusters):
    # filter embeddings of the current cluster
    cluster_i_embeddings = embeddings[cluster_labels == i]

    # sort the cluster embeddings by the distance to the cluster centroid
    cluster_i_embeddings = sort_by_centroid_distance(
        cluster_i_embeddings, cluster_centers[i], descending=True
    )

    # compute the pairwise cosine similarity between embeddings
    pairwise_sim_matrix = cosine_similarity(cluster_i_embeddings)

    # get upper triangular part of the matrix (excluding the diagonal)
    triu_sim_matrix = np.triu(pairwise_sim_matrix, k=1)

    # find max value in each column
    M = np.max(triu_sim_matrix, axis=0)[0]

    # Check if the maximum similarity <= the threshold.
    points_to_keep_from_cluster_i = cluster_i_embeddings[M <= 1 - epsilon]

    # add the points to keep to the list
    points_to_keep.extend(points_to_keep_from_cluster_i)
