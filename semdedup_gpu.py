import faiss
import numpy as np
import torch
from datasets import load_dataset
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer, LoggingHandler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# helpers


def sort_by_centroid_distance(embeddings, centroid, descending=True):
    distances = cdist(embeddings, centroid.reshape(1, -1), "euclidean")
    sorted_indices = np.argsort(distances, axis=0)
    if descending:
        sorted_indices = sorted_indices[::-1]
    return embeddings[sorted_indices.flatten()]


# Load the dataset
dataset = load_dataset("openwebtext", split="train")

# Save the original length
original_length = len(dataset)

dataset = load_dataset("conceptofmind/facebook_ads", split="train")

model = SentenceTransformer('sentence-transformers/sentence-t5-xxl')

sentences = dataset["text"]

#Start the multi-process pool on all available CUDA devices
pool = model.start_multi_process_pool()

#Compute the embeddings using the multi-process pool
embeddings = model.encode_multi_process(sentences, pool, batch_size=16)

embeddings_list = embeddings.tolist()

dataset = dataset.add_column("embeddings", embeddings_list)

# get the embeddings for clustering
embeddings = [embedding for example in dataset for embedding in example["embeddings"]]

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

# convert to numpy array

points_to_keep = np.array(points_to_keep)

# Filter the original dataset
filtered_dataset = dataset.filter(
    lambda example, idx: idx in points_to_keep, with_indices=True
)

print("Original dataset length: ", len(dataset))
print("Filtered dataset length: ", len(filtered_dataset))