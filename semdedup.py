import faiss
import numpy as np
from datasets import load_dataset
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# helpers


def sort_by_centroid_distance(embeddings, centroid, descending=True):
    distances = cdist(embeddings, centroid.reshape(1, -1), "euclidean")
    sorted_indices = np.argsort(distances, axis=0)
    if descending:
        sorted_indices = sorted_indices[::-1]
    return embeddings[sorted_indices.flatten()]


# Sentence transformer model for embeddings
model = SentenceTransformer("sentence-transformers/sentence-t5-xxl")

# Load the dataset
dataset = load_dataset("openwebtext", split="train")

# Apply the embed function to the 'text' column


def embed_text(examples):
    # Get the embeddings for the text
    embeddings = model.encode(examples["text"])

    # Convert to list as datasets work better with list type
    embeddings = embeddings.tolist()

    return {"embeddings": embeddings}


dataset = dataset.map(embed_text, batched=True, batch_size=16)

# Let's get the embeddings in a suitable format for clustering
embeddings = [embedding for example in dataset for embedding in example["embeddings"]]
embeddings = np.array(embeddings).astype("float32")  # FAISS uses float32

# Normalize the embeddings to unit length
embeddings = normalize(embeddings)

# perform clustering with FAISS
num_clusters = 11000
niter = 20
epsilon = 0.9  # Define the similarity threshold
niter = 20
verbose = True
d = embeddings.shape[1]  # dimension

# Initialize the clustering
cpu_index = faiss.Kmeans(d, num_clusters, niter=niter, verbose=verbose, spherical=True)
cpu_index.train(embeddings)

# Define GPU resources
ngpus = faiss.get_num_gpus()
res = [faiss.StandardGpuResources() for _ in range(ngpus)]

# Define cloner options
co = faiss.GpuMultipleClonerOptions()
co.shard = True

# Initialize the index. clone index to GPUs
index = faiss.index_cpu_to_all_gpus(cpu_index, co=co, resources=res)

D, I = index.search(embeddings, 1)
cluster_labels = I.reshape(-1)
cluster_centers = index.centroids

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
filtered_dataset = dataset.filter(lambda example, idx: idx in points_to_keep, with_indices=True)

print("Original dataset length: ", len(dataset))
print("Filtered dataset length: ", len(filtered_dataset))