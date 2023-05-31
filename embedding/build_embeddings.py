from sentence_transformers import SentenceTransformer, LoggingHandler
from datasets import load_dataset
from sklearn.preprocessing import normalize
import time

import logging

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset("conceptofmind/facebook_ads", split="train")

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    sentences = dataset["text"]

    #Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    # Record the start time
    print("Compute embeddings")
    start_time = time.time()

    #Compute the embeddings using the multi-process pool
    embeddings = model.encode_multi_process(
        sentences, 
        pool, 
        batch_size=32,
    )

    # Print the time taken
    end_time = time.time()
    print("Time taken:", end_time - start_time, "seconds")

    embeddings = normalize(embeddings)

    embeddings_list = embeddings.tolist()

    fdataset = dataset.add_column("normalized_embeddings", embeddings_list)

    fdataset.push_to_hub('conceptofmind/facebook_ads')