import logging

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from nltk.tokenize import word_tokenize
from sentence_transformers import LoggingHandler, SentenceTransformer

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def chunk_text(texts, chunk_size: int = 256):
    """
    Given a list of texts, tokenizes each text into words, and chunks them into smaller texts of size 'chunk_size'.
    This function also provides the indices in the resulting list where the chunks for each original text start and end.

    Args:
        texts: List of texts to chunk.
        chunk_size: The desired size of each chunk. Defaults to 256.

    Returns:
        - A list of chunked texts (as strings).
        - A dictionary where each key is the index of an original text in the 'texts' list,
          and the value is a tuple of two indices (start and end) indicating the range in the chunked_texts list
          that the chunks for this original text occupy.
    """
    chunked_texts = []
    text_to_chunk_indices = {}

    for text_idx, text in enumerate(texts):
        # Tokenize the text
        tokens = word_tokenize(text)

        # Split the tokens into chunks of size chunk_size
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunks.append(tokens[i : i + chunk_size])

        # Detokenize chunks
        detokenized_chunks = [" ".join(chunk) for chunk in chunks]

        # Append chunked text to the list
        chunked_texts.extend(detokenized_chunks)

        # Record the indices of the chunks for this text
        start_idx = len(chunked_texts) - len(detokenized_chunks)
        end_idx = start_idx + len(detokenized_chunks)
        text_to_chunk_indices[text_idx] = (start_idx, end_idx)

    return chunked_texts, text_to_chunk_indices


def get_embeddings(
    texts,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 128,
):
    """
    Computes embeddings for a list of texts using a specified SentenceTransformer model. The computations are performed
    in a multi-process pool on all available CUDA devices.

    Args:
        texts: List of texts for which to compute embeddings.
        model_name: Name of the SentenceTransformer model to use for computing embeddings.
            Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        batch_size: Batch size for the model. Defaults to 128.

    Returns:
        List of embeddings computed by the model. Each element in the list is a numpy array
            representing the embedding of one text.
    """
    # Initialize model
    model = SentenceTransformer(model_name)

    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    # Compute embeddings
    embeddings = model.encode_multi_process(texts, pool, batch_size=batch_size)

    return embeddings


def reassemble(texts, chunked_texts, chunked_embeddings, text_to_chunk_indices):
    """
    Reassembles the original texts and their corresponding embeddings from the chunked texts and embeddings.

    Args:
        texts: List of original texts that were chunked.
        chunked_texts: List of chunked texts.
        chunked_embeddings: List of embeddings for each chunked text.
        text_to_chunk_indices: Dictionary mapping from each original text's index
            to the start and end indices of its chunks in the chunked_texts list.

    Returns:
        - A list of reassembled texts.
        - A list of reassembled embeddings for each text.
    """
    reassembled_texts = []
    reassembled_embeddings = []

    for text_idx in range(len(texts)):
        # Retrieve indices of chunks for this text
        start_idx, end_idx = text_to_chunk_indices[text_idx]

        # Reassemble text
        reassembled_text = " ".join(chunked_texts[start_idx:end_idx])
        reassembled_texts.append(reassembled_text)

        # Reassemble embeddings
        reassembled_embedding = np.concatenate(
            chunked_embeddings[start_idx:end_idx], axis=0
        )
        reassembled_embeddings.append(reassembled_embedding)

    return reassembled_texts, reassembled_embeddings


def pad_embeddings(embeddings):
    """
    Pads all embeddings in the list to match the length of the longest one by adding zeros to the end.

    Args:
        embeddings: List of numpy arrays, each representing an embedding. The arrays
            can have different lengths.

    Returns:
        List of numpy arrays, each representing a padded embedding. All the arrays
            have the same length, equal to the length of the longest embedding in the original list.
    """
    # Calculate maximum length of embeddings
    max_len = max(len(emb) for emb in embeddings)

    # Apply padding to all embeddings
    padded_embeddings = [
        np.pad(emb, (0, max_len - len(emb)), mode="constant") for emb in embeddings
    ]

    return padded_embeddings


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-v1",
        split="train",
        # cache_dir="/mnt/hdd/pile_8k",
    )
    dataset = dataset.filter(lambda example: bool(example["text"].strip()))
    texts = dataset["text"]

    chunked_texts, text_to_chunk_indices = chunk_text(texts, chunk_size=256)

    chunked_embeddings = get_embeddings(chunked_texts)

    reassembled_texts, reassembled_embeddings = reassemble(
        texts, chunked_texts, chunked_embeddings, text_to_chunk_indices
    )

    padded_embeddings = pad_embeddings(reassembled_embeddings)

    # Assume that `full_texts` and `full_embeddings` are your processed data
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "text": reassembled_texts,
                    "embeddings": reassembled_embeddings,
                    "pad_embeddings": padded_embeddings,
                }
            )
            # Add more splits here if you have them
        }
    )

    # Push the dataset to the Hub
    dataset_dict.push_to_hub("wikitext-2-v1-0")
