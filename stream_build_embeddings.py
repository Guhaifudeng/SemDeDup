from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import jsonlines
import itertools

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

CHECKPOINT_FILE = 'embed_checkpoint.txt'

def write_checkpoint(batch_num):
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(batch_num))

def read_checkpoint():
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            return int(f.read())
    except FileNotFoundError:
        return 0

if __name__ == '__main__':
    #torch.cuda.empty_cache()
    data_stream_size = 16384
    chunk_size = 1024
    encode_batch_size = 32
    
    dataset = load_dataset(
        "openwebtext",
        split="train", 
        streaming=True
    )
    
    dataloader = DataLoader(
        dataset.with_format("torch"), 
        batch_size=data_stream_size
    )

    model = SentenceTransformer('sentence-transformers/sentence-t5-xxl')

    pool = model.start_multi_process_pool()

    checkpoint = read_checkpoint()
    print(f'Resuming from batch {checkpoint}.')

    with jsonlines.open('wikipedia_en_emb_text.jsonl', mode='a') as writer:
        for i, batch in enumerate(itertools.islice(tqdm(dataloader), checkpoint, None)):
            sentences = batch['text']
            batch_emb = model.encode_multi_process(
                sentences, 
                pool, 
                chunk_size=chunk_size, 
                batch_size=encode_batch_size
            )
            
            for sent, emb in zip(sentences, batch_emb):
                writer.write({"sentence": sent, "embedding": emb.tolist()})  # Convert numpy array to list before writing
            
            write_checkpoint(i + checkpoint)  # Update checkpoint after successful batch processing

    model.stop_multi_process_pool(pool)