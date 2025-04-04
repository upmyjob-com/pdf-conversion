import logging
import time
import torch
from sentence_transformers import SentenceTransformer
import config
from tqdm import tqdm # For progress bar during embedding

logger = logging.getLogger(__name__)

# --- Model Loading ---
model = None
embedding_dim = None

def load_embedding_model():
    """Loads the Sentence Transformer model specified in config.py."""
    global model, embedding_dim
    if model is not None:
        logger.debug("Embedding model already loaded.")
        return model

    model_name = config.EMBEDDING_MODEL_NAME
    device = config.DEVICE
    logger.info(f"Loading embedding model '{model_name}' onto device '{device}'...")
    start_time = time.time()
    try:
        model = SentenceTransformer(model_name, device=device)
        # Check if model loaded successfully and get embedding dimension
        if hasattr(model, 'get_sentence_embedding_dimension'):
            embedding_dim = model.get_sentence_embedding_dimension()
        elif hasattr(model, 'encode'):
             # Try encoding a dummy sentence to infer dimension
             try:
                 dummy_embedding = model.encode("test")
                 embedding_dim = len(dummy_embedding)
             except Exception:
                 logger.warning("Could not automatically determine embedding dimension.")
                 embedding_dim = None # Mark as unknown
        else:
             logger.warning("Could not determine embedding dimension from the loaded model.")
             embedding_dim = None

        end_time = time.time()
        logger.info(f"Successfully loaded model '{model_name}' in {end_time - start_time:.2f} seconds.")
        if embedding_dim:
            logger.info(f"Model embedding dimension: {embedding_dim}")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {e}", exc_info=True)
        model = None # Ensure model is None if loading failed
        embedding_dim = None
        return None

def get_embedding_dimension():
    """Returns the dimension of the loaded embedding model."""
    global embedding_dim
    if model is None:
        load_embedding_model() # Attempt to load if not already loaded
    return embedding_dim


def generate_embeddings(texts: list[str], batch_size: int = config.EMBEDDING_BATCH_SIZE) -> list[list[float]] | None:
    """
    Generates embeddings for a list of text chunks using the loaded model.

    Args:
        texts: A list of strings (text chunks) to embed.
        batch_size: The number of texts to process in each batch.

    Returns:
        A list of embeddings (each embedding is a list of floats),
        or None if the model is not loaded or an error occurs.
    """
    global model
    if model is None:
        logger.error("Embedding model is not loaded. Cannot generate embeddings.")
        return None
    if not texts:
        logger.warning("Received empty list of texts for embedding.")
        return []

    logger.info(f"Generating embeddings for {len(texts)} text chunks with batch size {batch_size}...")
    all_embeddings = []
    start_time = time.time()

    try:
        # Use tqdm for progress bar visualization
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Chunks", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            # The encode method handles moving data to the model's device
            batch_embeddings = model.encode(
                batch_texts,
                batch_size=len(batch_texts), # Inner batch size for the model call
                show_progress_bar=False # Disable sentence-transformers internal bar, use tqdm
            )
            all_embeddings.extend(batch_embeddings.tolist()) # Convert numpy arrays to lists

        end_time = time.time()
        logger.info(f"Successfully generated {len(all_embeddings)} embeddings in {end_time - start_time:.2f} seconds.")
        return all_embeddings

    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)
        return None


# Example usage
if __name__ == '__main__':
    from utils import setup_logging
    setup_logging()

    # Ensure model is loaded
    loaded_model = load_embedding_model()

    if loaded_model:
        print(f"Model loaded successfully. Embedding dimension: {get_embedding_dimension()}")
        sample_chunks = [
            "This is the first chunk of text.",
            "Here is another piece of text to embed.",
            "Sentence transformers are useful for generating dense vector representations.",
            "This is chunk number four.",
            "The final chunk for this example test."
        ]
        embeddings = generate_embeddings(sample_chunks, batch_size=2)

        if embeddings:
            logger.info(f"\n--- Generated {len(embeddings)} Embeddings ---")
            for i, emb in enumerate(embeddings):
                print(f"--- Embedding {i+1} (Dim: {len(emb)}) ---")
                # Print only the first few dimensions for brevity
                print(f"[{', '.join(map(str, emb[:5]))}, ...]")
                print("-" * 20)
        else:
            logger.error("Embedding generation failed.")
    else:
        logger.error("Failed to load the embedding model. Cannot run example.")