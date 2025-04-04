import logging
from transformers import AutoTokenizer # To count tokens accurately for the model
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config
import time

logger = logging.getLogger(__name__)

# Load the tokenizer for the specified embedding model to calculate token length
try:
    tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
    logger.info(f"Successfully loaded tokenizer for model: {config.EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load tokenizer for {config.EMBEDDING_MODEL_NAME}. Chunking might be inaccurate. Error: {e}", exc_info=True)
    # Fallback to simple character length if tokenizer fails
    tokenizer = None

def count_tokens(text: str) -> int:
    """Counts tokens using the loaded Hugging Face tokenizer."""
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        # Fallback or raise error if tokenizer is essential
        logger.warning("Tokenizer not available, using character count for length.")
        return len(text)

def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into chunks using RecursiveCharacterTextSplitter based on token count.

    Args:
        text: The input text string to be chunked.
        chunk_size: The target maximum size of each chunk (in tokens).
        chunk_overlap: The number of tokens to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if not text:
        logger.warning("Received empty text for chunking.")
        return []

    logger.debug(f"Starting chunking process with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    start_time = time.time()

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens, # Use our token counter
        add_start_index=False, # We don't need start index for this pipeline
        separators=["\n\n", "\n", ". ", " ", ""], # Default separators are usually good
        keep_separator=True # Keep separators to maintain context
    )

    try:
        chunks = text_splitter.split_text(text)
        num_chunks = len(chunks)
        end_time = time.time()
        logger.info(f"Successfully split text into {num_chunks} chunks in {end_time - start_time:.2f} seconds.")
        if num_chunks > 0:
             # Log stats about first and last chunk for debugging
             first_chunk_len = count_tokens(chunks[0])
             last_chunk_len = count_tokens(chunks[-1])
             logger.debug(f"First chunk length: {first_chunk_len} tokens. Last chunk length: {last_chunk_len} tokens.")
        return chunks
    except Exception as e:
        logger.error(f"Error during text splitting: {e}", exc_info=True)
        return [] # Return empty list on error

# Example usage
if __name__ == '__main__':
    from utils import setup_logging
    setup_logging()

    sample_text = """
    This is the first paragraph. It contains several sentences and provides an introduction.
    Tokens are counted using the specified model's tokenizer.

    Here is the second paragraph. It discusses further details and expands on the initial topic.
    The RecursiveCharacterTextSplitter tries to split along semantic boundaries like paragraphs first,
    then sentences, then words, ensuring chunks remain coherent. Overlap helps maintain context
    between adjacent chunks. The target chunk size is {config.CHUNK_SIZE} tokens.

    Finally, a third paragraph to ensure we have enough text to potentially create multiple chunks.
    Testing the overlap feature is important. This sentence bridges the gap.
    Another sentence follows. And one more for good measure.
    """

    logger.info(f"Original text length: {count_tokens(sample_text)} tokens")
    chunks = chunk_text(sample_text)

    if chunks:
        logger.info(f"\n--- Created {len(chunks)} Chunks ---")
        for i, chunk in enumerate(chunks):
            print(f"--- Chunk {i+1} ({count_tokens(chunk)} tokens) ---")
            print(chunk)
            print("-" * 20)
    else:
        logger.error("Chunking failed.")