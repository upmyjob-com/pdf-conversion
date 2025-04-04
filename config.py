import os
import torch

# --- File Paths ---
PDF_INPUT_DIR = "pdfs"  # Relative path to the directory containing input PDF files

# --- Chunking Parameters ---
CHUNK_SIZE = 512  # Target size for text chunks (in tokens)
CHUNK_OVERLAP = 50  # Number of tokens to overlap between consecutive chunks

# --- Embedding Model ---
# Recommended model: efficient and good performance
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Alternative (if more VRAM available and higher quality needed):
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# EMBEDDING_MODEL_NAME = "thenlper/gte-large" # Requires transformer install

# --- Qdrant Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "pdf_embeddings"
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None) # Uncomment if using API key
# QDRANT_GRPC_PORT = 6334 # Default gRPC port if needed

# --- Processing Parameters ---
# Determine the number of CPU cores to use for parallel processing (e.g., PDF extraction)
# Uses all available cores by default. Adjust if needed.
CPU_WORKER_COUNT = os.cpu_count()

# Determine the device for embedding computation (CPU or CUDA GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Batch size for embedding generation. Adjust based on GPU VRAM.
# Smaller batches use less VRAM but might be slower overall.
EMBEDDING_BATCH_SIZE = 32 # Good starting point for models like all-MiniLM-L6-v2

# Batch size for upserting data to Qdrant
QDRANT_BATCH_SIZE = 128

# --- Resource Limits ---
# Placeholder for potential future use, logic currently handles memory by processing file-by-file
MAX_RAM_GB = 100

# --- Logging ---
LOG_LEVEL = "INFO" # e.g., DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'