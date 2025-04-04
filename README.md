# PDF Text Extraction, Chunking, Embedding, and Storage Pipeline

This project provides a set of Python scripts to process PDF documents, extract their text content, split the text into manageable chunks, generate vector embeddings for each chunk using Hugging Face Sentence Transformers, and store the results in a local Qdrant vector database.

The pipeline is designed to handle potentially large PDF files and utilizes multiprocessing for CPU-bound tasks (PDF text extraction) and GPU acceleration (CUDA) for embedding generation if available.

## Features

* **PDF Text Extraction:** Uses `PyMuPDF` (fitz) for robust text extraction.
* **Text Chunking:** Employs `langchain`'s `RecursiveCharacterTextSplitter` with token-based length calculation using the specified model's tokenizer.
* **Vector Embeddings:** Leverages `sentence-transformers` library to generate embeddings using models from the Hugging Face Hub. Supports CUDA for GPU acceleration.
* **Vector Storage:** Stores embeddings and associated text chunks in a Qdrant vector database collection.
* **Parallel Processing:** Uses Python's `multiprocessing` to speed up PDF processing across multiple CPU cores.
* **Configurable:** Pipeline parameters (paths, model names, chunk sizes, Qdrant details, etc.) are managed in `config.py`.
* **Resource Aware:** Designed to process files sequentially in the main loop to manage memory usage, with configurable batch sizes for embedding and Qdrant upserts.

## Project Structure

```
/home/ron/pdf-conversion/
├── pdfs/                 # Directory to place your input PDF files
├── .venv/                # Python virtual environment (Recommended)
├── config.py             # Configuration settings for the pipeline
├── main.py               # Main orchestration script to run the pipeline
├── pdf_extractor.py      # Module for extracting text from PDFs
├── text_chunker.py       # Module for splitting text into chunks
├── embedder.py           # Module for generating sentence embeddings
├── qdrant_storer.py      # Module for interacting with Qdrant
├── utils.py              # Utility functions (e.g., logging setup)
├── requirements.txt      # Python package dependencies
└── README.md             # This documentation file
```

## Setup

1. **Clone/Download:** Obtain the project files.
2. **Create Virtual Environment (Recommended):**
   
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/macOS
   # .\ .venv\Scripts\activate  # On Windows
   ```
3. **Install Dependencies:**
   
   ```bash
   pip install -r requirements.txt
   ```
   * **Note on PyTorch/CUDA:** `sentence-transformers` will pull in `torch`. Ensure your PyTorch installation matches your CUDA version if you intend to use GPU acceleration. You might need to install a specific PyTorch version manually first by following instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).
4. **Qdrant Setup:** Ensure you have a Qdrant instance running and accessible. By default, the script expects it at `localhost:6333`. You can run Qdrant easily using Docker:
   
   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
       -v $(pwd)/qdrant_storage:/qdrant/storage \
       qdrant/qdrant:latest
   ```
   
   *(See [Qdrant documentation](https://qdrant.tech/documentation/guides/installation/) for more options.)*
5. **Place PDFs:** Copy the PDF files you want to process into the `pdfs/` directory.

## Configuration

Modify the `config.py` file to adjust pipeline behavior:

* `PDF_INPUT_DIR`: Path to the directory containing PDFs (relative to project root).
* `CHUNK_SIZE`, `CHUNK_OVERLAP`: Parameters for text chunking (in tokens).
* `EMBEDDING_MODEL_NAME`: The Hugging Face Sentence Transformer model to use.
* `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION_NAME`: Details for your Qdrant instance.
* `CPU_WORKER_COUNT`: Number of CPU cores for parallel PDF processing (defaults to all available).
* `DEVICE`: Automatically set to `cuda` if available, otherwise `cpu`.
* `EMBEDDING_BATCH_SIZE`: Batch size for embedding generation (adjust based on GPU VRAM).
* `QDRANT_BATCH_SIZE`: Batch size for upserting data to Qdrant.

## Running the Pipeline

1. Ensure your virtual environment is activated (`source .venv/bin/activate`).
2. Make sure your Qdrant server is running.
3. Place your PDF files in the `pdfs/` directory.
4. Execute the main script:
   
   ```bash
   python main.py
   ```

The script will:

* Load the embedding model.
* Connect to Qdrant and ensure the collection exists.
* Find PDF files in the input directory.
* Process each PDF in parallel:
  * Extract text.
  * Chunk text.
  * Generate embeddings (using CUDA if available).
  * Upsert embeddings and text chunks to Qdrant.
* Log progress and provide a summary upon completion.

Check the console output and `pipeline.log` (if configured) for detailed information and potential errors.