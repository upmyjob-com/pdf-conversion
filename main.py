import os
import glob
import logging
import time
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import config
import utils
import pdf_extractor
import text_chunker
import embedder
import qdrant_storer

# Setup logging as early as possible
utils.setup_logging()
logger = logging.getLogger(__name__)

def process_pdf(pdf_path: str, success_list: list, failure_list: list):
    """
    Worker function to process a single PDF file.
    Extracts text, chunks, embeds, and upserts to Qdrant.

    Args:
        pdf_path: Path to the PDF file.
        success_list: A multiprocessing Manager list to record successful files.
        failure_list: A multiprocessing Manager list to record failed files.
    """
    base_filename = os.path.basename(pdf_path)
    logger.info(f"[{base_filename}] Starting processing...")
    start_time = time.time()

    try:
        # 1. Extract Text
        extracted_text = pdf_extractor.extract_text_from_pdf(pdf_path)
        if not extracted_text:
            logger.error(f"[{base_filename}] Failed to extract text or text was empty. Skipping.")
            failure_list.append(base_filename)
            return

        # 2. Chunk Text
        chunks = text_chunker.chunk_text(extracted_text)
        if not chunks:
            logger.error(f"[{base_filename}] Failed to chunk text or no chunks were generated. Skipping.")
            failure_list.append(base_filename)
            return
        logger.info(f"[{base_filename}] Generated {len(chunks)} chunks.")

        # 3. Generate Embeddings
        # Note: embedder.load_embedding_model() should have been called in the main process.
        # The model object might be shared via fork (on Linux/macOS).
        # If issues arise, model loading might need to happen per-process.
        embeddings = embedder.generate_embeddings(chunks)
        if not embeddings or len(embeddings) != len(chunks):
            logger.error(f"[{base_filename}] Failed to generate embeddings or mismatch in count. Skipping.")
            failure_list.append(base_filename)
            return
        logger.info(f"[{base_filename}] Generated {len(embeddings)} embeddings.")

        # 4. Upsert to Qdrant
        # Note: qdrant_storer.get_qdrant_client() handles client initialization per process if needed.
        # Collection should exist from the main process check.
        qdrant_storer.upsert_data_to_qdrant(embeddings, chunks, base_filename)
        # Assuming upsert logs its own success/failure internally for now

        end_time = time.time()
        logger.info(f"[{base_filename}] Successfully processed in {end_time - start_time:.2f} seconds.")
        success_list.append(base_filename)

    except Exception as e:
        logger.error(f"[{base_filename}] Unhandled exception during processing: {e}", exc_info=True)
        failure_list.append(base_filename)


def main():
    """
    Main function to orchestrate the PDF processing pipeline.
    """
    logger.info("--- Starting PDF Processing Pipeline ---")

    # --- Pre-flight Checks and Setup ---
    logger.info("Performing pre-flight checks...")

    # 1. Load Embedding Model (do this once before multiprocessing)
    if not embedder.load_embedding_model():
        logger.critical("Failed to load embedding model. Exiting.")
        return
    logger.info("Embedding model loaded successfully.")

    # 2. Ensure Qdrant Client is connectable and Collection exists
    qdrant_client = qdrant_storer.get_qdrant_client()
    if not qdrant_client:
        logger.critical("Failed to connect to Qdrant. Please ensure it's running. Exiting.")
        return
    if not qdrant_storer.create_qdrant_collection():
        logger.critical(f"Failed to create or verify Qdrant collection '{config.QDRANT_COLLECTION_NAME}'. Exiting.")
        return
    logger.info(f"Qdrant connection verified and collection '{config.QDRANT_COLLECTION_NAME}' is ready.")

    # 3. Find PDF Files
    pdf_pattern = os.path.join(config.PDF_INPUT_DIR, "*.pdf")
    pdf_files = glob.glob(pdf_pattern, recursive=False) # Find PDFs only in the top level of the input dir

    if not pdf_files:
        logger.warning(f"No PDF files found matching pattern '{pdf_pattern}'. Exiting.")
        # Create a dummy PDF in pdfs/ for testing if none exist? Optional.
        # pdf_extractor.extract_text_from_pdf("") # Hack to potentially trigger dummy file creation in pdf_extractor
        # pdf_files = glob.glob(pdf_pattern, recursive=False)
        # if not pdf_files:
        #     logger.warning("Still no PDFs found after attempting dummy creation. Exiting.")
        #     return
        return # Exit if no files found

    logger.info(f"Found {len(pdf_files)} PDF files to process in '{config.PDF_INPUT_DIR}'.")

    # --- Parallel Processing ---
    num_workers = min(config.CPU_WORKER_COUNT, len(pdf_files)) # Don't use more workers than files
    logger.info(f"Initializing multiprocessing pool with {num_workers} workers.")

    # Use Manager lists to collect results from worker processes
    with Manager() as manager:
        successful_files = manager.list()
        failed_files = manager.list()

        # Create the pool *after* loading model and checking Qdrant
        with Pool(processes=num_workers) as pool:
            # Create arguments for each task
            tasks = [(pdf_path, successful_files, failed_files) for pdf_path in pdf_files]

            # Use imap_unordered for potentially better performance and progress tracking
            # Wrap with tqdm for overall progress bar
            results = list(tqdm(pool.starmap(process_pdf, tasks), total=len(tasks), desc="Processing PDFs"))

            # Note: process_pdf appends to lists directly, results list here might be all None

        # Convert Manager lists back to regular lists after pool closes
        final_successful = list(successful_files)
        final_failed = list(failed_files)

    # --- Summary ---
    logger.info("--- Pipeline Finished ---")
    logger.info(f"Successfully processed: {len(final_successful)} files.")
    if final_successful:
        logger.debug(f"Successful files: {', '.join(final_successful)}")
    logger.info(f"Failed to process: {len(final_failed)} files.")
    if final_failed:
        logger.warning(f"Failed files: {', '.join(final_failed)}")
    logger.info("-------------------------")


if __name__ == "__main__":
    main()