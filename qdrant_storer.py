import logging
import time
import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
import config
import embedder # To get embedding dimension

logger = logging.getLogger(__name__)

client = None

def get_qdrant_client() -> QdrantClient | None:
    """
    Initializes and returns a Qdrant client instance based on config.py settings.
    Caches the client instance.
    """
    global client
    if client:
        return client

    logger.info(f"Connecting to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}...")
    try:
        client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            # grpc_port=config.QDRANT_GRPC_PORT, # Uncomment if using gRPC
            # api_key=config.QDRANT_API_KEY,     # Uncomment if using API key
            prefer_grpc=False # Set to True if gRPC is preferred and configured
        )
        # Test connection
        client.list_collections()
        logger.info("Successfully connected to Qdrant.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}", exc_info=True)
        client = None
        return None

def create_qdrant_collection(collection_name: str = config.QDRANT_COLLECTION_NAME):
    """
    Creates the Qdrant collection if it doesn't already exist.
    """
    qdrant_client = get_qdrant_client()
    if not qdrant_client:
        logger.error("Qdrant client not available. Cannot create collection.")
        return False

    vector_dim = embedder.get_embedding_dimension()
    if not vector_dim:
        logger.error("Could not determine embedding dimension. Cannot create collection with correct vector params.")
        return False

    logger.info(f"Checking if collection '{collection_name}' exists...")
    try:
        # Check if collection exists
        try:
             qdrant_client.get_collection(collection_name=collection_name)
             logger.info(f"Collection '{collection_name}' already exists.")
             return True
        except (UnexpectedResponse, ValueError) as e:
             # Handle potential errors if collection doesn't exist (specific error types might vary)
             logger.info(f"Collection '{collection_name}' does not exist or error checking: {e}. Attempting creation.")
             pass # Proceed to create

        logger.info(f"Creating collection '{collection_name}' with vector dimension {vector_dim}...")
        qdrant_client.recreate_collection( # Use recreate_collection for simplicity, create_collection works too
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)
            # Add other configurations like HNSW indexing parameters if needed
            # hnsw_config=models.HnswConfig(m=16, ef_construct=100)
        )
        logger.info(f"Successfully created collection '{collection_name}'.")
        return True
    except Exception as e:
        logger.error(f"Failed to create or verify Qdrant collection '{collection_name}': {e}", exc_info=True)
        return False


def upsert_data_to_qdrant(
    embeddings: list[list[float]],
    texts: list[str],
    source_filename: str,
    collection_name: str = config.QDRANT_COLLECTION_NAME,
    batch_size: int = config.QDRANT_BATCH_SIZE
):
    """
    Upserts embeddings and corresponding text chunks into the specified Qdrant collection.

    Args:
        embeddings: A list of vector embeddings.
        texts: A list of corresponding text chunks.
        source_filename: The name of the PDF file these chunks came from.
        collection_name: The name of the Qdrant collection.
        batch_size: The number of points to upsert in each batch.
    """
    qdrant_client = get_qdrant_client()
    if not qdrant_client:
        logger.error("Qdrant client not available. Cannot upsert data.")
        return

    if len(embeddings) != len(texts):
        logger.error(f"Mismatch between number of embeddings ({len(embeddings)}) and texts ({len(texts)}). Aborting upsert.")
        return

    if not embeddings:
        logger.info("No embeddings provided to upsert.")
        return

    logger.info(f"Upserting {len(embeddings)} points to collection '{collection_name}' for file '{source_filename}'...")
    start_time = time.time()

    points_to_upsert = []
    for i, (vector, text) in enumerate(zip(embeddings, texts)):
        point_id = str(uuid.uuid4()) # Generate a unique ID for each chunk
        payload = {
            "text": text,
            "source": source_filename,
            "chunk_index": i # Store the original index within the document
        }
        points_to_upsert.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

    # Upsert in batches
    total_upserted = 0
    try:
        for i in range(0, len(points_to_upsert), batch_size):
            batch_points = points_to_upsert[i:i + batch_size]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=batch_points,
                wait=False # Set to True if you need confirmation before proceeding
            )
            total_upserted += len(batch_points)
            logger.debug(f"Upserted batch {i//batch_size + 1}, total points: {total_upserted}")

        end_time = time.time()
        logger.info(f"Successfully upserted {total_upserted} points for '{source_filename}' in {end_time - start_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Failed to upsert data to Qdrant collection '{collection_name}': {e}", exc_info=True)


# Example usage
if __name__ == '__main__':
    from utils import setup_logging
    setup_logging()

    # 1. Ensure embedding model is loaded to get dimension
    if not embedder.load_embedding_model():
        logger.error("Failed to load embedding model. Cannot run Qdrant example.")
    else:
        # 2. Ensure collection exists
        if create_qdrant_collection():
            logger.info("Qdrant collection is ready.")

            # 3. Prepare dummy data
            dummy_embeddings = [[0.1 * i] * embedder.get_embedding_dimension() for i in range(5)]
            dummy_texts = [f"This is dummy text chunk {i}." for i in range(5)]
            dummy_source = "example_test.pdf"

            # 4. Upsert data
            upsert_data_to_qdrant(dummy_embeddings, dummy_texts, dummy_source)

            # 5. Verify (optional)
            q_client = get_qdrant_client()
            if q_client:
                try:
                    count_result = q_client.count(collection_name=config.QDRANT_COLLECTION_NAME, exact=True)
                    logger.info(f"Collection '{config.QDRANT_COLLECTION_NAME}' now contains {count_result.count} points.")
                except Exception as e:
                    logger.error(f"Failed to count points in collection: {e}")
        else:
            logger.error("Failed to prepare Qdrant collection. Cannot run upsert example.")