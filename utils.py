import logging
import sys
import config

def setup_logging():
    """
    Configures the root logger based on settings in config.py.
    """
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout) # Log to standard output
            # Optionally add FileHandler here if needed:
            # logging.FileHandler("pipeline.log")
        ]
    )
    # Set higher level for noisy libraries if necessary
    # logging.getLogger("PIL").setLevel(logging.WARNING)
    # logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {config.LOG_LEVEL}")

# Example usage (will run if you execute utils.py directly)
if __name__ == '__main__':
    setup_logging()
    logging.info("Logging setup complete.")
    logging.debug("This is a debug message.") # Won't show if LOG_LEVEL is INFO
    logging.warning("This is a warning.")