import fitz  # PyMuPDF
import logging
import os
import config

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str | None:
    """
    Extracts text content from all pages of a given PDF file.

    Args:
        pdf_path: The full path to the PDF file.

    Returns:
        A string containing the concatenated text from all pages,
        or None if the file doesn't exist or an error occurs during extraction.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return None
    if not pdf_path.lower().endswith(".pdf"):
        logger.warning(f"File is not a PDF, skipping: {pdf_path}")
        return None

    full_text = ""
    try:
        logger.debug(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        logger.info(f"Extracting text from {doc.page_count} pages in '{os.path.basename(pdf_path)}'...")

        for page_num, page in enumerate(doc):
            try:
                page_text = page.get_text("text", sort=True) # Use sort=True for better reading order
                if page_text:
                    full_text += page_text + "\n\n" # Add double newline as page separator
            except Exception as page_err:
                logger.warning(f"Could not extract text from page {page_num + 1} in '{os.path.basename(pdf_path)}': {page_err}")
                continue # Skip problematic pages

        doc.close()
        logger.info(f"Successfully extracted text from '{os.path.basename(pdf_path)}'. Total length: {len(full_text)} chars.")
        return full_text.strip() if full_text else None

    except fitz.errors.FitzError as fitz_err:
        logger.error(f"PyMuPDF error processing '{os.path.basename(pdf_path)}': {fitz_err}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error extracting text from '{os.path.basename(pdf_path)}': {e}", exc_info=True)
        return None

# Example usage (will run if you execute pdf_extractor.py directly)
if __name__ == '__main__':
    from utils import setup_logging
    setup_logging()

    # Create a dummy PDF in the input directory for testing
    dummy_pdf_dir = config.PDF_INPUT_DIR
    os.makedirs(dummy_pdf_dir, exist_ok=True)
    dummy_pdf_path = os.path.join(dummy_pdf_dir, "test_document.pdf")

    # Simple text content for the dummy PDF
    try:
        doc = fitz.open() # New empty PDF
        page = doc.new_page()
        page.insert_text((72, 72), "This is page 1 of the test PDF.")
        page = doc.new_page()
        page.insert_text((72, 72), "This is page 2. It has some text.")
        doc.save(dummy_pdf_path)
        doc.close()
        logger.info(f"Created dummy PDF for testing: {dummy_pdf_path}")
    except Exception as create_err:
         logger.error(f"Failed to create dummy PDF: {create_err}")


    if os.path.exists(dummy_pdf_path):
        extracted_content = extract_text_from_pdf(dummy_pdf_path)
        if extracted_content:
            logger.info("--- Extracted Content ---")
            print(extracted_content)
            logger.info("--- End Extracted Content ---")
        else:
            logger.error("Failed to extract text from the dummy PDF.")
        # Clean up the dummy file
        # os.remove(dummy_pdf_path)
        # logger.info(f"Removed dummy PDF: {dummy_pdf_path}")
    else:
        logger.warning("Dummy PDF was not created, skipping extraction test.")

    # Test non-existent file
    extract_text_from_pdf("non_existent_file.pdf")
    # Test non-pdf file
    non_pdf_path = "requirements.txt"
    extract_text_from_pdf(non_pdf_path)