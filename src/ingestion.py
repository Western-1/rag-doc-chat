import logging
from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag import vector_store
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

def process_pdf(file_content: bytes, filename: str):
    """
    Background task for ETL: Extract -> Transform (Split) -> Load (Embed & Store).
    """
    try:
        logger.info(f"Starting ingestion for {filename}")
        file_stream = BytesIO(file_content)
        reader = PdfReader(file_stream)
        text = "".join(page.extract_text() or "" for page in reader.pages)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        chunks = splitter.split_text(text)
        
        if chunks:
            vector_store.add_texts(chunks)
            logger.info(f"Successfully indexed {len(chunks)} chunks from {filename}")
            return len(chunks)
        
        logger.warning(f"No text found in {filename}")
        return 0

    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        raise e