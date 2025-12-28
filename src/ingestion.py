import logging
import re
from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.rag import engine
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Видаляє шум Wikipedia: [1], [show], [edit] та зайві пробіли."""
    text = text.replace('\x00', '')
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_pdf(file_content: bytes, filename: str):
    try:
        logger.info(f"Starting ingestion for {filename}")
        file_stream = BytesIO(file_content)
        reader = PdfReader(file_stream)
        
        raw_text = "".join(page.extract_text() or "" for page in reader.pages)
        cleaned_text = clean_text(raw_text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        chunks = splitter.split_text(cleaned_text)
        
        if chunks:
            engine.vector_store.add_texts(chunks)
            logger.info(f"Successfully indexed {len(chunks)} chunks from {filename}")
            return len(chunks)
        return 0
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        raise e