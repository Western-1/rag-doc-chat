import logging
import re
import hashlib
from io import BytesIO
from typing import List, Dict

from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langfuse.decorators import observe

from src.rag import engine
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Cleans extracted text: removes null bytes, fixes hyphenation, 
    removes citations, and normalizes whitespace.
    """
    if not text:
        return ""
        
    # 1. Fix broken words from line breaks (e.g., "process- \n ing" -> "processing")
    text = text.replace("-\n", "")
    
    # 2. Fix broken sentences (replace regular newlines with space)
    text = text.replace("\n", " ")
    
    # 3. Remove null bytes
    text = text.replace('\x00', '')
    
    # 4. Remove citation artifacts like [1], [12]
    text = re.sub(r'\[\d+\]', '', text)
    
    # 5. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@observe(name="pdf-ingestion")
def process_pdf(file_content: bytes, filename: str):
    """
    Ingests PDF content, preserves page metadata, handles deduplication,
    and indexes into Qdrant.
    """
    try:
        logger.info(f"Starting ingestion for {filename}")
        file_stream = BytesIO(file_content)
        reader = PdfReader(file_stream)
        
        documents: List[Document] = []

        # --- STEP 1: Extract Text PER PAGE ---
        for i, page in enumerate(reader.pages):
            raw_text = page.extract_text()
            if not raw_text:
                continue
                
            cleaned_text = clean_text(raw_text)
            
            # Create a Document object immediately to attach metadata
            doc = Document(
                page_content=cleaned_text,
                metadata={
                    "source": filename,
                    "page": i + 1  # Human-readable page number (starts at 1)
                }
            )
            documents.append(doc)

        if not documents:
            logger.warning(f"No text extracted from {filename}")
            return 0

        # --- STEP 2: Intelligent Splitting ---
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[". ", "? ", "! ", "; ", " ", ""] # Try to split by sentences first
        )
        
        # split_documents preserves the metadata from step 1!
        chunks = splitter.split_documents(documents)
        
        # --- STEP 3: Deduplication Hashing ---
        valid_chunks = []
        for chunk in chunks:
            # Create a deterministic hash of the content
            content_hash = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()
            chunk.metadata["doc_hash"] = content_hash
            valid_chunks.append(chunk)

        # --- STEP 4: Indexing ---
        if valid_chunks:
            # Use add_documents instead of add_texts to keep metadata
            engine.vector_store.add_documents(valid_chunks)
            logger.info(f"Successfully indexed {len(valid_chunks)} chunks from {filename}")
            return len(valid_chunks)
        
        return 0

    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        # In background tasks, raising error might crash the worker, 
        # but for observability, it's good to bubble it up or log strictly.
        raise e