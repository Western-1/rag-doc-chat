import logging
import tempfile
import os
import hashlib
from io import BytesIO
from typing import List
import re
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# compatible observe import
try:
    from langfuse import observe
except Exception:
    try:
        from langfuse.decorators import observe
    except Exception:
        def observe(name=None):
            def _decorator(fn):
                return fn
            return _decorator

from src.rag import engine
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    text = text.replace('\x00', '')
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@observe(name="process_pdf_background")
def process_pdf(file_content: bytes, filename: str):
    try:
        logger.info(f"📄 Starting ingestion: {filename}")
        file_stream = BytesIO(file_content)
        reader = PdfReader(file_stream)

        documents: List[Document] = []
        for i, page in enumerate(reader.pages):
            raw_text = page.extract_text()
            if not raw_text:
                continue
            cleaned_text = clean_text(raw_text)
            doc = Document(page_content=cleaned_text, metadata={"source": filename, "page": i + 1})
            documents.append(doc)

        if not documents:
            logger.warning(f"⚠️ No text extracted from {filename}")
            return 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[". ", "? ", "! ", "; ", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        valid_chunks = []
        for chunk in chunks:
            content_hash = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()
            chunk.metadata["doc_hash"] = content_hash
            valid_chunks.append(chunk)

        if valid_chunks:
            engine.vector_store.add_documents(valid_chunks)
            logger.info(f"✅ Indexed {len(valid_chunks)} chunks from {filename}")
            return len(valid_chunks)

        return 0
    except Exception as e:
        logger.exception(f"❌ Error processing {filename}: {e}")
        raise
