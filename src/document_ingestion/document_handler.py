from __future__ import annotations
import os
from typing import Optional
import fitz # PyMuPDF
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentRAGException
from utils.file_io import generate_session_id

class DocHandler:
    """
    PDF save + read (page-wise) for analysis.
    """
    def __init__(self, data_dir: Optional[str]=None, session_id: Optional[str]=None):
        self.data_dir= data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis"))
        self.session_id= session_id or generate_session_id("session")
        self.session_path= os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        log.info("Dohandler Initialized", session_id=self.session_id, session_path=self.session_path)

    def save_pdf(self, uploaded_file) -> str:
        try:
            filename=os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Invalid file type. Only PDFs are allowed")
            save_path=os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            log.info("PDF saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save PDF", error= str(e), session_id=self.session_id)
            raise DocumentRAGException(f"Failed to save PDF: {str(e)}", e) from e

    def read_pdf(self, pdf_path: str)->str:
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page.get_text()}")  # type: ignore
            text = "\n".join(text_chunks)
            log.info("PDF read successfully", pdf_path=pdf_path, session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            log.error("Failed to read PDF", error=str(e), pdf_path=pdf_path, session_id=self.session_id)
            raise DocumentRAGException(f"Could not process PDF: {pdf_path}", e) from e
