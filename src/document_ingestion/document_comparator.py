from __future__ import annotations
import shutil
from pathlib import Path
from typing import Optional
import fitz # PyMuPDF
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentRAGException
from utils.file_io import generate_session_id


class DocumentComparator:
    """
    Save, read and combine PDFs for comparision with session-based versioning.
    """
    def __init__(self, base_dir: str= "data/document_compare", session_id: Optional[str]=None):
        self.base_dir= Path(base_dir)
        self.session_id= session_id or generate_session_id()
        self.session_path= self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        log.info("DocumentComparator initialized", session_path=str(self.session_path))        
    
    def save_uploaded_files(self, reference_file, actual_file):
        try:
            ref_path = self.session_path / reference_file.name
            act_path = self.session_path / actual_file.name
            for fobj, out in ((reference_file, ref_path), (actual_file, act_path)):
                if not fobj.name.lower().endswith(".pdf"):
                    raise ValueError("Only PDF files are allowed.")
                with open(out, "wb") as f:
                    if hasattr(fobj, "read"):
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())
            log.info("Files saved", reference=str(ref_path), actual=str(act_path), session=self.session_id)
            return ref_path, act_path
        except Exception as e:
            log.error("Error saving PDF files", error=str(e), session=self.session_id)
            raise DocumentRAGException("Error saving files", e) from e
    def read_pdf(self, pdf_path: Path):
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")
                parts=[]
                for page_num in range(doc.page_count):
                    page= doc.load_page(page_num)
                    text=page.get_text()
                    if text.strip():
                        parts.append(f"\n --- Page {page_num+1} --- \n{text}")
            log.info("PDF read successfully", file=str(pdf_path), pages=len(parts))
            return "\n".join(parts)
        except Exception as e:
            log.error("Error reading PDF", file=str(pdf_path), error=str(e))
            raise DocumentRAGException("Error reading PDF", e) from e

    def combine_documents(self)->str:
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() == ".pdf":
                    content = self.read_pdf(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")
            combined_text = "\n\n".join(doc_parts)
            log.info("Documents combined", count=len(doc_parts), session=self.session_id)
            return combined_text
        except Exception as e:
            log.error("Error combining documents", error=str(e), session=self.session_id)
            raise DocumentRAGException("Error combining documents", e) from e

    def clean_old_sessions(self, keep_latest: int=3):
        try:
            sessions = sorted([f for f in self.base_dir.iterdir() if f.is_dir()], reverse=True)
            for folder in sessions[keep_latest:]:
                shutil.rmtree(folder, ignore_errors=True)
                log.info("Old session folder deleted", path=str(folder))
        except Exception as e:
            log.error("Error cleaning old sessions", error=str(e))
            raise DocumentRAGException("Error cleaning old sessions", e) from e


