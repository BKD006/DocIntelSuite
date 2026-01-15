from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentRAGException
from utils.file_io import generate_session_id, save_uploaded_files
from utils.document_ops import load_documents
from src.document_ingestion.vectorstore import FaissManager


class ChatIngestor:
    def __init__(self,
                 temp_base: str= "data",
                 faiss_base: str= "faiss_index",
                 use_session_dirs: bool= True,
                 session_id: Optional[str] =None):
        try:
            self.model_loader=ModelLoader()

            self.use_session= use_session_dirs
            self.session_id= session_id or generate_session_id()

            self.temp_base= Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base= Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)

            self.temp_dir= self._resolve_dir(self.temp_base)
            self.faiss_dir=self._resolve_dir(self.faiss_base)

            log.info("ChatIngestor initialized",
                     session_id=self.session_id,
                     temp_dir=str(self.temp_dir),
                     faiss_dir=str(self.faiss_dir),
                     sessionized=self.use_session)
            
        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentRAGException("Initialization error in ChatIngestor", e) from e
            
    def _resolve_dir(self, base: Path):
        if self.use_session:
            d= base / self.session_id # faiss_index/zyd123
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base # fallback: "faiss_index/"
    
    def _split(self, docs: List[Document], chunk_size= 1000, chunk_overlap=200)->List[Document]:
        splitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks= splitter.split_documents(docs)
        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return chunks

    def built_retriever(self,
                        uploaded_files: Iterable,
                        *,
                        chunk_size: int= 1000,
                        chunk_overlap: int= 200,
                        k: int = 5):
        try:
            paths= save_uploaded_files(uploaded_files, self.temp_dir)
            docs=load_documents(paths)
            if not docs:
                raise ValueError("No valid documnets loaded")
            chunks= self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            fm=FaissManager(self.faiss_dir, self.model_loader)
            texts= [c.page_content for c in chunks]
            metas= [c.metadata for c in chunks]

            try:
                vector_store = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vector_store = fm.load_or_create(texts=texts, metadatas=metas)

            added=fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index= str(self.faiss_dir))

            return vector_store.as_retriever(search_type="similarity", search_kwargs={"k":k})
        
        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentRAGException("Failed to build retriever", e) from e