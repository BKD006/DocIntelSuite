from __future__ import annotations
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentRAGException

# FAISS Manager (load-or-create)
class VectorIndexManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader]= None):
        self.index_dir= Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path= self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}

        if self.meta_path.exists():
            try:
                self._meta= json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                self._meta= {"rows": {}}
        
        self.model_loader= model_loader or ModelLoader()
        self.embedding= self.model_loader.load_embeddings()
        self.vector_store: Optional[FAISS]= None
    
    def _exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    @staticmethod
    def _fingerprint(text: str, mdata: Dict[str, Any]) -> str:
        src= mdata.get("source") or mdata.get("file_path")
        rowid= mdata.get("row_id")
        if src is not None:
            return f"{src}::{''if rowid is None else rowid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_metadata(self):
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_documents(self, docs: List[Document]):
        if self.vector_store is None:
            raise RuntimeError("Call load_or_create() befor add_documents().")
        new_docs: List[Document]=[]

        for d in docs:
            key= self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key]= True
            new_docs.append(d)
        
        if new_docs:
            self.vector_store.add_documents(new_docs)
            self.vector_store.save_local(str(self.index_dir))
            self._save_metadata()
        return len(new_docs)

    def load_or_create(self, texts:Optional[List[str]]=None, metadatas: Optional[List[dict]]=None):
        ## if we running first time then it will not go in this block
        if self._exists():
            self.vector_store= FAISS.load_local(
                str(self.index_dir),
                embeddings=self.embedding,
                allow_dangerous_deserialization=True
            )
            return self.vector_store
        
        if not texts:
            raise DocumentRAGException("No FAISS index and no data to create one", sys)
        self.vector_store= FAISS.from_texts(texts=texts, embedding=self.embedding, metadatas=metadatas or [])
        self.vector_store.save_local(str(self.index_dir))
        return self.vector_store