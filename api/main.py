import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, List, Any, Optional
from utils.document_ops import FastAPIFileAdapter, read_pdf_via_handler

from src.document_ingestion.rag_ingestor import RAGIngestor
from src.document_ingestion.document_handler import DocHandler
from src.document_ingestion.document_comparator import DocumentComparator
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.doc_compare import DocumentCompareLLM
from src.document_chat.retrieval import ConversationalRAG
from logger import GLOBAL_LOGGER as log

FAISS_BASE= os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE= os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME= os.getenv("FAISS_INDEX_NAME", "index") # <----keep consistent with save local()

app=FastAPI(title="Document Portal API", version="0.1")

BASE_DIR= Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates= Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    log.info("Serving UI homepage")
    res=templates.TemplateResponse("index.html", {"request": request})
    res.headers["Cache-Control"]="no-store"
    return res

@app.get("/health")
def health() -> Dict[str, str]:
    log.info("Health check passed.")
    return {"status": "ok", "service":"document-portal"}

# ---------ANALYZE--------    
@app.post("/analyze")
async def analyze_document(file: UploadFile=File(...)) -> Any:
    try:
        log.info(f"Recieved file for analysis: {file.filename}")
        dh= DocHandler()
        saved_path= dh.save_pdf(FastAPIFileAdapter(file))
        text=read_pdf_via_handler(dh, saved_path)

        analyzer= DocumentAnalyzer()
        result=analyzer.analyze_document(text)
        log.info("Document analysis completed.")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis Failed: {e}")
    
# -------COMPARE-------
@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...))-> Any:
    try:
        log.info(f"Comparing files: {reference.filename} vs {actual.filename}")
        dc= DocumentComparator()
        ref_path, act_path= dc.save_uploaded_files(FastAPIFileAdapter(reference), FastAPIFileAdapter(actual))
        _ = ref_path, act_path
        combined_text= dc.combine_documents()
        comp= DocumentCompareLLM()
        df= comp.compare_documents(combined_text)
        log.info("Document Comparison completed.")
        return {"rows": df.to_dict(orient="records"), "session_id":dc.session_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")
    
# -------CHAT: INDEX-------
@app.post("/chat/index")
async def build_chat_index(
    files: List[UploadFile]= File(...),
    session_id: Optional[str]= Form(None),
    use_session_dirs: bool= Form(True),
    chunk_size: int= Form(1000),
    chunk_overlap: int= Form(200),
    k: int= Form(5)
)-> Any:
    try:
        log.info(f"Indexing chat session. Session ID: {session_id}, Files: {[f.filename for f in files]}")
        wrapped= [FastAPIFileAdapter(f) for f in files]
        ci= RAGIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs= use_session_dirs,
            session_id=session_id or None
        )
        # NOTE: ensure your RAGIngestor saves with index_name="index" or FAISS_INDEX_NAME
        ci.built_retriever(wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k)
        log.info(f"Index created successfully for session: {ci.session_id}")
        return {"session_id": ci.session_id, "k":k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index Creation failed: {e}")

# ------CHAT: QUERY------
@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5)
    )-> Any:
    try:
        log.info(f"Received chat query: '{question}' | session: {session_id}")
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=500, detail="session_id is required when use use_session_dirs=True")
        
        index_dir=os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE #type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")
        
        rag= ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME) # build retriever + chain
        response = rag.invoke(question, chat_history=[])
        log.info("Chat query handled successfully.")

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LECL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")