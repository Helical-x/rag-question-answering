import os
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from haystack import Pipeline
from src.indexing_pipeline import indexing_pipeline
from src.rag_pipeline import rag_pipeline

load_dotenv()

app = FastAPI(title="My Haystack RAG API")

# Get the absolute path to the directory containing this script
# SCRIPT_DIR = Path(__file__).parent.absolute()
# indexing_pipeline_file = os.path.join(SCRIPT_DIR, "pipelines", "indexing_pipeline.yaml")
# rag_pipeline_file = os.path.join(SCRIPT_DIR, "pipelines", "rag_pipeline.yaml")
# # Load the pipelines from the YAML files
# with open(indexing_pipeline_file, "rb") as f:
#     indexing_pipeline = Pipeline.load(f)
# with open(rag_pipeline_file, "rb") as f:
#     rag_pipeline = Pipeline.load(f)

# Create the file upload directory if it doesn't exist
FILE_UPLOAD_PATH = os.getenv("FILE_UPLOAD_PATH", str((Path(__file__).parent.parent / "file-upload").absolute()))
Path(FILE_UPLOAD_PATH).mkdir(parents=True, exist_ok=True)


@app.get("/ready")
def check_status():
    """
    This endpoint can be used during startup to understand if the
    server is ready to take any requests, or is still loading.

    The recommended approach is to call this endpoint with a short timeout,
    like 500ms, and in case of no reply, consider the server busy.
    """
    return True


@app.post("/file-upload")
def upload_files(files: List[UploadFile] = File(...), keep_files: Optional[bool] = False):
    """
    Upload a list of files to be indexed.

    Note: files are removed immediately after being indexed. If you want to keep them, pass the
    `keep_files=true` parameter in the request payload.
    """

    file_paths: list = []

    for file_to_upload in files:
        try:
            file_path = Path(FILE_UPLOAD_PATH) / f"{uuid.uuid4().hex}_{file_to_upload.filename}"
            with file_path.open("wb") as fo:
                fo.write(file_to_upload.file.read())
            file_paths.append(file_path)
        finally:
            file_to_upload.file.close()

    result = indexing_pipeline.run({"converter": {"sources": file_paths}})

    # Clean up indexed files
    if not keep_files:
        for p in file_paths:
            p.unlink()

    return result


@app.get("/query")
def ask_rag_pipeline(query: str):
    """
    Ask a question to the RAG pipeline.
    """
    result = rag_pipeline.run({"text_embedder": {"text": query}, "prompt_builder": {"question": query}})

    return result
