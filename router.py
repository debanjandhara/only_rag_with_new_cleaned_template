from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import os
import shutil
import json

from vectorisation import *

router = APIRouter()
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "data/uploaded_docs"
FILE_DATA_PATH = "data/file_data.json"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(FILE_DATA_PATH):
    with open(FILE_DATA_PATH, 'w') as f:
        json.dump({}, f)

# Sample dictionary to track processed files
processed_files = {}

# Utility functions to read/write the processed_files dictionary
def load_processed_files():
    global processed_files
    with open(FILE_DATA_PATH, 'r') as f:
        processed_files = json.load(f)

def save_processed_files():
    with open(FILE_DATA_PATH, 'w') as f:
        json.dump(processed_files, f, indent=4)

# Load processed files on startup
load_processed_files()

@router.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    files = os.listdir(UPLOAD_FOLDER)
    files_status = {file: processed_files.get(file, "unprocessed") for file in files}
    return templates.TemplateResponse("upload.html", {"request": request, "files": files_status})

@router.get("/chatbot", response_class=HTMLResponse)
async def get_chatbot(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

@router.post("/upload", response_class=JSONResponse)
async def upload_files(files: list[UploadFile] = File(...)):
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        processed_files[file.filename] = "unprocessed"
    save_processed_files()
    return {"status": "success", "message": "Files uploaded successfully"}

@router.post("/process", response_class=JSONResponse)
async def process_file(filename: str = Form(...)):
    if filename in processed_files:
        processed_files[filename] = "processed"
        refresh_vector(filename)
        save_processed_files()
        return {"status": "success", "message": f"{filename} processed successfully"}
    return {"status": "error", "message": f"{filename} not found"}

@router.post("/delete", response_class=JSONResponse)
async def delete_file(filename: str = Form(...)):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        processed_files.pop(filename, None)
        delete_vector(filename)
        save_processed_files()
        return {"status": "success", "message": f"{filename} deleted successfully"}
    return {"status": "error", "message": f"{filename} not found"}
