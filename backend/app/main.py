from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import docs_upload, streaming_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(docs_upload.router)
app.include_router(streaming_response.router)

@app.get("/health")
def home():
    return {"message": "Welcome to the Document Chatbot 😇"}
