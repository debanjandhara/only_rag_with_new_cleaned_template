# ----------- EXTRAA IMPORTSS -----------------

from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import hashlib
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import hashlib

from typing import List
import os
import asyncio
import pandas as pd
import re
from typing import Any
from fastapi import FastAPI, Body, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse 
from fastapi.middleware.cors import CORSMiddleware
from queue import Queue
from pydantic import BaseModel
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.schema import LLMResult
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import openai
import shutil


import tiktoken

load_dotenv()
# include first_vectorise here()
VECTOR_DB1_PATH="./data/vectorDB"
openai.api_key = os.getenv("OPENAI_API_KEY")


# --------------------


from fastapi import FastAPI
from router import router
import uvicorn

app = FastAPI()

app.include_router(router)

# add here if data / uploaded docs not present , create one

# ------------ STREAMING CODE HERE -----------------------

import os
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Logging Function
from datetime import datetime

def log_error(error_id, function, reason):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("errorfile.txt", "a") as f:
        f.write(f"Timestamp: {timestamp}, Error ID: {error_id}, Function: {function}, Reason: {reason}\n")

# Initialize the model and retriever
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
print("LLM initialized")

vectorstore = FAISS.load_local("./data/vectorDB", embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
print("Documents loaded and indexed")

# Contextualize question prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
print("History aware retriever created")

# Answer question prompt
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
print("RAG chain created")

# Statefully manage chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
print("Conversational RAG chain created")

# Invocation function
import asyncio
from fastapi.responses import StreamingResponse


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, JSONResponse


class QueryRequest(BaseModel):
    query: str
    session_id: str
    history_id: str

# Updated invoke_rag_chain function to support streaming
async def invoke_rag_chain(input_text, history_id=None):
    try:
        print("It Starts Here:")
        if history_id:
            async for response in conversational_rag_chain.astream(
                {"input": input_text, "chat_history": get_session_history},
                config={
                    "configurable": {"session_id": history_id}
                },
            ):
                print(response)
                yield response
        else:
            async for response in conversational_rag_chain.astream(
                {"input": input_text},
                config={
                    "configurable": {"session_id": history_id}
                },
            ):
                print(response)
                yield response
    except Exception as e:
        error_id = "ERR009"
        log_error(error_id, "invoke_rag_chain", str(e))
        print(f"Error {error_id} occurred during RAG chain invocation: {str(e)}")
        # return None

async def handle_request(session_id, query, history_id):
    try:
        print("Handling request for session:", session_id)
        responses = []
        async for response in invoke_rag_chain(query, history_id):
            responses.append(response)
        print("AI response received")
        return responses
    except Exception as e:
        error_id = "ERR001"
        log_error(error_id, "handle_request", str(e))
        print(f"Error {error_id} occurred in handle_request: {str(e)}")
        return None

async def stream_response_chunks(responses):
    try:
        for response in responses:
            if 'answer' in response:
                yield response['answer']
                await asyncio.sleep(0.01)  # Adjust the sleep time as needed
    except Exception as e:
        error_id = "ERR002"
        log_error(error_id, "stream_response_chunks", str(e))
        print(f"Error {error_id} occurred in stream_response_chunks: {str(e)}")
        yield f"Error {error_id}: {str(e)}"

@app.post("/stream")
async def stream_response(request_body: QueryRequest):
    try:
        print(f"Received query: {request_body.query} with session_id: {request_body.session_id} and history_id: {request_body.history_id}")

        if not request_body.query or not request_body.session_id or not request_body.history_id:
            error_message = "All 'query', 'session_id', and 'history_id' must be provided"
            print(error_message)
            return JSONResponse(content={"error": error_message}, status_code=400)

        responses = await handle_request(request_body.session_id, request_body.query, request_body.history_id)
        if responses is None:
            error_message = "Error processing the request"
            print(error_message)
            return JSONResponse(content={"error": error_message}, status_code=500)

        return StreamingResponse(stream_response_chunks(responses), media_type="text/plain")
    except HTTPException as http_err:
        error_id = "ERR010"
        error_message = f"HTTPException {error_id}: {str(http_err)}"
        print(error_message)
        return JSONResponse(content={"error": error_message}, status_code=http_err.status_code)
    except Exception as e:
        error_id = "ERR010"
        error_message = f"Error {error_id}: {str(e)}"
        print(error_message)
        return JSONResponse(content={"error": error_message}, status_code=500)

# --------------------------------------------------------

if __name__ == "__main__":
    # create_data_directory()
    uvicorn.run(app, host="0.0.0.0", port=8000)