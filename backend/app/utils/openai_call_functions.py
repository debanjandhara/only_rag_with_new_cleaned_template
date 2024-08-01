import os
import openai

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory

from app.utils.logger import log_error
from app.utils.vectorisation import first_vectorise

VECTOR_DB_PATH = "./data/vectorDB"

# OpenAI Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
print("LLM initialized")

# Initialize Retriever
vectorstore = None
retriever = None

# Initialize VectorDB 
first_vectorise()

def load_vectorstore_and_retriever():
    global vectorstore, retriever
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    print("Documents loaded and indexed")

load_vectorstore_and_retriever()

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

# Invoke_rag_chain function : Main Function - Supports Streaming
async def invoke_rag_chain(input_text, session_id=None):
    try:
        if session_id:
            async for response in conversational_rag_chain.astream(
                {"input": input_text, "chat_history": get_session_history},
                config={
                    "configurable": {"session_id": session_id}
                },
            ):
                yield response
        else:
            async for response in conversational_rag_chain.astream(
                {"input": input_text},
                config={
                    "configurable": {"session_id": session_id}
                },
            ):
                yield response
    except Exception as e:
        error_id = "ERR009"
        log_error(error_id, "invoke_rag_chain", str(e))
        print(f"Error {error_id} occurred during RAG chain invocation: {str(e)}")