import os
import openai
import pandas as pd

from IPython.display import display

from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader

openai.api_key = os.getenv("OPENAI_API_KEY")

DATA_FOLDER_DIRECTORY = "./data"
UPLOADED_DATA_FOLDER_DIRECTORY = "./data/uploaded_docs"
FORMATTED_UPLOAD_DIRECTORY_FOR_DELETION="data\\uploaded_docs"
VECTOR_DB_PATH = "./data/vectorDB"
VECTOR_DB_INIT_FILE = "./data/initialize.txt"

# Dictionary to map file extensions to their respective loaders
loaders = {
    '.pdf': PyPDFLoader,
    '.txt': TextLoader,
    '.docx': Docx2txtLoader,
    '.csv': CSVLoader,
    '.xlsx': UnstructuredExcelLoader,
}

def first_vectorise():
  
  if os.path.isfile(f"{VECTOR_DB_PATH}/index.faiss")!=True:
    
    if not os.path.exists(DATA_FOLDER_DIRECTORY):
      os.makedirs(DATA_FOLDER_DIRECTORY)

    if not os.path.exists(VECTOR_DB_INIT_FILE):
      os.makedirs(os.path.dirname(VECTOR_DB_INIT_FILE), exist_ok=True)
      with open(VECTOR_DB_INIT_FILE, 'w') as file:
          file.write("Hi, I am a chatbot.")

    if not os.path.exists(VECTOR_DB_PATH):
      os.makedirs(VECTOR_DB_PATH)

    Merged_Vector_Path = VECTOR_DB_PATH

    loader = DirectoryLoader(DATA_FOLDER_DIRECTORY, glob=f"./initialize.txt", loader_cls=TextLoader, loader_kwargs=dict(encoding="utf-8"))
    loaded_txt = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    docs = text_splitter.split_documents(loaded_txt)
    embeddings = OpenAIEmbeddings()
    docs_db = FAISS.from_documents(docs, embeddings)
    docs_db.save_local(Merged_Vector_Path)
    save_VectorDB_to_CSV()

def add_to_vector(filename):

  Merged_Vector_Path = VECTOR_DB_PATH
    
  file_extension = os.path.splitext(filename)[1].lower()

  # File Extension to Loader Selection
  if file_extension in loaders:
      loader_cls = loaders[file_extension]
      loader_kwargs = {}

  # Loader-specific arguments - Fixes any special characters
  if file_extension == '.txt':
      loader_kwargs['encoding'] = 'utf-8'
      
  # Initialize the loader
  loader = DirectoryLoader(UPLOADED_DATA_FOLDER_DIRECTORY, glob=f"./{filename}", loader_cls=loader_cls, loader_kwargs=loader_kwargs)
  loaded_txt = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
  docs = text_splitter.split_documents(loaded_txt)
  embeddings = OpenAIEmbeddings()
  VectorStore = FAISS.from_documents(docs, embeddings)

  Merged_VectorStore = FAISS.load_local(Merged_Vector_Path, embeddings, allow_dangerous_deserialization=True)
  Merged_VectorStore.merge_from(VectorStore)
  Merged_VectorStore.save_local(Merged_Vector_Path)


#--------- Save to CSV, Delete and Refresh Function Dependencies -----------

# Display Documents in VectorStore
def show_vstore(store):
 vector_df = store_to_df(store)
 vector_df.to_csv("output.csv", index=False)
#  display(vector_df)   # For Displaying in Console 

# Convert VectorStore into df to convenient access
def store_to_df(store):
 v_dict = store.docstore._dict
 data_rows = []
 for k in v_dict.keys():
   doc_name = v_dict[k].metadata['source'].split('/')[-1]
   content = v_dict[k].page_content
   data_rows.append({"chunk_id":k, "document":doc_name, "content":content})
 vector_df = pd.DataFrame(data_rows)
 return vector_df

# Deleting a document from a vectorstore
def delete_document(store, document):
  vector_df = store_to_df(store)
  chunks_list = vector_df.loc[vector_df['document']==document]['chunk_id'].tolist()  
  if len(chunks_list) == 0:
    return f"{document} does not exists"
  store.delete(chunks_list)

#-----------------------------------------

def save_VectorDB_to_CSV():
  Merged_Vector_Path = VECTOR_DB_PATH
  
  embeddings = OpenAIEmbeddings()
  
  Merged_VectorStore = FAISS.load_local(Merged_Vector_Path, embeddings, allow_dangerous_deserialization=True)
  
  show_vstore(Merged_VectorStore)

def delete_vector(filename):
    
    Merged_Vector_Path = VECTOR_DB_PATH
    embeddings = OpenAIEmbeddings()
    Merged_VectorStore = FAISS.load_local(Merged_Vector_Path, embeddings, allow_dangerous_deserialization=True)
    
    delete_document(Merged_VectorStore, f"{FORMATTED_UPLOAD_DIRECTORY_FOR_DELETION}\\{filename}")
    delete_document(Merged_VectorStore, f"{filename}")
    
    Merged_VectorStore.save_local(Merged_Vector_Path)
    
    save_VectorDB_to_CSV()
    

def refresh_vector(filename):
    
  Merged_Vector_Path = VECTOR_DB_PATH
  embeddings = OpenAIEmbeddings()
  Merged_VectorStore = FAISS.load_local(Merged_Vector_Path, embeddings, allow_dangerous_deserialization=True)
  
  delete_document(Merged_VectorStore, f"{FORMATTED_UPLOAD_DIRECTORY_FOR_DELETION}\\{filename}")
  delete_document(Merged_VectorStore, f"{filename}")
  
  print("Refreshing Vector : ",f"{FORMATTED_UPLOAD_DIRECTORY_FOR_DELETION}\\{filename}")
  
  Merged_VectorStore.save_local(Merged_Vector_Path)

  add_to_vector(filename)
  
  save_VectorDB_to_CSV()
