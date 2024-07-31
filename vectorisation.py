# // Initialization : adding / modifying / deleting is easy, when there is a sample vectorDB initializexd.. using a initialize txt to build a vectorDB

# if initialize.txt not present create one

import os
import pandas as pd
import openai

from IPython.display import display

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
from langchain.chains import VectorDBQA
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain

from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


DATA_FOLDER_DIRECTORY = "./data/uploaded_docs"

VECTOR_DB_PATH = ".\\data\\vectorDB"

# Dictionary to map file extensions to their respective loaders
loaders = {
    '.pdf': PyPDFLoader,
    '.txt': TextLoader,
    '.docx': Docx2txtLoader,
    '.csv': CSVLoader,
    '.xlsx': UnstructuredExcelLoader,
}

def first_vectorise():
  
  if os.path.isfile("./data/vectorDB/index.faiss")!=True:

    Merged_Vector_Path = VECTOR_DB_PATH

    loader = DirectoryLoader(f'./data', glob=f"./initialize.txt", loader_cls=TextLoader, loader_kwargs=dict(encoding="utf-8"))
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

def add_to_vector(filename):

  Merged_Vector_Path = VECTOR_DB_PATH
  
    
  file_extension = os.path.splitext(filename)[1].lower()  # Get the file extension

  # Check if the file extension is in the loaders dictionary
  if file_extension in loaders:
      loader_cls = loaders[file_extension]
      loader_kwargs = {}

  # Set loader-specific arguments if needed
  if file_extension == '.txt':
      loader_kwargs['encoding'] = 'utf-8'
      
  # Initialize the loader
  loader = DirectoryLoader(DATA_FOLDER_DIRECTORY, glob=f"./{filename}", loader_cls=loader_cls, loader_kwargs=loader_kwargs)
  loaded_txt = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
  docs = text_splitter.split_documents(loaded_txt)
  print(f"docs : {docs}")
  embeddings = OpenAIEmbeddings()
  VectorStore = FAISS.from_documents(docs, embeddings)

  Merged_VectorStore = FAISS.load_local(Merged_Vector_Path, embeddings, allow_dangerous_deserialization=True)
  print(f"\n\nVectorstore : {VectorStore}\n\n")
  print(f"Merged_VectorStore : {Merged_VectorStore}\n\n")
  Merged_VectorStore.merge_from(VectorStore)
  Merged_VectorStore.save_local(Merged_Vector_Path)
  print("Added to Vector : ",f"{DATA_FOLDER_DIRECTORY}/{filename}")



#--------- refresh_vector() Dependencies -----------

# Display Documents in VectorStore
def show_vstore(store):
 vector_df = store_to_df(store)
 vector_df.to_csv("output.csv", index=False)
 display(vector_df)

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

def delete_vector(filename):
    
    Merged_Vector_Path = VECTOR_DB_PATH
    embeddings = OpenAIEmbeddings()
    Merged_VectorStore = FAISS.load_local(Merged_Vector_Path, embeddings, allow_dangerous_deserialization=True)
    
    delete_document(Merged_VectorStore, f"{DATA_FOLDER_DIRECTORY}\\{filename}")
    delete_document(Merged_VectorStore, f"{filename}")
    
    Merged_VectorStore.save_local(Merged_Vector_Path)
    print_VectorDB()
    

def refresh_vector(filename):
    
  Merged_Vector_Path = VECTOR_DB_PATH
  embeddings = OpenAIEmbeddings()
  Merged_VectorStore = FAISS.load_local(Merged_Vector_Path, embeddings, allow_dangerous_deserialization=True)
  
  delete_document(Merged_VectorStore, f"{DATA_FOLDER_DIRECTORY}\\{filename}")
  delete_document(Merged_VectorStore, f"{filename}")
  
  print("Refreshing Vector : ",f"{DATA_FOLDER_DIRECTORY}\\{filename}")
  
  Merged_VectorStore.save_local(Merged_Vector_Path)

  add_to_vector(filename)
  
  print_VectorDB()

def print_VectorDB():
  Merged_Vector_Path = VECTOR_DB_PATH
  
  embeddings = OpenAIEmbeddings()
  
  Merged_VectorStore = FAISS.load_local(Merged_Vector_Path, embeddings, allow_dangerous_deserialization=True)
  
  show_vstore(Merged_VectorStore)
  

first_vectorise()
# add_to_vector_derived()
# add_to_vector('Debanjan - CBTS Cover Letter.pdf')
# add_to_vector('Debanjan - CBTS Cover Letter.pdf')

