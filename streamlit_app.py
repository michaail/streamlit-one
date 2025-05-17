import streamlit as st
from openai import OpenAI
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import pdfminer
from pdfminer.high_level import extract_text
import os
import shutil
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


key = st.secrets["API_KEY"]
base_url = st.secrets["BASE_URL"]
model = st.secrets["MODEL"]

client = OpenAI(
  base_url=base_url,
  api_key=key
)

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# Specify embedding model
embed_model_id = 'intfloat/e5-small-v2'
model_kwargs = {"device": "cpu", "trust_remote_code": True}

embeddings_model = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)


# Path to the directory to save Chroma database
CHROMA_PATH = "chroma"
def save_to_chroma(chunks):
  """
  Save the given list of Document objects to a Chroma database.
  Args:
  chunks (list[Document]): List of Document objects representing text chunks to save.
  Returns:
  None
  """

  # Clear out the existing database directory if it exists
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

  # Create a new Chroma database from the documents using OpenAI embeddings
  db = Chroma.from_documents(
    chunks,
    embeddings_model,
    persist_directory=CHROMA_PATH
  )

  # Persist the database to disk
  db.persist()
  print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

with st.sidebar:
  uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

if uploaded_file is not None:
  text = extract_text(uploaded_file)
  doc = Document(page_content=text, metadata={"source": uploaded_file.name})
  # data = extract_data(uploaded_file)
  # print(data)

  chunks = splitter.split_documents([doc])
  st.write("## Chunks")
  st.write(chunks[0])

  save_to_chroma(chunks)

  st.write("## Index")
  # st.write(index)

