import streamlit as st
from openai import OpenAI
import base64
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.document_stores.local_file import LocalFileStore
# from langchain.document_stores.faiss import FAISS
# from langchain.retrievers import CacheBackedEmbeddings
# from langchain.chains.rephrase_history_chain import RephraseHistoryChain
from langchain_community.vectorstores import Chroma




key = st.secrets["API_KEY"]
base_url = st.secrets["BASE_URL"]
model = st.secrets["MODEL"]

client = OpenAI(
  base_url=base_url,
  api_key=key
)

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# store = LocalFileStore("./cache/")

# Specify embedding model
embed_model_id = 'intfloat/e5-small-v2'
model_kwargs = {"device": "cpu", "trust_remote_code": True}

embeddings_model = HuggingFaceEmbeddings(model_name=embed_model_id, model_kwargs=model_kwargs)

# Create embeddings cache
# embedder = CacheBackedEmbeddings.from_bytes_store(embeddings_model, store, namespace=embed_model_id)


def extract_data(feed):
  data = []
  with pdfplumber.open(feed) as pdf:
    pages = pdf.pages
    for p in pages:
      data.append(p.extract_text_simple())
  return None # build more code to return a dataframe 


with st.sidebar:
  uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")


if uploaded_file is not None:
  data = extract_data(uploaded_file)
  print(data)

  chunks = splitter.split_documents(data)

  index = Chroma.from_documents(documents=chunks, embedding=embeddings_model)

  

  # Create retriever
  # retriever = vector_index.as_retriever()

# uploaded_file = st.file_uploader("Import image", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
# if uploaded_file is not None:
#   st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
#   st.write("Image uploaded successfully!")
#   # Convert the image to base64
#   bytes_data = uploaded_file.getvalue()
#   base64_image = base64.b64encode(bytes_data).decode('utf-8')
    

  # st.write("## Image Analysis")

  # completion = client.chat.completions.create(
  #   extra_body={},
  #   model=model,
  #   messages=[
  #     {
  #       "role": "user",
  #       "content": [
  #         {
  #           "type": "text",
  #           "text": "What is in this image?"
  #         },
  #         {
  #           "type": "image_url",
  #           "image_url": {
  #             "url": f"data:image/jpeg;base64,{base64_image}"
  #           }
  #         }
  #       ]
  #     }
  #   ]
  # )

  st.write("Application")