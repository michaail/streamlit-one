import streamlit as st
from openai import OpenAI
import base64
import pdfplumber

key = st.secrets["API_KEY"]
base_url = st.secrets["BASE_URL"]
model = st.secrets["MODEL"]

client = OpenAI(
  base_url=base_url,
  api_key=key
)

def extract_data(feed):
    data = []
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_tables())
    return None # build more code to return a dataframe 

with st.sidebar:
  uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")


if uploaded_file is not None:
  df = extract_data(uploaded_file)

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