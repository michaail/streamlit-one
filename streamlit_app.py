import streamlit as st
from openai import OpenAI
import base64

key = st.secrets["API_KEY"]
base_url = st.secrets["BASE_URL"]
model = st.secrets["MODEL"]

client = OpenAI(
  base_url=base_url,
  api_key=key
)

uploaded_file = st.file_uploader("Import image", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
if uploaded_file is not None:
  st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
  st.write("Image uploaded successfully!")
  bytes_data = uploaded_file.getvalue()
  data = base64.b64encode(bytes_data)

  st.write("## Image Analysis")

  completion = client.chat.completions.create(
    extra_body={},
    model=model,
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "input_image",
            "image_url": f"data:image/jpg;{uploaded_file}"
          }
        ]
      }
    ]
  )

  st.write(completion.choices[0].message.content)