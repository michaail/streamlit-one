import streamlit as st
from openai import OpenAI

key = st.secrets["API_KEY"]
base_url = st.secrets["BASE_URL"]
model = st.secrets["MODEL"]

client = OpenAI(
  base_url=base_url,
  api_keys=key
)

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
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
          }
        }
      ]
    }
  ]
)

st.write(completion.choices[0].message.content)