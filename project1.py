#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install pyqt5==5.12.3 pyqtwebengine==5.12.1


# In[2]:


# pip install streamlit


# In[3]:


# pip install python-dotenv


# In[4]:


# Create a .env file
with open('.env', 'w') as f:
    f.write('HUGGINGFACEHUB_API_TOKEN = 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\n')


# In[5]:


from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Access the Hugging Face API key
hf_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

print(f"Hugging Face API Key: {hf_api_key}")  # For debugging; remove or comment out in production


# In[6]:


import pickle

# Example object to pickle (e.g., a trained model, vector store, etc.)
example_data = {"example_key": "example_value"}

# Save the object to a .pkl file
file_path = 'example_data.pkl'

with open(file_path, 'wb') as f:
    pickle.dump(example_data, f)

print(f"Data has been saved to {file_path}")


# In[7]:


# Load the object from the .pkl file
with open(file_path, 'rb') as f:
    loaded_data = pickle.load(f)

print(f"Loaded Data: {loaded_data}")


# In[8]:


# pip install -U langchain-huggingface


# In[9]:


# pip install huggingface_hub


# In[10]:


import os
import streamlit as st
import pickle
import time
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env (for Hugging Face API key if needed)

st.title("EV Market Segmentation Research Tool 📊")
st.sidebar.title("EV Market Data URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_huggingface.pkl"

main_placeholder = st.empty()

# Initialize Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="gpt2",  # Choose a suitable Hugging Face model repo
    temperature=0.9,  # Set temperature directly
    max_length=500    # Set max_length directly
)

if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings using Hugging Face Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_huggingface = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_huggingface, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # Display the result
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)


# In[11]:





# In[ ]:




