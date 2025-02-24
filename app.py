import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.llms import HuggingFaceHub



#Hugging face token api load
import os
import dotenv
from dotenv import load_dotenv

load_dotenv()
HuggingFaceHub_API_Key=os.getenv("Hf_api_key")
os.environ["HUGGINGFACEHUB_API_TOKEN"]=HuggingFaceHub_API_Key

llm= HuggingFaceHub(repo_id='mistralai/Mistral-7B-Instruct-v0.3', model_kwargs={"temperature":0.1,"max_length":512})

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="product_documents")

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Initialize Mistral 7B LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Streamlit UI setup
st.title("**Chatbot**")
st.write("Ask questions about company products!")

#query input
query = st.text_input("Ask Question:", "")

if query:
    # Embed query and retrieve relevant chunks
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    
    # Display retrieved chunks
    retrieved_chunks = results["documents"][0] if results["documents"] else ["No relevant data found."]
    context = "\n\n".join(retrieved_chunks)
    
    # Generate response using LLM
    prompt = f"Use the following information to answer the query:\n\n{context}\n\nQuery: {query}\nAnswer:"
    response = llm.invoke(prompt)
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Retrieved Chunks")
        st.write(context)
    
    with col2:
        st.subheader("Chatbot Response")
        st.write(response.split("Answer:")[-1].strip())
