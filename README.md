# Chatbot_with_Langchain
# Chatbot with LangChain

A chatbot using **LangChain**, **ChromaDB**, and **Mistral 7B** for Retrieval-Augmented Generation (RAG).  
It retrieves relevant product data from stored embeddings and uses Mistral 7B via Hugging Face API.

## Features
✅ Uses **ChromaDB** for fast retrieval  
✅ **Keyword Matching** for query classification  
✅ Runs on **Streamlit UI**  
✅ **No document upload** – Uses preloaded embeddings  
✅ Uses **Hugging Face Hub** for Mistral 7B inference  

## Setup Instructions
**1.** Clone the repo:  
   ```bash
   git clone https://github.com/IYNESHDURAI/chatbot-with-langchain.git
   cd chatbot-with-langchain

**2.** Install dependencies:
 ```bash
pip install -r requirements.txt


**3**.Run the chatbot:
 ```bash
streamlit run chatbot_ui.py
