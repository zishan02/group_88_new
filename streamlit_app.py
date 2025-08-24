import streamlit as st
import time
import json
import os
from rag_system import RAGSystem
from ft_system import FTSystem
from data_preprocessing import clean_and_segment_text, extract_text_from_pdf, chunk_text

st.set_page_config(page_title="RAG vs. Fine-Tuned Chatbot", layout="wide")

st.title("Financial Q&A System: RAG vs. Fine-Tuning")
st.markdown("This application demonstrates and compares a **Retrieval-Augmented Generation (RAG)** system and a **Fine-Tuned (FT)** language model for answering questions on company financial statements.")

@st.cache_resource
def load_rag_model():
    # Load and process data for RAG
    text_2023 = extract_text_from_pdf('data/financial_statements_2023.pdf')
    text_2024 = extract_text_from_pdf('data/financial_statements_2024.pdf')
    full_text = text_2023 + "\n" + text_2024
    cleaned_data = clean_and_segment_text(full_text)
    
    all_chunks = []
    for section, text in cleaned_data.items():
        all_chunks.extend(chunk_text(text, chunk_size=400, overlap=50))
    
    return RAGSystem(all_chunks)

@st.cache_resource
def load_ft_model():
    # Load the fine-tuned model
    return FTSystem(model_name="./models/fine_tuned_model")

st.sidebar.header("Settings")
mode = st.sidebar.radio("Select Model Mode:", ("RAG Chatbot", "Fine-Tuned Chatbot"))
st.sidebar.markdown("---")
st.sidebar.info("Enter a financial question about the company's last two years of statements.")

query = st.text_input("Your Question:")

if query:
    st.subheader("Results")
    st.markdown("---")

    if mode == "RAG Chatbot":
        st.header("RAG Chatbot")
        with st.spinner("Initializing and generating RAG response..."):
            rag_system = load_rag_model()
            start_time = time.time()
            retrieved_docs = rag_system.hybrid_retrieve(query)
            answer, confidence = rag_system.generate_response(query, retrieved_docs)
            response_time = time.time() - start_time
            
            st.write(f"**Answer:** {answer}")
            st.write(f"**Method Used:** {mode}")
            st.write(f"**Confidence Score:** {confidence:.2f}")
            st.write(f"**Response Time:** {response_time:.2f} seconds")
            
    elif mode == "Fine-Tuned Chatbot":
        st.header("Fine-Tuned Chatbot")
        with st.spinner("Initializing and generating Fine-Tuned response..."):
            ft_system = load_ft_model()
            start_time = time.time()
            answer, confidence = ft_system.get_ft_answer(query)
            response_time = time.time() - start_time
            
            st.write(f"**Answer:** {answer}")
            st.write(f"**Method Used:** {mode}")
            st.write(f"**Confidence Score:** {confidence:.2f}")
            st.write(f"**Response Time:** {response_time:.2f} seconds")