import fitz  # PyMuPDF
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_text_from_pdf(pdf_path):
    """
    Extracts plain text from a PDF document.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def clean_and_segment_text(text):
    """
    Cleans text by removing noise and segments it into logical sections.
    """
    # Simple cleaning for headers/footers (can be improved with more complex regex)
    cleaned_text = re.sub(r'Page \d+ of \d+', '', text)
    cleaned_text = re.sub(r'[\r\n]+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Simple segmentation based on keywords (e.g., "Income Statement")
    segments = re.split(r'(Income Statement|Balance Sheet|Cash Flow Statement)', cleaned_text, flags=re.IGNORECASE)
    
    financial_data = {}
    current_section = None
    for segment in segments:
        if re.match(r'Income Statement|Balance Sheet|Cash Flow Statement', segment, re.IGNORECASE):
            current_section = segment.strip().lower()
            financial_data[current_section] = ""
        elif current_section:
            financial_data[current_section] += segment.strip()
    
    # This example is a placeholder. For robust segmentation, use a more sophisticated approach.
    return financial_data

def chunk_text(text, chunk_size, overlap):
    """
    Splits text into chunks of a specified size with overlap.
    """
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

if __name__ == "__main__":
    # Path to your financial documents
    pdf_2023_path = 'data/financial_statements_2023.pdf'
    pdf_2024_path = 'data/financial_statements_2024.pdf'

    # Extract and clean text
    text_2023 = extract_text_from_pdf(pdf_2023_path)
    text_2024 = extract_text_from_pdf(pdf_2024_path)
    
    if text_2023 and text_2024:
        full_text = text_2023 + "\n" + text_2024
        cleaned_data = clean_and_segment_text(full_text)
        
        # Save segmented data for later use
        with open('data/processed_financial_data.json', 'w') as f:
            json.dump(cleaned_data, f)
        
        # Create chunks for RAG
        all_chunks = []
        for section, text in cleaned_data.items():
            chunks_100 = chunk_text(text, chunk_size=100, overlap=20)
            chunks_400 = chunk_text(text, chunk_size=400, overlap=50)
            all_chunks.extend(chunks_100)
            all_chunks.extend(chunks_400)
            
        print(f"Generated {len(all_chunks)} chunks for RAG.")