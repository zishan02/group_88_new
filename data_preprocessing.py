import fitz  # PyMuPDF
import re
import json
import torch
import os

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
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A string containing the concatenated text from all pages of the PDF.
        Returns an empty string if the file does not exist or an error occurs.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' does not exist.")
        return ""
    
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or "" # Handle pages with no extractable text
    except Exception as e:
        print(f"An error occurred while reading '{pdf_path}': {e}")
        return ""
    
    return text

def clean_and_segment_text(text: str) -> dict:
    """
    Cleans up the raw text and segments it into logical sections.
    This is a simple segmentation based on common financial document headings.

    Args:
        text: The raw, concatenated text from one or more financial documents.

    Returns:
        A dictionary where keys are section names and values are the cleaned text.
    """
    # Use regular expressions to clean up common artifacts like multiple newlines,
    # form feed characters, and unnecessary whitespace.
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # Simple segmentation based on keywords. This can be made more sophisticated.
    segments = {
        "Executive Summary": "",
        "Financial Statements": "",
        "Notes to Financial Statements": "",
        "Management Discussion": "",
        "Other Information": ""
    }

    # Split the text based on section headings
    sections = re.split(r'(?i)(executive summary|financial statements|notes to financial statements|management discussion|other information)', cleaned_text)

    # Assign text to the corresponding segment
    if len(sections) > 1:
        # The first element is usually empty or pre-summary text
        for i in range(1, len(sections), 2):
            section_title = sections[i].strip()
            section_content = sections[i+1].strip()
            
            # Match the section title to the dictionary keys
            for key in segments:
                if key.lower() == section_title.lower():
                    segments[key] = section_content
                    break
    else:
        # If no specific sections are found, dump all text into a single segment
        segments["Financial Statements"] = cleaned_text

    return segments

def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """
    Splits a string of text into smaller, overlapping chunks.

    Args:
        text: The input string.
        chunk_size: The desired size of each chunk.
        overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move the start pointer back by the overlap amount
        start += chunk_size - overlap
    
    return chunks

def process_all_data(doc_paths: list) -> list:
    """
    Orchestrates the entire data processing pipeline for multiple documents.

    Args:
        doc_paths: A list of file paths to the financial documents.

    Returns:
        A list of all generated text chunks.
    """
    full_text = ""
    for path in doc_paths:
        full_text += extract_text_from_pdf(path) + "\n"

    if not full_text.strip():
        print("No text could be extracted from the provided documents.")
        return []

    # Clean and segment the combined text
    cleaned_data = clean_and_segment_text(full_text)
    
    # Save the segmented data to a JSON file
    with open('data/processed_financial_data.json', 'w') as f:
        json.dump(cleaned_data, f, indent=4)
    print("Segmented data saved to 'data/processed_financial_data.json'.")

    all_chunks = []
    for section_name, text_content in cleaned_data.items():
        if text_content:
            # Create chunks of different sizes to capture various levels of detail
            chunks_100 = chunk_text(text_content, chunk_size=100, overlap=20)
            chunks_400 = chunk_text(text_content, chunk_size=400, overlap=50)
            
            all_chunks.extend(chunks_100)
            all_chunks.extend(chunks_400)
    
    print(f"Generated {len(all_chunks)} chunks for RAG.")
    return all_chunks
if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Example usage:
    # First, make sure you have the required library installed: `pip install pdfplumber`
    # Then, place your financial documents in a 'data' folder.
    # For this example, let's assume 'data/Q1-2023.pdf' and 'data/Q1-2024.pdf' exist.
    
    # Path to your financial documents
    pdf_path = ['data/q4-2025', 'data/q3-2025','data/q2-2025', 'data/q1-2025','data/q4-2024', 'data/q3-2024','data/q2-2024', 'data/q1-2024']
    
    # Process the documents and get the final chunks
    final_chunks = process_all_data(pdf_path)
    
    if final_chunks:
        print("\nFirst 3 chunks generated for your RAG system:")
        for chunk in final_chunks[:3]:
            print("---")
            print(chunk)
