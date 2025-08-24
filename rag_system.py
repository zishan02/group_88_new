import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class RAGSystem:
    def __init__(self, documents, model_name="sentence-transformers/all-MiniLM-L6-v2", llm_name="distilgpt2"):
        self.documents = documents
        self.encoder = SentenceTransformer(model_name)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        
        # Build dense and sparse indexes
        self.doc_embeddings = self.encoder.encode(documents)
        self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.doc_embeddings.shape[1]))
        self.faiss_index.add_with_ids(self.doc_embeddings, np.array(range(len(documents))))
        
        self.tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def hybrid_retrieve(self, query, top_k=5):
        # Dense Retrieval (Vector Search)
        query_embedding = self.encoder.encode(query)
        D, I = self.faiss_index.search(np.array([query_embedding]), k=top_k)
        dense_results = [{'doc_id': int(i), 'score': float(d)} for i, d in zip(I[0], D[0])]
        
        # Sparse Retrieval (BM25)
        tokenized_query = query.split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_indices = np.argsort(sparse_scores)[::-1][:top_k]
        sparse_results = [{'doc_id': int(i), 'score': float(sparse_scores[i])} for i in sparse_indices]

        # Simple weighted fusion (can be improved with RRF)
        combined_results = {}
        for res in dense_results:
            combined_results[res['doc_id']] = combined_results.get(res['doc_id'], 0) + res['score'] * 0.7  # Weighted score
        for res in sparse_results:
            combined_results[res['doc_id']] = combined_results.get(res['doc_id'], 0) + res['score'] * 0.3
            
        sorted_results = sorted(combined_results.items(), key=lambda item: item[1], reverse=True)[:top_k]
        retrieved_docs = [self.documents[doc_id] for doc_id, _ in sorted_results]
        return retrieved_docs

    def generate_response(self, query, retrieved_docs):
        context = " ".join(retrieved_docs)
        prompt = f"Based on the following financial information, answer the question.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        input_ids = self.llm_tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True)
        attention_mask = (input_ids != self.llm_tokenizer.pad_token_id).float()
        
        with torch.no_grad():
            outputs = self.llm.generate(
                input_ids,
                max_length=512,
                num_return_sequences=1,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                attention_mask=attention_mask
            )
            
        answer = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Guardrail: Check for hallucinations
        confidence_score = 1.0  # Placeholder confidence calculation
        for doc in retrieved_docs:
            if answer in doc or any(term in doc for term in answer.split()):
                confidence_score -= 0.1 # Decrement for non-matches
        
        # Placeholder guardrail logic, needs refinement
        if "data not in scope" in answer.lower() or "not found" in answer.lower():
            confidence_score = 0.3
        
        return answer.split("Answer:")[1].strip(), confidence_score