import os
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

class RAGEngine:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.index = None

    def process_pdf(self, filepath):
        raw_text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n"
        return self._split_text(raw_text)

    def _split_text(self, text, chunk_size=500, overlap=100):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def build_index(self, chunks):
        self.chunks = chunks
        self.embeddings = embed_model.encode(chunks)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(self.embeddings))

    def query(self, question):
        query_embedding = embed_model.encode([question])
        D, I = self.index.search(np.array(query_embedding), k=5)
        context = "\n".join([self.chunks[i] for i in I[0]])
        prompt = f"""
You are a helpful assistant. Use the context below to answer the question:

Context:
{context}

Question:
{question}

Answer:
"""
        response = llm.generate_content(prompt)
        return response.text
