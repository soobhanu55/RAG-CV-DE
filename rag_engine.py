import numpy as np
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
import json
import os

class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) Engine for CV processing
    Implements semantic search with FAISS and LLM integration
    """
    
    def __init__(self, api_key: str = None, embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"):
        self.api_key = api_key
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.dimension = 768  # Standard dimension for multilingual models
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using neural network-based transformer"""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return np.array(embeddings).astype('float32')
    
    def add_document(self, cv_data: Dict[str, Any], filename: str):
        """Add CV document to RAG system"""
        # Create document representation
        doc_text = self._cv_to_text(cv_data)
        
        doc_entry = {
            'filename': filename,
            'content': doc_text,
            'data': cv_data,
            'metadata': {
                'name': cv_data.get('personal_info', {}).get('name', 'Unknown'),
                'skills': cv_data.get('skills', []),
                'experience_years': cv_data.get('total_experience_years', 0)
            }
        }
        
        self.documents.append(doc_entry)
        self._rebuild_index()
    
    def _cv_to_text(self, cv_data: Dict[str, Any]) -> str:
        """Convert structured CV data to text for embedding"""
        parts = []
        
        # Personal info
        if 'personal_info' in cv_data:
            info = cv_data['personal_info']
            parts.append(f"Name: {info.get('name', '')}")
            parts.append(f"Email: {info.get('email', '')}")
            parts.append(f"Phone: {info.get('phone', '')}")
        
        # Professional summary
        if 'summary' in cv_data:
            parts.append(f"Summary: {cv_data['summary']}")
        
        # Skills
        if 'skills' in cv_data:
            parts.append(f"Skills: {', '.join(cv_data['skills'])}")
        
        # Experience
        if 'experience' in cv_data:
            for exp in cv_data['experience']:
                parts.append(f"Position: {exp.get('title', '')} at {exp.get('company', '')}")
                parts.append(f"Duration: {exp.get('duration', '')}")
                parts.append(f"Description: {exp.get('description', '')}")
        
        # Education
        if 'education' in cv_data:
            for edu in cv_data['education']:
                parts.append(f"Education: {edu.get('degree', '')} from {edu.get('institution', '')}")
        
        return "\n".join(parts)
    
    def _rebuild_index(self):
        """Rebuild FAISS index with current documents"""
        if not self.documents:
            return
        
        # Extract texts and create embeddings
        texts = [doc['content'] for doc in self.documents]
        self.embeddings = self.create_embeddings(texts)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search across CV database"""
        if not self.documents or self.index is None:
            return []
        
        # Create query embedding
        query_embedding = self.create_embeddings([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Prepare results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'filename': doc['filename'],
                    'content': doc['content'][:500] + '...',
                    'metadata': doc['metadata'],
                    'score': float(1 / (1 + dist))  # Convert distance to similarity score
                })
        
        return results
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        RAG-based query with LLM integration
        Retrieves relevant CVs and generates answer using LLM
        """
        # Retrieve relevant documents
        relevant_docs = self.search(question, top_k=top_k)
        
        if not relevant_docs:
            return {
                'answer': 'No relevant CVs found in the database.',
                'sources': []
            }
        
        # Build context from retrieved documents
        context = self._build_context(relevant_docs)
        
        # Generate answer (mock implementation - replace with actual LLM API call)
        answer = self._generate_answer(question, context)
        
        return {
            'answer': answer,
            'sources': relevant_docs,
            'context': context[:500]
        }
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context from retrieved documents"""
        context_parts = []
        for idx, doc in enumerate(documents, 1):
            context_parts.append(f"CV {idx} ({doc['filename']}):\n{doc['content']}\n")
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM (OpenAI API)
        Falls back to extractive answer if API key not available
        """
        if not self.api_key:
            return self._extractive_answer(question, context)
        
        try:
            import openai
            openai.api_key = self.api_key
            
            prompt = f"""You are an expert HR assistant analyzing German CVs. 
Based on the following CV information, answer the question comprehensively.

Context from CVs:
{context}

Question: {question}

Please provide a detailed answer in German or English, as appropriate. Focus on specific candidates and their qualifications."""
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert HR assistant for analyzing CVs."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"LLM API Error: {str(e)}. Returning extractive answer.\n\n{self._extractive_answer(question, context)}"
    
    def _extractive_answer(self, question: str, context: str) -> str:
        """Simple extractive answer when LLM is not available"""
        lines = context.split('\n')[:10]
        return "Based on the CVs in the database:\n\n" + "\n".join(lines)
