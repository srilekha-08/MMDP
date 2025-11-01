"""
Gemini API handler for LLM operations
"""
import google.generativeai as genai
from typing import List, Dict, Optional
from PIL import Image


class GeminiHandler:
    """Handle Gemini API operations"""
    
    def __init__(self, api_key: str, model_name: str = "models/gemini-1.5-flash-latest"):
        """
        Initialize Gemini handler
        
        Args:
            api_key: Gemini API key
            model_name: Model name to use
        """
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        # Generation config
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate response using Gemini
        
        Args:
            prompt: User prompt/query
            context: Optional context to include
            
        Returns:
            Generated response
        """
        try:
            if context:
                full_prompt = f"""Context Information:
{context}

User Question: {prompt}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain relevant information, say so clearly."""
            else:
                full_prompt = prompt
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=self.generation_config
            )
            
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze image using Gemini Vision
        
        Args:
            image_path: Path to image file
            prompt: Question about the image
            
        Returns:
            Analysis result
        """
        try:
            # Use vision model (gemini-1.5-flash supports vision)
            vision_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
            
            # Load image
            img = Image.open(image_path)
            
            # Generate response
            response = vision_model.generate_content([prompt, img])
            
            return response.text
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text (placeholder - Gemini doesn't provide direct embeddings)
        For now, this is handled by ChromaDB's built-in embeddings
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # ChromaDB handles embeddings automatically
        # This method is here for future enhancements
        pass
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize long text
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary
        """
        try:
            prompt = f"Provide a concise summary of the following text in approximately {max_length} words:\n\n{text}"
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error summarizing text: {str(e)}"
    
    def answer_with_context(self, query: str, contexts: List[Dict]) -> str:
        """
        Answer query with multiple context documents
        
        Args:
            query: User query
            contexts: List of context documents with metadata
            
        Returns:
            Answer
        """
        try:
            # Build context string
            context_str = "\n\n".join([
                f"Document {i+1} (from {ctx.get('file_name', 'unknown')}):\n{ctx.get('content', '')}"
                for i, ctx in enumerate(contexts)
            ])
            
            prompt = f"""Based on the following documents, please answer the user's question.

Documents:
{context_str}

User Question: {query}

Instructions:
1. Provide a clear and comprehensive answer based on the documents
2. If relevant information is in multiple documents, synthesize them
3. Cite which document(s) you're referencing
4. If the documents don't contain enough information to answer, say so
5. Be concise but thorough

Answer:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"
