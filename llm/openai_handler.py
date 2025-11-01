"""
OpenAI API handler for LLM operations
"""
from openai import OpenAI
from typing import List, Dict, Optional
from PIL import Image
import base64
import io


class OpenAIHandler:
    """Handle OpenAI API operations"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        """
        Initialize OpenAI handler
        
        Args:
            api_key: OpenAI API key (or OpenRouter key)
            model_name: Model name to use (gpt-4o-mini, gpt-4o, gpt-3.5-turbo)
        """
        # Check if using OpenRouter
        if api_key.startswith("sk-or-"):
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            # Use OpenRouter model names
            if model_name == "gpt-4o-mini":
                model_name = "openai/gpt-4o-mini"
            elif model_name == "gpt-4o":
                model_name = "openai/gpt-4o"
        else:
            self.client = OpenAI(api_key=api_key)
        
        self.model_name = model_name
        
        # Generation config
        self.temperature = 0.7
        self.max_tokens = 2048  # Increased for more detailed responses
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate response using OpenAI
        
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
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an exceptionally intelligent and helpful AI assistant. You have strong language understanding capabilities and can automatically interpret user questions even with spelling mistakes, typos, or grammatical errors. Always answer in clear, natural, conversational language that's easy to understand. Focus on being helpful and providing accurate information based on the provided context."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze image using vision model
        
        Args:
            image_path: Path to image file
            prompt: Question about the image
            
        Returns:
            Analysis result
        """
        try:
            # Load and encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Use appropriate vision model based on API
            if "openai/" in self.model_name or self.model_name.startswith("sk-or"):
                vision_model = "google/gemini-2.0-flash-exp:free"
            else:
                vision_model = "gpt-4o"
            
            print(f"Analyzing image with model: {vision_model}")
            
            response = self.client.chat.completions.create(
                model=vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            print(f"Image analysis successful: {len(result)} characters")
            return result
        except Exception as e:
            error_msg = f"Error analyzing image: {str(e)}"
            print(error_msg)
            return error_msg
    
    def analyze_video_frames(self, frame_paths: List[str], prompt: str) -> str:
        """
        Analyze multiple video frames and generate a summary
        
        Args:
            frame_paths: List of paths to frame images
            prompt: Question about the video
            
        Returns:
            Video analysis result
        """
        # Encode all frames
        frame_images = []
        for frame_path in frame_paths[:10]:  # Limit to 10 frames
            try:
                with open(frame_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    frame_images.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
            except Exception as e:
                print(f"Error encoding frame {frame_path}: {e}")
        
        if not frame_images:
            return "Error: No frames could be loaded for analysis"
        
        # Try multiple vision models in order
        vision_models = [
            "google/gemini-2.0-flash-exp:free",
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet"
        ]
        
        if not ("openai/" in self.model_name or self.model_name.startswith("sk-or")):
            vision_models = ["gpt-4o"]  # Use OpenAI's model directly
        
        for vision_model in vision_models:
            try:
                print(f"Analyzing {len(frame_images)} video frames with model: {vision_model}")
                
                # Build content with text prompt and all frames
                content = [{"type": "text", "text": prompt}] + frame_images
                
                response = self.client.chat.completions.create(
                    model=vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=1000
                )
                
                result = response.choices[0].message.content
                print(f"Video analysis successful with {vision_model}: {len(result)} characters")
                return result
            except Exception as e:
                error_msg = f"Error with {vision_model}: {str(e)}"
                print(error_msg)
                # Try next model
                continue
        
        # All models failed
        return "Error: All vision models failed to analyze the video. The service may be rate-limited. Please try again later or with fewer/shorter videos."
    
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
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes text concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=max_length * 2
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error summarizing text: {str(e)}"
    
    def answer_with_context(self, question: str, context_docs: List[str]) -> str:
        """
        Answer question based on provided context documents
        
        Args:
            question: User question
            context_docs: List of relevant documents
            
        Returns:
            Answer based on context
        """
        try:
            # Combine context documents
            combined_context = "\n\n".join(context_docs)
            
            # Limit context length
            max_context_chars = 4000
            if len(combined_context) > max_context_chars:
                combined_context = combined_context[:max_context_chars] + "..."
            
            return self.generate_response(question, combined_context)
        except Exception as e:
            return f"Error answering question: {str(e)}"
