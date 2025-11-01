"""
Configuration file for Multimodal Data Processing System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Select which LLM to use: "openai" or "gemini"
LLM_PROVIDER = "openai"  # Change to "gemini" if you want to use Gemini

# Database Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "multimodal_knowledge_base"

# File Upload Configuration
UPLOAD_FOLDER = "./uploads"
ALLOWED_TEXT_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.md', '.txt'}
ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
ALLOWED_MEDIA_EXTENSIONS = {'.mp3', '.mp4'}
MAX_FILE_SIZE_MB = 200

# Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM Configuration
GEMINI_MODEL = "models/gemini-1.5-flash-latest"
GEMINI_VISION_MODEL = "models/gemini-1.5-flash-latest"
OPENAI_MODEL = "gpt-4o-mini"  # or "gpt-4o", "gpt-3.5-turbo"
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 1024

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
