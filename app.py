from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import uuid

from config import OPENAI_API_KEY, OPENAI_MODEL, CHROMA_DB_PATH, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from database.vector_db import VectorDatabase
from llm.openai_handler import OpenAIHandler
from processors.text_processor import TextProcessor
from processors.image_processor import ImageProcessor
from processors.media_processor import MediaProcessor
from utils.file_handler import FileHandler

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

def get_db():
    """Get or create database instance for session"""
    if 'db' not in session or not hasattr(app, 'dbs'):
        if not hasattr(app, 'dbs'):
            app.dbs = {}
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        if session_id not in app.dbs:
            app.dbs[session_id] = VectorDatabase(CHROMA_DB_PATH, COLLECTION_NAME)
        return app.dbs[session_id]
    return app.dbs[session['session_id']]

def get_llm():
    """Get or create LLM instance for session"""
    if 'llm' not in session or not hasattr(app, 'llms'):
        if not hasattr(app, 'llms'):
            app.llms = {}
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        if session_id not in app.llms:
            app.llms[session_id] = OpenAIHandler(api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)
        return app.llms[session_id]
    return app.llms[session['session_id']]

def get_chat_history():
    """Get chat history for session"""
    return session.get('chat_history', [])

def add_chat_message(role, content):
    """Add message to chat history"""
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append({"role": role, "content": content})
    session.modified = True

def get_processed_files():
    """Get processed files for session"""
    return session.get('processed_files', [])

def add_processed_file(name, status, message):
    """Add processed file to session"""
    if 'processed_files' not in session:
        session['processed_files'] = []
    session['processed_files'].append({'name': name, 'status': status, 'message': message})
    session.modified = True

def process_text_document(file_path, file_name):
    """Process text document and add to database"""
    extension = os.path.splitext(file_name)[1].lower()
    try:
        if extension == '.pdf':
            text = TextProcessor.process_pdf(file_path)
        elif extension == '.docx':
            text = TextProcessor.process_docx(file_path)
        elif extension == '.pptx':
            text = TextProcessor.process_pptx(file_path)
        elif extension == '.txt':
            text = TextProcessor.process_txt(file_path)
        elif extension == '.md':
            text = TextProcessor.process_markdown(file_path)
        else:
            return None, "Unsupported format"

        if not text or len(text.strip()) == 0:
            return None, "No text found"

        chunks = []
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i:i + CHUNK_SIZE]
            if chunk.strip():
                chunks.append(chunk)

        metadatas = [{"source": file_name, "type": "text", "chunk": i} for i in range(len(chunks))]
        db = get_db()
        db.add_documents(chunks, metadatas)
        return len(chunks), None
    except Exception as e:
        return None, str(e)

def process_image_file(file_path, file_name):
    """Process image file and add to database"""
    try:
        image_info = ImageProcessor.process_image(file_path)
        prompt = "Describe this image in detail. What do you see? Include objects, people, colors, text, and any other relevant details."
        llm = get_llm()
        description = llm.analyze_image(file_path, prompt)

        if description:
            db = get_db()
            db.add_documents([description], [{"source": file_name, "type": "image"}])
            return description, None
        return None, "Failed to analyze"
    except Exception as e:
        return None, str(e)

def process_video_file(file_path, file_name):
    """Process video file and add to database"""
    try:
        frame_paths = MediaProcessor.extract_video_frames(file_path, num_frames=8)

        if frame_paths:
            prompt = "Analyze these video frames in sequence and provide a comprehensive summary of what happens in the video. Describe the main events, actions, objects, and any text visible."
            llm = get_llm()
            summary = llm.analyze_video_frames(frame_paths, prompt)

            if summary:
                db = get_db()
                db.add_documents([summary], [{"source": file_name, "type": "video"}])

                for fp in frame_paths:
                    try:
                        if os.path.exists(fp):
                            os.remove(fp)
                    except:
                        pass
                return summary, None
            return None, "Failed to analyze"
        return None, "Failed to extract frames"
    except Exception as e:
        return None, str(e)

def process_audio_file(file_path, file_name):
    """Process audio file and add to database"""
    try:
        transcript = MediaProcessor.transcribe_audio(file_path)

        if transcript:
            chunks = []
            for i in range(0, len(transcript), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk = transcript[i:i + CHUNK_SIZE]
                if chunk.strip():
                    chunks.append(chunk)

            metadatas = [{"source": file_name, "type": "audio", "chunk": i} for i in range(len(chunks))]
            db = get_db()
            db.add_documents(chunks, metadatas)
            return len(chunks), None
        return None, "Failed to transcribe"
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    """Main page"""
    db = get_db()
    doc_count = db.count_documents()
    chat_history = get_chat_history()
    processed_files = get_processed_files()
    return render_template('index.html',
                         doc_count=doc_count,
                         chat_history=chat_history,
                         processed_files=processed_files)

@app.route('/api/process_files', methods=['POST'])
def api_process_files():
    """API endpoint to process uploaded files"""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({'success': False, 'error': 'No files provided'})

        success_count = 0
        total_files = len(files)

        for file in files:
            if file.filename == '':
                continue

            file_path = FileHandler.save_uploaded_file(file)
            if not file_path:
                add_processed_file(file.filename, 'error', 'Failed to save file')
                continue

            extension = os.path.splitext(file.filename)[1].lower()

            if extension in ['.pdf', '.docx', '.pptx', '.txt', '.md']:
                result, error = process_text_document(file_path, file.filename)
                if error:
                    add_processed_file(file.filename, 'error', f'Error: {error}')
                else:
                    add_processed_file(file.filename, 'success', f'✅ Processed {result} chunks')
                    success_count += 1
            elif extension in ['.png', '.jpg', '.jpeg']:
                result, error = process_image_file(file_path, file.filename)
                if error:
                    add_processed_file(file.filename, 'error', f'Error: {error}')
                else:
                    add_processed_file(file.filename, 'success', f'✅ Analyzed ({len(result)} chars)')
                    success_count += 1
            elif extension == '.mp4':
                result, error = process_video_file(file_path, file.filename)
                if error:
                    add_processed_file(file.filename, 'error', f'Error: {error}')
                else:
                    add_processed_file(file.filename, 'success', f'✅ Analyzed ({len(result)} chars)')
                    success_count += 1
            elif extension == '.mp3':
                result, error = process_audio_file(file_path, file.filename)
                if error:
                    add_processed_file(file.filename, 'error', f'Error: {error}')
                else:
                    add_processed_file(file.filename, 'success', f'✅ Processed {result} chunks')
                    success_count += 1
            else:
                add_processed_file(file.filename, 'error', f'Unsupported format: {extension}')

        return jsonify({
            'success': True,
            'processed_count': success_count,
            'total_count': total_files
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/process_youtube', methods=['POST'])
def api_process_youtube():
    """API endpoint to process YouTube video"""
    try:
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'success': False, 'error': 'No URL provided'})

        result = MediaProcessor.process_youtube(url)

        if result and 'content' in result:
            transcript = result['content']

            if transcript and len(transcript.strip()) > 0:
                chunks = []
                for i in range(0, len(transcript), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk = transcript[i:i + CHUNK_SIZE]
                    if chunk.strip():
                        chunks.append(chunk)

                metadatas = [{"source": url, "type": "youtube", "chunk": i} for i in range(len(chunks))]
                db = get_db()
                db.add_documents(chunks, metadatas)

                message = f"✅ Processed {len(chunks)} chunks ({len(transcript)} characters)"
                add_processed_file(url, 'success', message)
                return jsonify({'success': True, 'message': message})
            else:
                return jsonify({'success': False, 'error': 'No transcript found or empty'})
        return jsonify({'success': False, 'error': 'Failed to extract content'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for chat functionality"""
    try:
        data = request.get_json()
        message = data.get('message')

        if not message:
            return jsonify({'success': False, 'error': 'No message provided'})

        add_chat_message('user', message)

        # Try multiple search strategies to find relevant content
        db = get_db()
        results = db.query(message, n_results=15)

        if results and results.get('documents') and results['documents'][0]:
            context_docs = results['documents'][0]

            # Filter out empty contexts
            context_docs = [doc for doc in context_docs if doc and doc.strip()]

            # If we got very few results, try broader search
            if len(context_docs) < 5 and db.count_documents() > 10:
                results = db.query(message, n_results=20)
                if results and results.get('documents') and results['documents'][0]:
                    context_docs = [doc for doc in results['documents'][0] if doc and doc.strip()]

            if context_docs:
                context = "\n\n".join(context_docs)

                enhanced_prompt = f"""You are an intelligent AI assistant with strong language understanding capabilities. You should:
1. Automatically understand and correct any spelling mistakes or typos in the user's question
2. Interpret the user's intent even if the question is poorly written
3. Answer in clear, natural, easy-to-understand language
4. Be conversational and helpful

Context from uploaded documents:
{context}

User's Question (may contain typos - understand the intent): {message}

Instructions:
- First, understand what the user is really asking (correct any spelling/grammar issues mentally)
- Then, provide a comprehensive answer based ONLY on the context above
- Answer in natural, conversational language that's easy to understand
- If the context doesn't contain the answer, politely say "I don't have that information in the uploaded documents"

Your helpful answer:"""

                llm = get_llm()
                response = llm.generate_response(enhanced_prompt, context=None)
            else:
                response = "I found some documents but they don't contain relevant information to answer your question. 📄"
        else:
            response = "I don't have any relevant information to answer that question. Please upload some documents first! 📁"

        add_chat_message('assistant', response)
        return jsonify({'success': True, 'response': response})

    except Exception as e:
        error_msg = f"Error processing chat: {str(e)}"
        add_chat_message('assistant', error_msg)
        return jsonify({'success': False, 'error': error_msg})

@app.route('/api/clear_database', methods=['POST'])
def api_clear_database():
    """API endpoint to clear database"""
    try:
        session_id = session.get('session_id')
        if session_id and hasattr(app, 'dbs') and session_id in app.dbs:
            app.dbs[session_id].delete_collection()
            app.dbs[session_id] = VectorDatabase(CHROMA_DB_PATH, COLLECTION_NAME)

        if session_id and hasattr(app, 'llms') and session_id in app.llms:
            del app.llms[session_id]

        session['chat_history'] = []
        session['processed_files'] = []
        session.modified = True

        return jsonify({'success': True, 'message': 'Database cleared successfully'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)