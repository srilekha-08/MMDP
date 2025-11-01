"""
Media processor for MP3, MP4, and YouTube videos
"""
import os
from typing import Dict
import tempfile

# Optional imports with fallbacks
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    from moviepy import VideoFileClip, AudioFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False


class MediaProcessor:
    """Process audio and video files"""
    
    @staticmethod
    def extract_audio_from_video(video_path: str) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
        """
        if not MOVIEPY_AVAILABLE:
            raise Exception("moviepy is not installed. Install it with: pip install moviepy")
        
        try:
            video = VideoFileClip(video_path)
            audio = video.audio
            
            # Create temp audio file
            audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
            audio.write_audiofile(audio_path, logger=None)
            
            video.close()
            audio.close()
            
            return audio_path
        except Exception as e:
            raise Exception(f"Error extracting audio from video: {str(e)}")
    
    @staticmethod
    def split_audio_into_chunks(audio_path: str, chunk_duration_seconds: int = 60) -> list:
        """
        Split audio file into smaller chunks (default 60 seconds)
        
        Args:
            audio_path: Path to audio file
            chunk_duration_seconds: Duration of each chunk in seconds
            
        Returns:
            List of chunk file paths
        """
        if not MOVIEPY_AVAILABLE:
            return [audio_path]
        
        try:
            from moviepy import AudioFileClip
            audio = AudioFileClip(audio_path)
            duration = audio.duration
            
            chunks = []
            chunk_index = 0
            for i in range(0, int(duration), chunk_duration_seconds):
                end_time = min(i + chunk_duration_seconds, duration)
                chunk = audio.subclipped(i, end_time)
                chunk_path = audio_path.rsplit('.', 1)[0] + f'_chunk_{chunk_index}.wav'
                # Use mono, 16kHz to minimize file size
                chunk.write_audiofile(chunk_path, logger=None, fps=16000, nbytes=2, codec='pcm_s16le')
                chunks.append(chunk_path)
                chunk_index += 1
            
            audio.close()
            return chunks
        except Exception as e:
            print(f"Error splitting audio: {e}")
            return [audio_path]
    
    @staticmethod
    def transcribe_audio(audio_path: str) -> str:
        """
        Transcribe audio to text using speech recognition
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Convert to WAV first if needed (for proper size checking and compatibility)
            wav_path = audio_path
            if not audio_path.endswith('.wav'):
                if MOVIEPY_AVAILABLE:
                    from moviepy import AudioFileClip
                    audio = AudioFileClip(audio_path)
                    wav_path = audio_path.rsplit('.', 1)[0] + '_temp.wav'
                    # Use mono, 16kHz to reduce file size (perfect for speech recognition)
                    audio.write_audiofile(wav_path, logger=None, fps=16000, nbytes=2, codec='pcm_s16le')
                    audio.close()
            
            # Try OpenAI Whisper first (more reliable than Google Speech Recognition)
            from config import OPENAI_API_KEY
            if OPENAI_API_KEY:
                try:
                    from openai import OpenAI
                    
                    # OpenRouter doesn't support Whisper API, so use OpenAI directly
                    # If using OpenRouter key, skip Whisper and go straight to Google Speech
                    if OPENAI_API_KEY.startswith("sk-or-"):
                        print("OpenRouter doesn't support Whisper API, using Google Speech Recognition")
                        raise Exception("OpenRouter not compatible with Whisper")
                    
                    # Use OpenAI Whisper
                    client = OpenAI(api_key=OPENAI_API_KEY)
                    model = "whisper-1"
                    
                    # Check file size - if > 20MB, split into chunks
                    file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
                    
                    if file_size_mb > 20:
                        print(f"Audio file is {file_size_mb:.2f}MB, splitting into 1-minute chunks...")
                        chunks = MediaProcessor.split_audio_into_chunks(wav_path, chunk_duration_seconds=60)
                        transcripts = []
                        
                        for chunk_path in chunks:
                            try:
                                with open(chunk_path, "rb") as audio_file:
                                    transcript = client.audio.transcriptions.create(
                                        model=model,
                                        file=audio_file,
                                        response_format="text"
                                    )
                                    transcripts.append(transcript)
                                    print(f"Successfully transcribed chunk: {os.path.basename(chunk_path)}")
                            except Exception as chunk_error:
                                print(f"Error transcribing chunk {chunk_path}: {chunk_error}")
                            
                            # Clean up chunk file
                            if chunk_path != wav_path:
                                try:
                                    os.remove(chunk_path)
                                except:
                                    pass
                        
                        if transcripts:
                            full_transcript = " ".join(transcripts)
                            print(f"Successfully transcribed {len(transcripts)} chunks, total length: {len(full_transcript)} characters")
                            return full_transcript
                        else:
                            raise Exception("All chunk transcriptions failed")
                    else:
                        with open(wav_path, "rb") as audio_file:
                            transcript = client.audio.transcriptions.create(
                                model=model,
                                file=audio_file,
                                response_format="text"
                            )
                        return transcript
                except Exception as whisper_error:
                    print(f"Whisper API failed: {whisper_error}, falling back to Google Speech Recognition")
            
            # Fallback to Google Speech Recognition
            if not SPEECH_RECOGNITION_AVAILABLE:
                return "[Speech recognition not available. Install with: pip install SpeechRecognition]"
            
            recognizer = sr.Recognizer()
            
            # For large files, split and transcribe in chunks
            file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
            
            if file_size_mb > 10:
                print(f"Large audio file ({file_size_mb:.2f}MB), transcribing in chunks with Google Speech Recognition...")
                chunks = MediaProcessor.split_audio_into_chunks(wav_path, chunk_duration_seconds=30)
                transcripts = []
                
                for idx, chunk_path in enumerate(chunks):
                    try:
                        with sr.AudioFile(chunk_path) as source:
                            # Adjust for ambient noise
                            recognizer.adjust_for_ambient_noise(source, duration=0.5)
                            audio_data = recognizer.record(source)
                            # Try with language detection
                            text = recognizer.recognize_google(audio_data, show_all=False)
                            transcripts.append(text)
                            print(f"Transcribed chunk {idx+1}/{len(chunks)}: {len(text)} chars")
                    except sr.UnknownValueError:
                        print(f"Could not understand audio in chunk {idx+1}")
                        # Don't add empty transcripts
                    except sr.RequestError as e:
                        print(f"Error in chunk {idx+1}: {e}")
                    
                    # Clean up chunk
                    if chunk_path != wav_path:
                        try:
                            os.remove(chunk_path)
                        except:
                            pass
                
                if transcripts:
                    result = " ".join(transcripts)
                    print(f"Total transcription: {len(result)} characters from {len(transcripts)} chunks")
                    return result
                else:
                    return "[No speech detected in audio - may be music, sound effects, or non-English content]"
            else:
                # Small file, transcribe directly
                with sr.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)
                
                return text
        except sr.UnknownValueError:
            return "[Could not understand audio - speech may be unclear or in unsupported language]"
        except sr.RequestError as e:
            return f"[Speech recognition service error: {str(e)}. Check internet connection]"
        except Exception as e:
            return f"[Transcription failed: {str(e)}]"
    
    @staticmethod
    def extract_video_frames(video_path: str, num_frames: int = 8) -> list:
        """
        Extract frames from video for visual analysis
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of frame file paths
        """
        if not MOVIEPY_AVAILABLE:
            return []
        
        try:
            from moviepy import VideoFileClip
            video = VideoFileClip(video_path)
            duration = video.duration
            
            # Extract frames evenly distributed across the video
            frame_paths = []
            for i in range(num_frames):
                time = (i * duration) / num_frames
                frame = video.get_frame(time)
                
                # Save frame as JPEG
                frame_path = video_path.rsplit('.', 1)[0] + f'_frame_{i}.jpg'
                from PIL import Image
                Image.fromarray(frame).save(frame_path, 'JPEG', quality=85)
                frame_paths.append(frame_path)
            
            video.close()
            return frame_paths
        except Exception as e:
            print(f"Error extracting video frames: {e}")
            return []
    
    @staticmethod
    def process_audio(file_path: str) -> Dict:
        """
        Process audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio info and transcription
        """
        try:
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Transcribe audio
            transcription = MediaProcessor.transcribe_audio(file_path)
            
            return {
                "file_name": file_name,
                "file_type": file_ext[1:],
                "content": transcription,
                "content_type": "audio",
                "original_path": file_path
            }
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")
    
    @staticmethod
    def process_video(file_path: str, use_vision: bool = True) -> Dict:
        """
        Process video file using vision analysis or audio transcription
        
        Args:
            file_path: Path to video file
            use_vision: Use vision model to analyze frames (recommended)
            
        Returns:
            Dictionary with video info and analysis/transcription
        """
        try:
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if use_vision:
                # Extract frames for visual analysis
                print(f"Extracting frames from video: {file_name}")
                frame_paths = MediaProcessor.extract_video_frames(file_path, num_frames=8)
                
                if frame_paths:
                    # Return frame paths for vision analysis
                    return {
                        "file_name": file_name,
                        "file_type": file_ext[1:],
                        "content": "",  # Will be filled by vision model
                        "content_type": "video_frames",
                        "original_path": file_path,
                        "frame_paths": frame_paths
                    }
            
            # Fallback to audio transcription
            print(f"Using audio transcription for video: {file_name}")
            # Extract audio from video
            audio_path = MediaProcessor.extract_audio_from_video(file_path)
            
            # Transcribe audio
            transcription = MediaProcessor.transcribe_audio(audio_path)
            
            # Cleanup temp audio file
            if os.path.exists(audio_path) and audio_path != file_path:
                os.remove(audio_path)
            
            return {
                "file_name": file_name,
                "file_type": file_ext[1:],
                "content": transcription,
                "content_type": "video",
                "original_path": file_path
            }
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")
    
    @staticmethod
    def download_youtube_video(url: str, output_path: str = "./uploads") -> str:
        """
        Download YouTube video
        
        Args:
            url: YouTube video URL
            output_path: Directory to save video
            
        Returns:
            Path to downloaded video
        """
        if not YT_DLP_AVAILABLE:
            raise Exception("yt-dlp is not installed. Install it with: pip install yt-dlp")
        
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                return filename
        except Exception as e:
            raise Exception(f"Error downloading YouTube video: {str(e)}")
    
    @staticmethod
    def process_youtube(url: str) -> Dict:
        """
        Process YouTube video
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary with video info and transcription
        """
        try:
            # Download video/audio
            audio_path = MediaProcessor.download_youtube_video(url)
            
            # Process as audio
            result = MediaProcessor.process_audio(audio_path)
            result["content_type"] = "youtube"
            result["url"] = url
            
            return result
        except Exception as e:
            raise Exception(f"Error processing YouTube video: {str(e)}")
