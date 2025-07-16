# backend/services/transcription_service.py
import os
import io
import logging
import traceback
import whisper
import torch
import soundfile as sf
import numpy as np
from typing import Union, Optional

class TranscriptionService:
    def transcribe_bytes(self, 
                          file_bytes: Union[bytes, io.BytesIO], 
                          file_ext: str = '.wav', 
                          language: Optional[str] = 'en') -> str:
        """
        Transcribe audio from bytes with advanced error handling
        """
        try:
            # Validate input
            if not file_bytes:
                raise ValueError("No audio bytes provided")
            
            # Convert to BytesIO if needed
            if isinstance(file_bytes, bytes):
                file_bytes = io.BytesIO(file_bytes)
            
            # Temporary file path
            temp_file_path = f"temp_audio_{int(torch.rand(1)[0] * 10000)}{file_ext}"
            
            try:
                # Advanced audio reading with multiple fallback strategies
                try:
                    # Try reading with soundfile
                    audio_data, sample_rate = sf.read(file_bytes)
                except Exception as sf_error:
                    logger.warning(f"SoundFile reading failed: {sf_error}")
                    
                    # Fallback: Try with librosa if available
                    try:
                        import librosa
                        audio_data, sample_rate = librosa.load(file_bytes, sr=None)
                    except ImportError:
                        logger.error("Librosa not installed. Install with: pip install librosa")
                        raise sf_error
                
                # Ensure audio data is valid
                if audio_data is None or len(audio_data) == 0:
                    raise ValueError("No audio data found in the file")
                
                # Save to temp file
                sf.write(temp_file_path, audio_data, sample_rate)
                
                # Transcribe
                transcript = self.transcribe_file(
                    temp_file_path, 
                    language=language
                )
                
                return transcript
            
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        
        except Exception as e:
            logger.error(f"Bytes transcription error: {e}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Audio processing failed: {str(e)}")

# Wrapper function with more informative error handling
def transcribe_audio(audio_input: Union[str, bytes, io.BytesIO], file_ext: str = '.wav') -> str:
    """
    Unified transcription wrapper with user-friendly messages
    """
    try:
        # Validate input type
        if audio_input is None:
            return "Error: No audio provided. Please upload an audio file."
        
        # Check input type and transcribe accordingly
        if isinstance(audio_input, str):
            # File path
            return transcription_service.transcribe_file(audio_input)
        else:
            # Bytes or BytesIO
            return transcription_service.transcribe_bytes(audio_input, file_ext)
    
    except Exception as e:
        error_message = str(e)
        
        # Provide more specific error messages
        if "Format not recognised" in error_message:
            return "❌ Oops! The audio file format is not supported. Try another file."
        elif "No audio data found" in error_message:
            return "⚠️ The audio file seems empty. Please check your file."
        else:
            logger.error(f"Transcription wrapper error: {e}")
            return f"😓 Transcription failed: {error_message}"