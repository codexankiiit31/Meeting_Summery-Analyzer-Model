# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

import os
import uuid
import shutil
import json
import logging
from typing import Optional

# Import services
from services.transcription_service import transcribe_audio
from services.llm_service import process_transcript
from services.rag_services import create_and_add_to_faiss, search_faiss
from services.email_service import send_email_summary, format_meeting_summary_email

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="AI Meeting Summary API",
    description="Intelligent meeting summarization and CRM insights generator",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
origins = [
    "http://localhost:8501",   # Streamlit default
    "http://127.0.0.1:8501",
    "http://localhost:3000",   # React/Next.js default
    "http://localhost:8000",   # Backend proxy
    "*"  # Be cautious with this in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create data directory
os.makedirs("data/uploads", exist_ok=True)

# Optional API Key Authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def validate_api_key(api_key: str = None):
    """
    Validate API key for enhanced security
    """
    expected_api_key = os.getenv("API_KEY")
    if not expected_api_key:
        logger.warning("No API key configured in environment")
        return True
    
    if not api_key or api_key != expected_api_key:
        raise HTTPException(
            status_code=403, 
            detail="Invalid or missing API key"
        )
    return True

@app.get("/", tags=["Health Check"])
async def root():
    """
    Basic health check endpoint
    """
    return {
        "status": "healthy",
        "service": "AI Meeting Summary API",
        "version": "1.0.0"
    }

@app.post("/process_meeting/", tags=["Meeting Processing"])
async def process_meeting_endpoint(
    file: Optional[UploadFile] = File(None),
    transcript_text: Optional[str] = Form(None),
    api_key: Optional[str] = None
):
    """
    Process a meeting recording or transcript
    
    Supports:
    - Audio file upload
    - Text transcript input
    - Comprehensive input validation
    """
    # Validate API key
    validate_api_key(api_key)
    
    # Input validation
    if not file and not transcript_text:
        raise HTTPException(
            status_code=400, 
            detail="Either upload a file or provide transcript text."
        )
    
    # Transcript length validation
    if transcript_text and len(transcript_text) < 50:
        raise HTTPException(
            status_code=400,
            detail="Transcript is too short. Provide a more detailed meeting transcript."
        )
    
    # File size limit
    if file and file.size > 50_000_000:  # 50 MB limit
        raise HTTPException(
            status_code=413, 
            detail="File too large. Maximum file size is 50 MB."
        )
    
    transcript = None
    temp_file_path = None
    
    try:
        # Handle file upload
        if file:
            # Generate unique filename
            unique_filename = f"{uuid.uuid4()}_{file.filename}"
            temp_file_path = f"data/uploads/{unique_filename}"
            
            # Save uploaded file
            with open(temp_file_path, "wb+") as file_object:
                shutil.copyfileobj(file.file, file_object)
            
            # Transcribe audio if needed
            if file.content_type.startswith("audio/"):
                logger.info(f"Transcribing audio file: {file.filename}")
                transcript = transcribe_audio(temp_file_path)
            elif file.content_type == "text/plain" or file.filename.endswith(".txt"):
                logger.info(f"Processing text file: {file.filename}")
                with open(temp_file_path, "r", encoding="utf-8") as f:
                    transcript = f.read()
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="Unsupported file type. Upload audio or text files."
                )
        
        # Handle text transcript
        elif transcript_text:
            logger.info("Processing provided transcript text")
            transcript = transcript_text
        
        # Validate transcript
        if not transcript or len(transcript.strip()) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Unable to generate transcript. The content is too short."
            )
        
        # Process transcript with LLM
        logger.info(f"Processing transcript of length {len(transcript)}")
        processed_data = process_transcript(transcript)
        
        # Generate unique meeting ID if not present
        if "meeting_id" not in processed_data:
            processed_data["meeting_id"] = str(uuid.uuid4())
        
        # Add to RAG index for future retrieval
        logger.info("Adding processed data to RAG index")
        create_and_add_to_faiss(processed_data)
        
        return processed_data
    
    except Exception as e:
        logger.error(f"Meeting processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/search_past_meetings/", tags=["Meeting Search"])
async def search_past_meetings_endpoint(
    query: str, 
    top_k: int = 3,
    api_key: Optional[str] = None
):
    """
    Search through past processed meetings
    
    Supports:
    - Semantic search
    - Configurable result count
    """
    # Validate API key
    validate_api_key(api_key)
    
    try:
        # Perform semantic search
        results = search_faiss(query, top_k)
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Meeting search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/email_summary/", tags=["Email"])
async def email_summary_endpoint(
    recipient_email: str = Form(...), 
    meeting_data_json: str = Form(...),
    api_key: Optional[str] = None
):
    """
    Email meeting summary to specified recipient
    
    Supports:
    - JSON meeting data
    - Email validation
    """
    # Validate API key
    validate_api_key(api_key)
    
    try:
        # Parse meeting data
        meeting_data = json.loads(meeting_data_json)
        
        # Format email content
        subject, body = format_meeting_summary_email(meeting_data)
        
        # Send email
        success = send_email_summary(
            to_email=recipient_email,
            subject=subject,
            body=body
        )
        
        if success:
            return {"message": "Email sent successfully"}
        else:
            return JSONResponse(
                status_code=500, 
                content={"message": "Failed to send email"}
            )
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid meeting data format")
    
    except Exception as e:
        logger.error(f"Email sending error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Global error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Global exception handler for HTTP exceptions
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": str(request.url)
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Perform startup initialization
    """
    logger.info("Starting AI Meeting Summary API")
    # Optional: Initialize any global resources or connections

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Perform cleanup on application shutdown
    """
    logger.info("Shutting down AI Meeting Summary API")
    # Optional: Close any open resources or connections

# Run the app (for direct script execution)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )