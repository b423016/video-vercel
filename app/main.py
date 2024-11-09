from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import zipfile
import os
import uvicorn
import logging
from datetime import datetime
from pydantic import BaseModel
from typing import Optional
import json
# Import all utilities
from app.utils import generate_top_thumbnails
from app.utils import add_emoji_to_text_and_size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Video Thumbnail Generator",
    description="Generate AI-powered thumbnails from video content",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Create output directory
OUTPUT_DIR = Path("processed_images")
OUTPUT_DIR.mkdir(exist_ok=True)


# Pydantic model for the JSON input
class ThumbnailRequest(BaseModel):
    num_thumbnails: int


# Helper function to save thumbnail
def save_thumbnail(image_bytes: bytes, filename: str) -> Path:
    """Save thumbnail to file system."""
    file_path = OUTPUT_DIR / f"{filename}.png"
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    return file_path


# Helper function to create a zip file
def create_zip(paths: list[Path], filename: str = "thumbnails.zip") -> Path:
    """Create ZIP archive of thumbnails."""
    zip_path = OUTPUT_DIR / filename
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for path in paths:
            zipf.write(path, arcname=path.name)
    return zip_path


# Cleanup old files
def cleanup_old_files():
    """Clean up files older than 1 hour."""
    current_time = datetime.now().timestamp()
    for file_path in OUTPUT_DIR.glob("*"):
        if current_time - file_path.stat().st_mtime > 3600:  # 1 hour
            file_path.unlink()


# Endpoint to process video and JSON data
@app.post("/process_video_and_json")
async def process_video_and_json(
        file: UploadFile = File(...),  # Video file
        json_data: UploadFile = File(...),  # JSON file containing num_thumbnails
):
    """
    Process uploaded video and generate AI-powered thumbnails.

    Parameters:
    - file: Video file (MP4 format recommended)
    - json_data: JSON file containing num_thumbnails

    Returns:
    - ZIP file containing generated thumbnails
    """
    try:
        # Read JSON data
        json_bytes = await json_data.read()
        json_content = json.loads(json_bytes.decode("utf-8"))
        num_thumbnails = json_content.get("num_thumbnails", 5)

        # Log the text with emoji and font size
        logger.info(add_emoji_to_text_and_size(f"Processing video: {file.filename}", font_size=100))
        logger.info(add_emoji_to_text_and_size("Generating thumbnails...", font_size=100))

        # Validate video file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload MP4, AVI, or MOV file."
            )

        # Read video file
        video_bytes = await file.read()

        # Generate thumbnails
        logger.info("Generating thumbnails...")
        thumbnails = generate_top_thumbnails(video_bytes)

        # Save thumbnails
        thumbnail_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = Path(file.filename).stem

        for i, thumbnail in enumerate(thumbnails[:num_thumbnails]):  # Only generate num_thumbnails
            filename = f"{original_filename}_{timestamp}_thumb_{i + 1}"
            path = save_thumbnail(thumbnail, filename)
            thumbnail_paths.append(path)

        # Create ZIP file with thumbnails
        zip_filename = f"{original_filename}_{timestamp}_thumbnails.zip"
        zip_path = create_zip(thumbnail_paths, zip_filename)

        # Clean up old files
        cleanup_old_files()

        logger.info("Processing completed successfully")

        # Return ZIP file
        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=zip_filename
        )

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
