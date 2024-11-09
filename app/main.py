from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.utils import (
    extract_frame_from_video, preprocess_image,
    detect_scene, detect_emotions, detect_style,
    generate_branding_text, add_text_overlay
)
from pathlib import Path
from datetime import datetime

import os
import cv2
import numpy as np

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


def save_image(image_bytes: bytes, filename: str) -> str:
    output_dir = Path("processed_images")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"{filename}_{timestamp}.jpg"
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    return str(file_path)


@app.post("/process_video")
async def process_video(file: UploadFile = File(...)):
    try:
        video_bytes = await file.read()
        image = extract_frame_from_video(video_bytes)
        image = preprocess_image(image)

        scene = detect_scene(image)
        emotions = detect_emotions(image)
        style = detect_style(image)

        branding_text = generate_branding_text(scene, style)
        image_with_text_bytes = add_text_overlay(image, branding_text)

        original_filename = Path(file.filename).stem
        saved_path = save_image(image_with_text_bytes, original_filename)

        return FileResponse(path=saved_path, media_type="image/jpeg", filename=Path(saved_path).name)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
