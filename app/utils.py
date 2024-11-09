from transformers import (
    CLIPModel, CLIPProcessor, pipeline,
    ViTForImageClassification, ViTFeatureExtractor,
    AutoTokenizer, AutoModelForCausalLM
)
from PIL import Image, ImageDraw, ImageFont
import torch
import io
import cv2
import tempfile
import os
import easyocr
import numpy as np


# Load necessary models
def load_clip_model():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32")


def load_emotion_model():
    return pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')


def load_scene_model():
    return ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224"), ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")


def load_style_model():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32")


def add_emoji_to_text_and_size(text, font_size=1):
    # Simple keyword-to-emoji mapping for demonstration
    emoji_dict = {
        "Default": "😮",
        "happy": "😊",
        "sad": "😢",
        "love": "❤️",
        "exciting": "🎉",
        "surprised": "😮",
        "world": "🌍",
        "AI": "🤖",
        "fire": "🔥"
    }

    # Append emojis to the text based on keywords
    for word, emoji in emoji_dict.items():
        if word in text:
            text += f" {emoji}"

    # Adjust the text size display (console formatting)
    formatted_text = f"\033[{font_size}m{text}\033[0m"  # ANSI escape sequence for font size

    return formatted_text


def load_text_generation_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model


# Initialize models
clip_model, clip_processor = load_clip_model()
emotion_model = load_emotion_model()
scene_model, scene_extractor = load_scene_model()
style_model, style_processor = load_style_model()
tokenizer, text_gen_model = load_text_generation_model()
ocr_reader = easyocr.Reader(['en'])


def extract_frames_from_video(video_bytes: bytes, num_frames=8) -> list[Image.Image]:
    """Extract multiple frames evenly distributed throughout the video."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(video_bytes)
        temp_video_path = temp_video_file.name

    try:
        frames = []
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise Exception("Error opening video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


def preprocess_image(image: Image) -> Image:
    """Preprocess image for model input."""
    # Resize if needed
    target_size = (224, 224)
    if image.size != target_size:
        image = image.resize(target_size)
    return image


def detect_scene(image: Image) -> str:
    """Detect scene type from image."""
    inputs = scene_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = scene_model(inputs['pixel_values'])
    predicted_class = outputs.logits.argmax(-1).item()
    scene_labels = ["outdoor", "indoor", "landscape", "cityscape", "portrait", "action", "nature"]
    return scene_labels[predicted_class % len(scene_labels)]


def detect_emotions(image: Image) -> dict:
    """Detect emotions in image text and visual content."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    extracted_text = ocr_reader.readtext(img_bytes.getvalue(), detail=0)
    text = " ".join(extracted_text)

    emotions = {
        "joy": 0.0,
        "excitement": 0.0,
        "neutral": 0.0,
        "calm": 0.0
    }

    if text.strip():
        results = emotion_model([text])
        score = results[0]['score']
        label = results[0]['label']

        if label == 'POSITIVE':
            emotions["joy"] = score
        elif label == 'NEUTRAL':
            emotions["neutral"] = score
        elif label == 'NEGATIVE':
            emotions["calm"] = score

    return emotions


def detect_style(image: Image) -> str:
    """Detect visual style of the image."""
    inputs = style_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        style_features = style_model.get_image_features(pixel_values=inputs['pixel_values'])

    styles = ["cinematic", "artistic", "natural", "professional", "casual"]
    style_idx = torch.argmax(style_features.mean(dim=1)).item()
    return styles[style_idx % len(styles)]


def analyze_frame_quality(image: Image.Image) -> float:
    """Analyze frame quality based on multiple metrics."""
    np_image = np.array(image)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

    # Brightness
    brightness = np.mean(gray) / 255.0

    # Contrast
    contrast = np.std(gray) / 128.0

    # Blur detection
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(blur / 1000.0, 1.0)

    # Edge complexity
    edges = cv2.Canny(gray, 100, 200)
    complexity = np.mean(edges > 0)

    # Combine scores
    quality_score = (
            brightness * 0.25 +
            contrast * 0.25 +
            blur_score * 0.3 +
            complexity * 0.2
    )

    return quality_score


def generate_engaging_text(scene: str, style: str, emotions: dict) -> str:
    """Generate engaging text using GPT-2 model."""
    context_prompt = f"""
Create an engaging title for a {scene} video with {style} style.
Emotional tone: {max(emotions.items(), key=lambda x: x[1])[0]}
Requirements:
- Captivating and memorable
- Relevant to {scene} content
- Matches {style} style
- Short and impactful
Title:"""

    inputs = tokenizer(context_prompt, return_tensors='pt', max_length=100)

    outputs = text_gen_model.generate(
        inputs['input_ids'],
        max_new_tokens=30,  # Specify only the number of new tokens to generate
        num_return_sequences=1,
        temperature=0.9,
        top_p=0.85,
        do_sample=True,
        no_repeat_ngram_size=2,
        top_k=50,
    )

    title = tokenizer.decode(outputs[0], skip_special_tokens=True)
    title = title.replace(context_prompt, "").strip()

    # Clean and format title
    title = title.split('\n')[0][:60]
    return title


def detect_empty_region(image: Image.Image) -> tuple:
    """Detect suitable empty region for text placement."""
    np_image = np.array(image)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

    # Create regions grid
    height, width = gray.shape
    grid_size = 64
    best_region = (width // 2, height // 4)  # Default position

    # Find dark regions suitable for text
    for y in range(0, height - grid_size, grid_size):
        for x in range(0, width - grid_size, grid_size):
            region = gray[y:y + grid_size, x:x + grid_size]
            if np.mean(region) < 128:  # Dark region
                best_region = (x, y)
                break

    return best_region


def add_text_overlay(image: Image, text: str, coordinates: tuple) -> bytes:
    """Add text overlay to image."""
    draw = ImageDraw.Draw(image)

    # Use a default font (you can add custom font support)
    font_size = 30
    font = ImageFont.load_default()

    # Add shadow/outline for better visibility
    shadow_offset = 2
    for offset_x, offset_y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        draw.text(
            (coordinates[0] + offset_x, coordinates[1] + offset_y),
            text,
            font=font,
            fill=(0, 0, 0)  # Black shadow
        )

    # Draw main text
    draw.text(
        coordinates,
        text,
        font=font,
        fill=(255, 255, 255)  # White text
    )

    # Convert to bytes
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def generate_top_thumbnails(video_bytes: bytes) -> list[bytes]:
    """Generate multiple thumbnails from video frames."""
    frames = extract_frames_from_video(video_bytes)

    # Analyze frames
    frame_data = []
    for frame in frames:
        preprocessed = preprocess_image(frame)
        quality_score = analyze_frame_quality(preprocessed)

        frame_info = {
            'frame': frame,
            'score': quality_score,
            'scene': detect_scene(preprocessed),
            'style': detect_style(preprocessed),
            'emotions': detect_emotions(preprocessed)
        }
        frame_data.append(frame_info)

    # Select top frames
    frame_data.sort(key=lambda x: x['score'], reverse=True)
    best_frames = frame_data[:5]

    # Generate thumbnails
    thumbnails = []
    for frame_info in best_frames:
        text = generate_engaging_text(
            frame_info['scene'],
            frame_info['style'],
            frame_info['emotions']
        )

        empty_region = detect_empty_region(frame_info['frame'])
        thumbnail = add_text_overlay(
            frame_info['frame'].copy(),
            text,
            empty_region
        )
        thumbnails.append(thumbnail)

    return thumbnails
