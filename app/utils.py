from transformers import (
    CLIPModel, CLIPProcessor, pipeline,
    ViTForImageClassification, ViTFeatureExtractor, AutoTokenizer, AutoModelForCausalLM
)
from PIL import Image, ImageDraw, ImageFont
import torch
import io
import cv2
import tempfile
import os
import easyocr


# 1. Load Models
def load_clip_model():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor


def load_emotion_model():
    return pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')


def load_scene_model():
    scene_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    scene_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    return scene_model, scene_extractor


def load_style_model():
    style_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    style_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return style_model, style_processor


def load_text_generation_model():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return tokenizer, model


# Initialize models
clip_model, clip_processor = load_clip_model()
emotion_model = load_emotion_model()
scene_model, scene_extractor = load_scene_model()
style_model, style_processor = load_style_model()
tokenizer, text_gen_model = load_text_generation_model()

# Initialize OCR reader
ocr_reader = easyocr.Reader(['en'])


# 2. Extract Frame from Video
def extract_frame_from_video(video_bytes: bytes) -> Image:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(video_bytes)
        temp_video_path = temp_video_file.name

    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise Exception("Error opening video file")
        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to extract frame from video")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        cap.release()
        return image
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


# 3. Preprocess Image
def preprocess_image(image: Image) -> Image:
    return image


# 4. Scene Detection
def detect_scene(image: Image) -> str:
    inputs = scene_extractor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values']
    with torch.no_grad():
        outputs = scene_model(pixel_values)
    predicted_class = outputs.logits.argmax(-1).item()
    scene_labels = ["outdoor", "indoor", "landscape", "cityscape"]
    return scene_labels[predicted_class] if predicted_class < len(scene_labels) else "unknown"


# 5. Emotion Detection using OCR and Sentiment Analysis
def detect_emotions(image: Image) -> dict:
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    extracted_text = ocr_reader.readtext(img_bytes.getvalue(), detail=0)
    text = " ".join(extracted_text)
    if not text.strip():
        return {"anger": 0.0, "joy": 0.0, "fear": 0.0, "sadness": 0.0, "message": "No text found"}

    results = emotion_model([text])
    return {
        "anger": results[0]['score'] if results[0]['label'] == 'Negative' else 0.0,
        "joy": results[0]['score'] if results[0]['label'] == 'Positive' else 0.0,
        "fear": 0.0,
        "sadness": 0.0
    }


# 6. Style Detection
def detect_style(image: Image) -> str:
    inputs = style_processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values']
    with torch.no_grad():
        style_features = style_model.get_image_features(pixel_values=pixel_values)
    return "Artistic" if style_features.mean() > 0.5 else "Realistic"


# 7. AI-Powered Branding Text Generation
def generate_branding_text(scene: str, style: str) -> str:
    prompt = f"Generate branding content for a {scene} scene with {style} style."
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = text_gen_model.generate(inputs['input_ids'], max_length=30, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 8. Add Branding Text to Image
def add_text_overlay(image: Image, text: str) -> bytes:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((10, 10), text, font=font, fill=(255, 255, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
