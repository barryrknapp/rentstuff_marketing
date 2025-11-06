from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os


#@tool
def generate_funny_image(tweet_text: str) -> str:
    """Generate a funny, gear-related image based on tweet text using Stable Diffusion."""
    # Load a lightweight Stable Diffusion model (free tier friendly)
#    model_id = "runwayml/stable-diffusion-v1-5"
    model_id = "digiplay/DreamShaper_8"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, dtype=torch.float32)
    pipe = pipe.to("cpu")  # Use CPU (change to "cuda" if you have a GPU)

    # Craft a humorous prompt from the tweet
    hashtags = [ht for ht in tweet_text.split() if ht.startswith("#")]

    prompt = f"A hilarious, clean, high-quality cartoon illustration of someone using rental gear from rentstuff.club, {tweet_text}, {', '.join(hashtags)}, cartoon style, vibrant colors, professional vector art, pixar-like face, correct hands, no text, no words"
    negative_prompt = "blurry, low quality, dark,text, watermark, logo, signature, letters, words, numbers, font, writing"

    # Generate image
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image_path = "generated_image.jpg"
    image.save(image_path)

    return image_path  # Return path to the generated image



generate_funny_image(
    "Need a megaphone for #Election2025 debates? Rent one cheap at https://rentstuff.club! 🎤"
);