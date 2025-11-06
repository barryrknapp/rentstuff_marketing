import logging
from pathlib import Path
from typing import Annotated
from langchain_core.tools import tool
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

log = logging.getLogger(__name__)

@tool
def generate_funny_image(tweet_text: Annotated[str, "Full tweet text (including hashtags)"]) -> str:
    """
    Generate a funny, gear-related cartoon image based on the tweet.
    Returns the absolute path to the saved image file.
    """
    try:
        log.info("Loading Stable Diffusion model...")
        model_id = "digiplay/DreamShaper_8"          # swap for a smaller model if needed
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        pipe = pipe.to("cpu")                       # change to "cuda" if GPU available

        hashtags = [ht for ht in tweet_text.split() if ht.startswith("#")]
        prompt = (
            f"A hilarious, clean, high-quality cartoon illustration of someone using rental gear, "
            f"{tweet_text}, {', '.join(hashtags)}, cartoon style, vibrant colors, "
            f"professional vector art, pixar-like cute face, no text, no words"
        )
        negative_prompt = "blurry, low quality, dark, text, watermark, logo, letters, words"

        log.info("Generating image...")
        image: Image.Image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
        ).images[0]

        # Unique filename per call
        out_path = Path(f"generated_{hash(tweet_text) & 0xFFFFFFFF:08x}.jpg")
        image.save(out_path)
        log.info(f"Image saved to {out_path}")
        return str(out_path.resolve())

    except Exception as e:
        log.error(f"generate_funny_image failed: {e}")
        raise