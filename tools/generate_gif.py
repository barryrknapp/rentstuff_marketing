import logging
from pathlib import Path
from typing import Annotated
from langchain_core.tools import tool
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageSequence
import numpy as np
from io import BytesIO

log = logging.getLogger(__name__)

# === CONFIG ===
MODEL_ID = "digiplay/DreamShaper_8"
HEIGHT = 768
WIDTH = 768
#decrese to 6 frames if this is taking too long
NUM_FRAMES = 10  
FPS = 6
STEPS = 20
GUIDANCE = 7.0
STRENGTH = 0.75  # For img2img transitions (0.0 = no change, 1.0 = full regen)

@tool
def generate_funny_gif(tweet_text: Annotated[str, "Full tweet text (including hashtags)"]) -> str:
    """
    Generate a short, funny, family-friendly animated GIF (cartoon style).
    Returns absolute path to saved GIF file.
    """
    try:
        log.info("Loading Stable Diffusion model on GPU...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(device)

        pipe.enable_attention_slicing()
        pipe.safety_checker = None

        hashtags = [ht for ht in tweet_text.split() if ht.startswith("#")]
        base_prompt = (
            f"A hilarious, clean, high-quality scene related to {tweet_text}, "
            f"{', '.join(hashtags)}, "
            f"professional, family friendly, no text, "
            f"character motion, looping animation, smooth motion"
        )
        negative_prompt = "blurry, low quality, dark, ugly, deformed, text, watermark, logo, words, letters"

        # Seed for reproducibility
        seed = hash(tweet_text) & 0xFFFFFFFF
        generator = torch.Generator(device=device).manual_seed(seed)

        log.info("Generating GIF frames...")
        frames = []
        prev_image = None

        for i in range(NUM_FRAMES):
            log.info(f"  Frame {i+1}/{NUM_FRAMES}")

            # Use previous frame as init_image for smooth motion
            init_image = prev_image.resize((WIDTH, HEIGHT)) if prev_image else None

            image = pipe(
                prompt=base_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                height=HEIGHT,
                width=WIDTH,
                generator=generator.manual_seed(seed + i),  # Slight variation per frame
                strength=STRENGTH if init_image else None,
                init_image=init_image,
            ).images[0]

            # Convert to PIL and optimize
            frame = image.convert("P", palette=Image.ADAPTIVE, colors=256)
            frames.append(frame)
            prev_image = image  # For next frame

        # Save GIF
        out_path = Path(f"generated_gif_{hash(tweet_text) & 0xFFFFFFFF:08x}.gif")
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / FPS),
            loop=0,
            optimize=True,
            disposal=2,
        )

        log.info(f"GIF saved to {out_path.resolve()}")
        return str(out_path.resolve())

    except Exception as e:
        log.error(f"generate_funny_image failed: {e}")
        raise