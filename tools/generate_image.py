import logging
from pathlib import Path
from typing import Annotated
from langchain_core.tools import tool
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

log = logging.getLogger(__name__)

@tool(description="Generate a funny image from tweet text")
def generate_funny_image(tweet_text: Annotated[str, "Full tweet text (including hashtags)"]) -> str:
    """
    Generate a funny, family-friendly cartoon image based on the tweet text.
    The image is created using a Stable Diffusion model and saved to disk.
    Returns the absolute path to the generated image file.
    """
    try:
        log.info("Loading image generation model...")
        #model_id = "cagliostrolab/animagine-xl-4.0"
        model_id = "dreamlike-art/dreamlike-anime-1.0"


        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,              # required on CPU
            use_safetensors=True,
            safety_checker=None,
            text_encoder_2=None                     # disable T5 encoder to avoid pooled embeds bug
        )

        pipe = pipe.to("cpu")

        # Enhanced patch (covers pooled embeds explicitly)
        original_call = pipe.__call__

        def safe_call(self, *args, **kwargs):
            if "added_cond_kwargs" not in kwargs or kwargs["added_cond_kwargs"] is None:
                kwargs["added_cond_kwargs"] = {}
            # Dummy pooled if still somehow None (SDXL pooled dim = 1280)
            if "pooled_prompt_embeds" in kwargs and kwargs.get("pooled_prompt_embeds") is None:
                kwargs["pooled_prompt_embeds"] = torch.zeros(
                    (1, 1280), device="cpu", dtype=torch.float32
                )
            return original_call(self, *args, **kwargs)

        pipe.__call__ = safe_call.__get__(pipe)

        pipe.enable_attention_slicing()  # safe on CPU, reduces memory spikes

        # Summarize tweet more aggressively to stay under ~60 tokens
        tweet_summary = tweet_text[:80].rsplit(' ', 1)[0] + '...' if len(tweet_text) > 80 else tweet_text
        hashtags = [ht.strip() for ht in tweet_text.split() if ht.startswith("#")]
        hashtags_str = ', '.join(hashtags[:3]) if hashtags else ''

        prompt = (
            f"funny cartoon illustration, cute chibi style, exaggerated expression, "
            f"whimsical family-friendly scene: {tweet_summary} {hashtags_str}, "
            f"vibrant colors, no text, no watermark, "
            f"masterpiece, best quality, highres"
        )
        negative_prompt = (
            "blurry, lowres, bad anatomy, deformed, extra limbs, text, watermark, "
            "logo, signature, ugly, mutated hands, jpeg artifacts"
        )

        log.info(f"Using prompt (len ~{len(prompt.split())} tokens): {prompt[:150]}...")

    except Exception as e:
        log.error(f"Model loading failed: {e}")
        raise

    # Generation
    try:
        log.info("Generating image...")

        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,
            guidance_scale=6.0,
            height=768,
            width=768,
        ).images[0]

    except Exception as e:
        error_str = str(e).lower()
        if "nonetype" in error_str or "truncated" in error_str or "iterable" in error_str:
            log.warning("Conditioning/truncation issue – trying minimal prompt fallback")
            minimal_prompt = f"funny cartoon: {tweet_summary}, cute style, no text"
            image = pipe(
                minimal_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25,
                guidance_scale=6.0,
                height=768,
                width=768,
            ).images[0]
        else:
            log.error(f"Generation failed: {e}")
            raise e

    # Save
    out_path = Path(f"generated_{hash(tweet_text) & 0xFFFFFFFF:08x}.png")
    image.save(out_path)
    log.info(f"Image saved to {out_path}")
    return str(out_path.resolve())