import os
import logging
from pathlib import Path
from typing import Annotated
from langchain_core.tools import tool
from typing import Dict, List
import tweepy


log = logging.getLogger(__name__)

@tool
def post_to_twitter(tweet: str, twitter_creds: Dict[str, str]) -> str:
#def post_to_twitter(tweet: Annotated[str, "Tweet text (≤280 chars)"]) -> str:
    """
    Generates a funny image from the tweet and posts it to X with the image attached.
    """
    if len(tweet) > 280:
        return "Error: Tweet too long (>280 chars)"

    # === 1. Generate Image ===

    try:
        from .generate_image import generate_funny_image  #generate_funny_image
        log.info("Generating image for tweet...")
        image_path =generate_funny_image.run({"tweet_text": tweet}) # generate_funny_image.run({"tweet_text": tweet})
        log.info(f"Image generated: {image_path}")
    except Exception as e:
        log.error(f"Image generation failed: {e}")
        return f"Image failed: {e}. Tweet not posted."

    if not Path(image_path).is_file():
        return "Image file not found. Tweet not posted."

    # === 2. Upload & Post ===
    try:
        # v1.1 API for media upload
        auth = tweepy.OAuth1UserHandler(
            twitter_creds["api_key"],
            twitter_creds["api_secret"],
            twitter_creds["access_token"],
            twitter_creds["access_token_secret"],
        )
        api = tweepy.API(auth)
        media = api.media_upload(image_path)
        log.info(f"Media uploaded: {media.media_id}")

        client = tweepy.Client(
            bearer_token=twitter_creds["bearer_token"],
            consumer_key=twitter_creds["api_key"],
            consumer_secret=twitter_creds["api_secret"],
            access_token=twitter_creds["access_token"],
            access_token_secret=twitter_creds["access_token_secret"],
        )
        response = client.create_tweet(text=tweet, media_ids=[media.media_id])
        tweet_id = response.data["id"]

        # Cleanup
        try:
            os.remove(image_path)
            log.info("Temp image deleted.")
        except:
            pass

        return f"Posted with image! ID: {tweet_id}"

    except Exception as e:
        log.error(f"Twitter post failed: {e}")
        return f"Post failed: {e}"