# tools/facebook_group_tool.py
import json
import logging
from typing import List, Dict, Any, Optional
import facebook
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Pydantic schemas – keep the same as before
# ----------------------------------------------------------------------
class FacebookSearchArgs(BaseModel):
    keyword: str = Field(..., description="Main keyword to look for.")
    max_posts: int = Field(20, description="Maximum recent posts per group.")
    min_likes: int = Field(0, description="Min likes to keep.")
    min_comments: int = Field(0, description="Min comments to keep.")

class FacebookReplyArgs(BaseModel):
    post_id: str = Field(..., description="Full post ID (groupid_postid).")
    message: str = Field(..., description="Reply text (≤ 2000 chars).")

# ----------------------------------------------------------------------
# Helper – create GraphAPI from **dict** (passed by the agent)
# ----------------------------------------------------------------------
def _get_graph(fb_creds: Dict[str, str]) -> facebook.GraphAPI:
    token = fb_creds.get("access_token")
    if not token:
        raise ValueError("Facebook access_token missing in credentials")
    return facebook.GraphAPI(access_token=token, version="19.0")

# ----------------------------------------------------------------------
# TOOL 1 – search_facebook_groups
# ----------------------------------------------------------------------
class FacebookGroupSearchTool(BaseTool):
    name: str = "search_facebook_groups"
    description: str = (
        "Search *your* Facebook groups for recent posts containing a keyword. "
        "Returns a JSON list of interesting posts."
    )
    args_schema = FacebookSearchArgs

    def _run(self,
             keyword: str,
             max_posts: int = 20,
             min_likes: int = 0,
             min_comments: int = 0,
             facebook_creds: Optional[Dict[str, str]] = None) -> str:

        if not facebook_creds:
            raise ValueError("facebook_creds dictionary is required")

        graph = _get_graph(facebook_creds)

        # 1. List groups the token can see
        groups_resp = graph.get_object("me/groups", fields="id,name")
        groups: List[Dict[str, Any]] = groups_resp.get("data", [])
        log.info(f"Found {len(groups)} groups.")

        interesting: List[Dict[str, Any]] = []

        for g in groups:
            gid, gname = g["id"], g["name"]
            try:
                feed = graph.get_object(
                    f"{gid}/feed",
                    fields="message,permalink_url,created_time,"
                           "reactions.summary(total_count),"
                           "comments.summary(total_count)",
                    limit=max_posts
                )
            except facebook.GraphAPIError as e:
                log.warning(f"Skip group {gname} ({gid}): {e}")
                continue

            for post in feed.get("data", []):
                msg = post.get("message") or ""
                if keyword.lower() not in msg.lower():
                    continue

                likes = post.get("reactions", {}).get("summary", {}).get("total_count", 0)
                comments = post.get("comments", {}).get("summary", {}).get("total_count", 0)

                if likes < min_likes or comments < min_comments:
                    continue

                interesting.append({
                    "group_id": gid,
                    "group_name": gname,
                    "post_id": post["id"],
                    "permalink": post.get("permalink_url"),
                    "created": post.get("created_time"),
                    "snippet": msg[:180] + ("…" if len(msg) > 180 else ""),
                    "likes": likes,
                    "comments": comments,
                })

                if len(interesting) >= max_posts:
                    break
            if len(interesting) >= max_posts:
                break

        return json.dumps(interesting, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------
# TOOL 2 – reply_to_facebook_post
# ----------------------------------------------------------------------
class FacebookReplyTool(BaseTool):
    name: str = "reply_to_facebook_post"
    description: str = (
        "Post a comment on a Facebook group post. "
        "Supply full post_id and a short message."
    )
    args_schema = FacebookReplyArgs

    def _run(self,
             post_id: str,
             message: str,
             facebook_creds: Optional[Dict[str, str]] = None) -> str:

        if not facebook_creds:
            raise ValueError("facebook_creds dictionary is required")
        if len(message) > 2000:
            raise ValueError("Message > 2000 chars")

        graph = _get_graph(facebook_creds)
        try:
            resp = graph.put_object(parent_object=post_id,
                                    connection_name="comments",
                                    message=message)
            return json.dumps({"success": True, "comment_id": resp["id"]})
        except facebook.GraphAPIError as e:
            return json.dumps({"success": False, "error": str(e)})

# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------
search_facebook_groups = FacebookGroupSearchTool()
reply_to_facebook_post = FacebookReplyTool()