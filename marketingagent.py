# marketingagent.py — FIXED: Posts tweets from JSON in content
import os
import sys
import logging
from typing import TypedDict, List
from dotenv import load_dotenv
from pathlib import Path
import json
import re
import unicodedata

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from tools import post_to_twitter  # Only need this one

load_dotenv()
logging.basicConfig(level=logging.INFO)

#LOAD INPUT DATA IF EXISTS

try:
    input_data = json.loads(sys.argv[1])
except Exception as e:
    print(json.dumps({"error": f"Invalid JSON: {e}"}))
    sys.exit(1)

# === BUSINESS INFO ===
company = input_data["company_name"]
domain = input_data["domain"]
website = input_data["website"]
category = input_data["category"]
focus = input_data["focus_keyword"]
creds = input_data["twitter"]

# LLM
llm = ChatOpenAI(
    base_url="http://strategyengine.one/api/localai/v1",
    api_key="ollama",
    model="llama3.1:8b",
    temperature=0.2,
    timeout=300
)

search_tool = DuckDuckGoSearchRun()
tools = [search_tool, post_to_twitter]

# Keep bind_tools for planning, but we'll parse manually
llm_with_tools = llm.bind_tools(tools)

# State
class AgentState(TypedDict):
    messages: List

# Nodes
def chatbot(state: AgentState) -> dict:
    # Default: use original system prompt
    injected_prompt = SYSTEM_PROMPT

    # Extract hashtags from search results
    search_results = ""
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage) and "duckduckgo_search" in msg.name:
            hashtags = re.findall(r'#\w+', msg.content)
            if hashtags:
                search_results = ", ".join(set(hashtags[:5]))
                break

    # Inject trending tags and business info
    if search_results:
        injected_prompt = SYSTEM_PROMPT.replace("${TRENDING_TAGS}", search_results)
    else:
        injected_prompt = SYSTEM_PROMPT.replace("${TRENDING_TAGS}", "general trends")

    # Inject business info
    injected_prompt = injected_prompt \
        .replace("${COMPANY}", company) \
        .replace("${DOMAIN}", domain) \
        .replace("${DOMAIN}", domain) \
        .replace("${WEBSITE}", website) \
        .replace("${CATEGORY}", category) \
        .replace("${FOCUS}", focus)

    # Apply to first message (system prompt)
    state["messages"][0].content = injected_prompt

    # Call LLM
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools=tools)  # Only runs search

# Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", chatbot)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")
app = workflow.compile(checkpointer=MemorySaver())

# System prompt
SYSTEM_PROMPT = Path("system_prompt.txt").read_text(encoding="utf-8").strip()

# Manual tweet posting from JSON in content
def post_tweets_from_content(content: str):
    print("\n=== POSTING TWEETS FROM JSON IN CONTENT ===")
    
    # Normalize Unicode
    import unicodedata
    content = unicodedata.normalize('NFKC', content)
    
    # BETTER REGEX: Match balanced braces
    import re
    pattern = r'(\{"name"\s*:\s*"post_to_twitter"[^}]*\{[^}]*\}[^}]*\})'
    json_blocks = re.findall(pattern, content)
    
    # Fallback: Try original if above fails
    if not json_blocks:
        json_blocks = re.findall(r'\{[^{}]*"name"\s*:\s*"post_to_twitter"[^{}]*\{[^{}]*\}[^{}]*\}', content)
    
    for block in json_blocks:
        try:
            # Clean up common issues
            block = block.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
            block = re.sub(r',\s*}', '}', block)  # Remove trailing commas
            block = re.sub(r'{\s*,', '{', block)  # Remove leading commas
            
            print(f"\nRAW JSON: {block[:200]}...")
            call = json.loads(block)
            tweet = call.get("parameters", {}).get("tweet", "")
            
            if tweet and len(tweet) <= 280:
                print(f"POSTING: {tweet[:60]}...")
                #result = post_to_twitter.run({"tweet": tweet})
                result = post_to_twitter.run({"tweet": tweet, "twitter_creds": creds})
                print(f"SUCCESS: {result}")
            else:
                print(f"SKIPPED: invalid tweet")
                
        except json.JSONDecodeError as e:
            print(f"JSON ERROR: {e}")
            print(f"FAILED BLOCK: {block[:300]}...")
        except Exception as e:
            print(f"POST ERROR: {e}")


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "rentstuff_marketing_001"}}
    initial_input = [
        HumanMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Develop and execute the plan for https://rentstuff.club now.")
    ]
    print("Running marketing agent for "  + company + "...\n")

    full_content = ""

    try:
        for chunk in app.stream({"messages": initial_input}, config, stream_mode="values"):
            msg = chunk["messages"][-1]

            if isinstance(msg, AIMessage):
                if msg.content:
                    print("ASSISTANT:", msg.content.strip())
                    full_content += msg.content  # Accumulate for final JSON parse
                if msg.tool_calls:
                    print("TOOL CALLS (search only):", msg.tool_calls)
                print("\n---\n")

            elif isinstance(msg, ToolMessage):
                print("TOOL RESULT (search):", msg.content)
                print("\n---\n")

        # After loop: post tweets from accumulated content
        post_tweets_from_content(full_content)

    except Exception as e:
        print(f"\nCRASH: {e}")
        import traceback; traceback.print_exc()