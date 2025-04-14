# agents/image_analyzer.py
import logging
from typing import Optional, Dict, Any, AsyncGenerator

from google.adk.agents import LlmAgent, AgentTool
from google.adk.agents.callback_context import InvocationContext
from google.adk.events import Event
from google.genai import types as genai_types # For Part creation

# Import placeholder for analysis and file reading
from placeholders import analyze_image_placeholder, read_local_file_bytes, get_mime_type
from config import IMAGE_ANALYSIS_MODEL, agent_flow_logger, tool_calls_logger

# --- Image Analysis Agent Definition ---
image_analysis_agent = LlmAgent(
    name="ImageAnalysisAgent",
    model=IMAGE_ANALYSIS_MODEL, # Must be a multimodal model identifier
    instruction="""
You are an AI assistant specialized in analyzing images, particularly plots related to Machine Learning tasks (like confusion matrices, feature importance plots, distribution plots, learning curves).
You will receive an image input and a question about it.
Analyze the image based on the question and provide a concise textual answer.
Focus on extracting meaningful insights relevant to the ML context.
""",
    description="Analyzes image artifacts (e.g., ML plots) using a multimodal model. Input should include 'artifact_name' and 'question'.",
    # This agent implicitly uses the multimodal capability of its model.
    # It needs access to load_artifact, which happens *before* the agent is called,
    # with the image data passed in the context or initial message.
    # We'll simulate this by having the CALLER load the artifact and pass the bytes.
)

# Override _run_async_impl to handle image input simulation
# In a real ADK setup with native multimodal support, this might be simpler.
async def image_analyzer_run_override(
    self: LlmAgent, ctx: InvocationContext
) -> AsyncGenerator[Event, None]:
    """
    Overrides the default run to simulate multimodal input handling.
    Expects 'image_bytes' and 'question' in the initial user_content parts
    or passed via state by the orchestrator/caller.
    """
    agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: Entering ImageAnalysisAgent (Override)")
    image_bytes = None
    question = "Analyze this image." # Default question

    # Try to get image bytes and question from the context/triggering message
    # This simulates how a multimodal call might be structured
    if ctx.user_content and ctx.user_content.parts:
        for part in ctx.user_content.parts:
            if part.inline_data and part.inline_data.data:
                image_bytes = part.inline_data.data
                print(f"--- Image Analyzer: Found image bytes ({len(image_bytes)}) in input content ---")
            elif part.text:
                # Assume the text part contains the question or JSON payload
                try:
                    # Check if text is JSON containing question
                    payload = json.loads(part.text)
                    question = payload.get("question", question)
                    # Could also load artifact here if name was passed in payload
                except json.JSONDecodeError:
                    # Assume text is the question directly
                    question = part.text
                print(f"--- Image Analyzer: Found question: {question} ---")

    # Fallback to check state if not in direct input (set by caller)
    if not image_bytes:
         image_bytes_b64 = ctx.session.state.get("temp:image_analysis_bytes_b64")
         if image_bytes_b64:
             image_bytes = base64.b64decode(image_bytes_b64)
             print(f"--- Image Analyzer: Loaded image bytes ({len(image_bytes)}) from state ---")
             ctx.session.state["temp:image_analysis_bytes_b64"] = None # Clear temp state

    if "temp:image_analysis_question" in ctx.session.state:
        question = ctx.session.state.get("temp:image_analysis_question", question)
        print(f"--- Image Analyzer: Loaded question from state: {question} ---")
        ctx.session.state["temp:image_analysis_question"] = None # Clear temp state


    if not image_bytes:
        error_msg = "ImageAnalysisAgent: No image data provided in input content or state."
        agent_flow_logger.error(f"INVOKE_ID={ctx.invocation_id}: {error_msg}")
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            error_message=error_msg,
            content=genai_types.Content(parts=[genai_types.Part(text=error_msg)])
        )
        ctx.end_invocation = True
        return

    # --- Call Placeholder Multimodal Analysis ---
    # Replace with actual API call to Gemini multimodal endpoint
    try:
        analysis_result_text = await analyze_image_placeholder(image_bytes, question)
    except Exception as e:
        error_msg = f"ImageAnalysisAgent: Error during analysis placeholder: {e}"
        agent_flow_logger.error(f"INVOKE_ID={ctx.invocation_id}: {error_msg}")
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            error_message=error_msg,
            content=genai_types.Content(parts=[genai_types.Part(text=error_msg)])
        )
        ctx.end_invocation = True
        return
    # --- End Placeholder Call ---

    agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: Image analysis successful.")
    # Yield the final result
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=genai_types.Content(parts=[genai_types.Part(text=analysis_result_text)]),
        turn_complete=True, # Indicate this is the final response for this agent's turn
    )
    agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: <--- Exiting ImageAnalysisAgent (Override)")

# Monkey-patch the agent's run method
image_analysis_agent._run_async_impl = image_analyzer_run_override.__get__(image_analysis_agent, LlmAgent)


# --- Wrap as AgentTool ---
# The caller needs to load the artifact bytes and set them (and the question)
# in state ('temp:image_analysis_bytes_b64', 'temp:image_analysis_question')
# before calling this tool.
try:
    image_analysis_tool = AgentTool(
        agent=image_analysis_agent,
        description="Use this tool to analyze an image artifact (e.g., a saved plot) and answer a question about it. IMPORTANT: Before calling, load the artifact's bytes, base64 encode them, and set 'temp:image_analysis_bytes_b64' in state. Also set the 'temp:image_analysis_question' in state.",
    )
except Exception as e:
    print(f"ERROR: Failed to wrap ImageAnalysisAgent as AgentTool: {e}")
    image_analysis_tool = None

