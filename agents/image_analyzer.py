# agents/image_analyzer.py
import logging
import json
import base64
from typing import Optional, Dict, Any, AsyncGenerator

from google.adk.agents import LlmAgent
# Corrected import: Import the 'agent_tool' module
from google.adk.tools import agent_tool
from google.adk.agents.invocation_context import InvocationContext # Corrected path
from google.adk.events import Event
from google.genai import types as genai_types # For Part creation

# Import placeholder for analysis and file reading
from placeholders import analyze_image_placeholder, read_local_file_bytes, get_mime_type
from config import IMAGE_ANALYSIS_MODEL, USE_LITELLM, agent_flow_logger, tool_calls_logger

# Import LiteLLM wrapper if needed
# Note: LiteLLM might have limitations with certain multimodal models/providers.
# Check LiteLLM documentation for compatibility.
if USE_LITELLM:
    try:
        from google.adk.models.lite_llm import LiteLlm
        print("LiteLLM imported successfully for ImageAnalyzer.")
        # If the configured IMAGE_ANALYSIS_MODEL is Google even when USE_LITELLM is true
        # (e.g., Ollama fallback), we don't wrap it.
        if not IMAGE_ANALYSIS_MODEL.startswith("gemini"):
             model_config = LiteLlm(model=IMAGE_ANALYSIS_MODEL)
        else:
             model_config = IMAGE_ANALYSIS_MODEL # Use Google model directly
    except ImportError:
        print("ERROR: LiteLLM specified in config, but 'litellm' package not found. pip install litellm")
        LiteLlm = None
        model_config = IMAGE_ANALYSIS_MODEL # Fallback to Google model string
else:
    LiteLlm = None # Define as None if not used
    model_config = IMAGE_ANALYSIS_MODEL # Use Google model string

# --- Image Analysis Agent Definition ---
image_analysis_agent = LlmAgent(
    name="ImageAnalysisAgent",
    model=model_config, # Use configured model (must be multimodal capable)
    instruction="""
You are an AI assistant specialized in analyzing images, particularly plots related to Machine Learning tasks (like confusion matrices, feature importance plots, distribution plots, learning curves).
You will receive an image input and a question about it.
Analyze the image based on the question and provide a concise textual answer.
Focus on extracting meaningful insights relevant to the ML context.
""",
    description="Analyzes image artifacts (e.g., ML plots) using a multimodal model. Input should include 'artifact_name' and 'question'.",
    # This agent relies on the underlying model's multimodal capabilities.
    # The current simulation passes image bytes via state due to AgentTool limitations.
)

# Override _run_async_impl to handle image input simulation
# In a real ADK setup with native multimodal support for AgentTool or direct model calls, this might be simpler.
async def image_analyzer_run_override(
    self: LlmAgent, ctx: InvocationContext, initial_user_content: Optional[genai_types.Content] = None
) -> AsyncGenerator[Event, None]:
    """
    Overrides the default run to simulate multimodal input handling via state.
    Expects 'image_bytes' (base64 encoded) and 'question' in state keys
    'temp:image_analysis_bytes_b64' and 'temp:image_analysis_question',
    set by the calling agent (e.g., Preprocessor) before invoking this tool.
    """
    agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: Entering ImageAnalysisAgent (Override)")
    image_bytes = None
    question = "Analyze this image." # Default question
    error_msg = None
    analysis_result_text = "Image analysis failed or was not performed."

    # --- Get image and question from state (set by caller) ---
    image_bytes_b64 = ctx.session.state.get("temp:image_analysis_bytes_b64")
    question_from_state = ctx.session.state.get("temp:image_analysis_question")

    if image_bytes_b64:
        try:
            image_bytes = base64.b64decode(image_bytes_b64)
            agent_flow_logger.debug(f"INVOKE_ID={ctx.invocation_id}: Decoded image bytes ({len(image_bytes)}) from state.")
        except Exception as e:
            error_msg = f"Failed to decode image bytes from state: {e}"
            agent_flow_logger.error(f"INVOKE_ID={ctx.invocation_id}: {error_msg}")
    else:
        error_msg = "ImageAnalysisAgent: No image data found in temporary state ('temp:image_analysis_bytes_b64')."
        agent_flow_logger.error(f"INVOKE_ID={ctx.invocation_id}: {error_msg}")

    if question_from_state:
        question = question_from_state
        agent_flow_logger.debug(f"INVOKE_ID={ctx.invocation_id}: Loaded question from state: {question}")
    else:
        # If question wasn't set, add to error or use default
        q_error = "ImageAnalysisAgent: Question not found in temporary state ('temp:image_analysis_question')."
        agent_flow_logger.error(f"INVOKE_ID={ctx.invocation_id}: {q_error}")
        if not error_msg: error_msg = q_error # Prioritize image error


    # Clean up temporary state immediately after reading
    if "temp:image_analysis_bytes_b64" in ctx.session.state:
        ctx.session.state.pop("temp:image_analysis_bytes_b64")
    if "temp:image_analysis_question" in ctx.session.state:
        ctx.session.state.pop("temp:image_analysis_question")


    if error_msg: # If image or question missing/invalid
        yield Event(
            author=self.name, invocation_id=ctx.invocation_id,
            error_message=error_msg,
            content=genai_types.Content(parts=[genai_types.Part(text=error_msg)]),
            turn_complete=True # End this agent's turn on error
        )
        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: <--- Exiting ImageAnalysisAgent (Override) due to input error.")
        return

    # --- Call Placeholder Multimodal Analysis ---
    # In a real implementation, you would construct the multimodal request
    # using the image_bytes and question and send it to the configured model API.
    # For now, we use the placeholder.
    agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: Calling analyze_image_placeholder...")
    try:
        # Using placeholder function for simulation
        analysis_result_text = await analyze_image_placeholder(image_bytes, question)
        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: Placeholder analysis successful.")
    except Exception as e:
        error_msg = f"ImageAnalysisAgent: Error during analysis placeholder: {e}"
        agent_flow_logger.error(f"INVOKE_ID={ctx.invocation_id}: {error_msg}", exc_info=True)
        analysis_result_text = f"Analysis failed: {e}" # Update result text
    # --- End Placeholder Call ---

    # Yield the final result
    yield Event(
        author=self.name,
        invocation_id=ctx.invocation_id,
        content=genai_types.Content(parts=[genai_types.Part(text=analysis_result_text)]),
        turn_complete=True, # Indicate this is the final response for this agent's turn
        error_message=error_msg # Pass error if analysis failed
    )
    agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: <--- Exiting ImageAnalysisAgent (Override)")

# Monkey-patch the agent's run method to use our simulation logic
image_analysis_agent._run_async_impl = image_analyzer_run_override.__get__(image_analysis_agent, LlmAgent)


# --- Wrap as AgentTool ---
try:
    image_analysis_tool = agent_tool.AgentTool(
        agent=image_analysis_agent,
        description=(
            "Use this tool to analyze an image file and answer a question about it. "
            "The calling agent should set 'temp:image_analysis_bytes_b64' and 'temp:image_analysis_question' in session state before invocation."
        )
    )
    print(f"--- ImageAnalysisAgent wrapped as tool: {image_analysis_tool.name} ---")
except Exception as e:
    print(f"ERROR: Could not create image_analysis_tool: {e}")
    image_analysis_tool = None

print(f"--- ImageAnalysisAgent Defined & Patched (Model: {image_analysis_agent.model}) ---")

