# config.py
import os
import logging
import datetime

# --- Model Configuration ---
# Choose the primary model provider ('google', 'openai', 'ollama')
# NOTE: For OpenAI, ensure OPENAI_API_KEY env var is set.
# NOTE: For Ollama, ensure Ollama service is running locally.
PRIMARY_PROVIDER = os.getenv("PRIMARY_PROVIDER", "google") # Default to Google

# --- Google Models (Requires GOOGLE_API_KEY env var) ---
# Ensure the Image Analysis model is multimodal (e.g., Gemini 1.5 Flash/Pro)
GOOGLE_DEFAULT_MODEL = "gemini-1.5-flash-latest"
GOOGLE_IMAGE_ANALYSIS_MODEL = "gemini-1.5-flash-latest" # Or gemini-1.5-pro-latest

# --- OpenAI Models (Requires OPENAI_API_KEY env var) ---
# Requires 'pip install litellm'
OPENAI_DEFAULT_MODEL = "openai/gpt-4o" # LiteLLM format
OPENAI_IMAGE_ANALYSIS_MODEL = "openai/gpt-4-vision-preview" # Example, check LiteLLM/OpenAI docs for current vision models

# --- Ollama Models (Requires Ollama running locally) ---
# Requires 'pip install litellm'
# Example: ollama pull llama3:8b
OLLAMA_DEFAULT_MODEL = "ollama/llama3:8b" # LiteLLM format (replace with your desired Ollama model)
# Ollama might not directly support robust multimodal analysis equivalent to Gemini/GPT-4V easily via LiteLLM yet.
# We'll fallback to the primary provider's image model if Ollama is selected as primary.

# --- Set Active Models based on PRIMARY_PROVIDER ---
if PRIMARY_PROVIDER == "openai":
    print("--- Using OpenAI Models ---")
    ORCHESTRATOR_MODEL = OPENAI_DEFAULT_MODEL
    CODE_GEN_MODEL = OPENAI_DEFAULT_MODEL
    TASK_AGENT_MODEL = OPENAI_DEFAULT_MODEL
    IMAGE_ANALYSIS_MODEL = OPENAI_IMAGE_ANALYSIS_MODEL # Use specific vision model
    USE_LITELLM = True
elif PRIMARY_PROVIDER == "ollama":
    print("--- Using Ollama Models (Image Analysis will use Google) ---")
    ORCHESTRATOR_MODEL = OLLAMA_DEFAULT_MODEL
    CODE_GEN_MODEL = OLLAMA_DEFAULT_MODEL
    TASK_AGENT_MODEL = OLLAMA_DEFAULT_MODEL
    # Fallback for image analysis if Ollama is primary
    IMAGE_ANALYSIS_MODEL = GOOGLE_IMAGE_ANALYSIS_MODEL
    USE_LITELLM = True
else: # Default to Google
    print("--- Using Google Models ---")
    ORCHESTRATOR_MODEL = GOOGLE_DEFAULT_MODEL
    CODE_GEN_MODEL = GOOGLE_DEFAULT_MODEL
    TASK_AGENT_MODEL = GOOGLE_DEFAULT_MODEL
    IMAGE_ANALYSIS_MODEL = GOOGLE_IMAGE_ANALYSIS_MODEL
    USE_LITELLM = False # Google models are handled natively by ADK

# --- Directory Constants ---
WORKSPACE_DIR = "ml_workspace" # For code execution outputs
LOG_DIR = "ml_agent_logs"      # For agent/tool logs

# --- Artifact Constants ---
ARTIFACT_PREFIX = "ml_copilot_artifact_" # Prefix for saved artifacts

# --- Logging Setup ---
LOG_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# --- Code Execution ---
# WARNING: Setting this to True uses a basic, insecure subprocess execution.
# ONLY use for local testing where you trust the model's output.
# Production systems REQUIRE a proper sandboxing solution (Docker, nsjail, etc.).
ALLOW_INSECURE_CODE_EXECUTION = True # SET TO FALSE TO DISABLE THE INSECURE TOOL
CODE_EXECUTION_TIMEOUT = 300 # Seconds

def setup_logging():
    """Configures basic logging and file handlers."""
    os.makedirs(LOG_DIR, exist_ok=True)

    # Configure root logger (console output)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    # General Agent Flow Logger
    agent_log_file = os.path.join(LOG_DIR, f"agent_flow_{datetime.datetime.now().strftime(LOG_TIMESTAMP_FORMAT)}.log")
    agent_file_handler = logging.FileHandler(agent_log_file)
    agent_file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    agent_logger = logging.getLogger("AgentFlow") # Specific logger for agent flow
    agent_logger.setLevel(logging.INFO)
    if not agent_logger.handlers: # Avoid duplicate handlers
        agent_logger.addHandler(agent_file_handler)
    agent_logger.propagate = False # Don't send to root console logger

    # Tool Call Logger
    tool_log_file = os.path.join(LOG_DIR, f"tool_calls_{datetime.datetime.now().strftime(LOG_TIMESTAMP_FORMAT)}.log")
    tool_file_handler = logging.FileHandler(tool_log_file)
    tool_file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    tool_logger = logging.getLogger("ToolCalls") # Specific logger for tool calls
    tool_logger.setLevel(logging.INFO)
    if not tool_logger.handlers: # Avoid duplicate handlers
        tool_logger.addHandler(tool_file_handler)
    tool_logger.propagate = False

    print(f"Logging configured. Agent logs: {agent_log_file}, Tool logs: {tool_log_file}")

# Call setup when this module is imported
setup_logging()

# Get specific loggers easily
agent_flow_logger = logging.getLogger("AgentFlow")
tool_calls_logger = logging.getLogger("ToolCalls")

print("--- Config Loaded ---")
print(f"Primary Provider: {PRIMARY_PROVIDER}")
print(f"Orchestrator Model: {ORCHESTRATOR_MODEL}")
print(f"Code Gen Model: {CODE_GEN_MODEL}")
print(f"Task Agent Model: {TASK_AGENT_MODEL}")
print(f"Image Analysis Model: {IMAGE_ANALYSIS_MODEL}")
print(f"Using LiteLLM: {USE_LITELLM}")
print(f"Allow Insecure Code Execution: {ALLOW_INSECURE_CODE_EXECUTION}")
