# config.py
import os
import logging
import datetime

# --- Model Configuration ---
# Replace with your desired model identifiers accessible via GOOGLE_API_KEY
# Ensure the Image Analysis model is multimodal (e.g., Gemini 1.5 Flash/Pro)
ORCHESTRATOR_MODEL = "gemini-1.5-flash-latest"
CODE_GEN_MODEL = "gemini-1.5-flash-latest"
TASK_AGENT_MODEL = "gemini-1.5-flash-latest"
IMAGE_ANALYSIS_MODEL = "gemini-1.5-flash-latest" # Ensure this is a multimodal model

# --- Directory Constants ---
WORKSPACE_DIR = "ml_workspace" # For code execution outputs
LOG_DIR = "ml_agent_logs"      # For agent/tool logs

# --- Artifact Constants ---
ARTIFACT_PREFIX = "ml_copilot_artifact_" # Prefix for saved artifacts

# --- Logging Setup ---
LOG_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

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

