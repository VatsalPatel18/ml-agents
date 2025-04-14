# core_tools/logging_tool.py
import logging
import os
import datetime
from typing import Dict, Any

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from config import LOG_DIR, LOG_TIMESTAMP_FORMAT, tool_calls_logger # Import shared config

@FunctionTool
def logging_tool(
    message: str,
    log_file_key: str, # e.g., 'preprocessing_log', 'orchestrator_log'
    tool_context: ToolContext,
    level: str = "INFO",
) -> Dict[str, Any]:
    """
    Appends a timestamped message to a specific log file.
    The log file path is managed via session state under 'system:log_paths'.

    Args:
        message: The message to log.
        log_file_key: Key used to identify the log file in state.
        tool_context: ADK ToolContext.
        level: Logging level ('INFO', 'WARNING', 'ERROR', 'DEBUG').

    Returns:
        Dict indicating success or failure and the log file path used.
    """
    invocation_id = tool_context.invocation_id
    agent_name = tool_context.agent_name
    tool_calls_logger.debug(f"INVOKE_ID={invocation_id}: Agent '{agent_name}' requested logging to '{log_file_key}'.")

    log_file_path_state_key = f"system:log_paths:{log_file_key}"
    log_file_path = tool_context.state.get(log_file_path_state_key)

    if not log_file_path:
        # Create a new log file path if it doesn't exist in state
        log_filename = f"{log_file_key}_{datetime.datetime.now().strftime(LOG_TIMESTAMP_FORMAT)}.log"
        # Ensure path is relative for state, but resolve for writing
        relative_log_path = os.path.join(LOG_DIR, log_filename)
        log_file_path = os.path.abspath(relative_log_path)

        # Store the absolute path in state for consistency during the session
        tool_context.state[log_file_path_state_key] = log_file_path
        tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Initialized log file path '{log_file_path}' for key '{log_file_key}' in state.")
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    else:
        # Ensure the path retrieved from state is absolute for writing
        log_file_path = os.path.abspath(log_file_path)


    timestamp = datetime.datetime.now().isoformat()
    log_entry = f"{timestamp} - {level.upper()} - InvID:{invocation_id} - Agent:{agent_name} - {message}\n"

    try:
        # Ensure directory exists (in case state was transferred but dir doesn't exist)
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
        tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Logged to {log_file_key}: {message}")
        return {"status": "success", "log_file": log_file_path}
    except Exception as e:
        tool_calls_logger.error(f"INVOKE_ID={invocation_id}: Failed to write to log file {log_file_path}: {e}")
        # Attempt to log error to the main tool log
        print(f"ERROR: Failed to write application log to {log_file_path}: {e}")
        return {"status": "error", "message": f"Failed to write log: {e}", "log_file": log_file_path}

