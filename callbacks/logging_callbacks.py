# callbacks/logging_callbacks.py
import logging
from typing import Optional, Dict, Any

# Import ADK types for type hinting
from google.adk.agents.callback_context import CallbackContext # Correct context for agent callbacks
# Removed incorrect InvocationContext import
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext # Correct context for tool callbacks
from google.adk.events import Event

# Get the loggers configured in config.py
agent_flow_logger = logging.getLogger("AgentFlow")
tool_calls_logger = logging.getLogger("ToolCalls") # Re-get logger instance

# --- Agent Callbacks ---
async def log_before_agent(callback_context: CallbackContext, **kwargs) -> None:
    """Logs before an agent's run method starts."""
    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id
    agent_flow_logger.info(f"INVOKE_ID={invocation_id}: ---> Entering Agent '{agent_name}'")

async def log_after_agent(callback_context: CallbackContext, result: Optional[Any], **kwargs) -> None:
    """Logs after an agent's run method finishes."""
    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id
    status = "Success"
    if isinstance(result, Event) and result.error_message:
        status = f"Finished with Error: {result.error_message}"
    elif isinstance(result, Exception):
         status = f"Finished with Exception: {result}"
    # Could potentially inspect state for more detailed status if needed

    agent_flow_logger.info(f"INVOKE_ID={invocation_id}: <--- Exiting Agent '{agent_name}'. Status: {status}")

# --- Tool Callbacks ---
async def log_before_tool(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, **kwargs) -> Optional[Dict]:
    """Logs before a tool is executed."""
    tool_name = tool.name
    agent_name = tool_context.agent_name
    invocation_id = tool_context.invocation_id
    func_call_id = getattr(tool_context, 'function_call_id', 'N/A') # function_call_id is specific to ToolContext

    # Sanitize args for logging (avoid logging huge code strings etc.)
    log_args = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 300: # Truncate long strings
            log_args[k] = v[:300] + '...'
        else:
            log_args[k] = v

    tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Agent '{agent_name}' -> Calling Tool '{tool_name}' (CallID: {func_call_id})")
    tool_calls_logger.debug(f"INVOKE_ID={invocation_id}: Tool '{tool_name}' Args: {log_args}")
    return None # Must return None or a Dict to override

async def log_after_tool(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, result: Optional[Any], **kwargs) -> Optional[Dict]:
    """Logs after a tool finishes execution."""
    tool_name = tool.name
    agent_name = tool_context.agent_name
    invocation_id = tool_context.invocation_id
    func_call_id = getattr(tool_context, 'function_call_id', 'N/A')

    status = "Success"
    log_result_summary = ""

    if isinstance(result, dict):
        status = result.get("status", "Success") # Check for status key convention
        log_result_summary = {k: (v[:200] + '...' if isinstance(v, str) and len(v) > 200 else v) for k, v in result.items()}
        if 'output_files' in log_result_summary:
             log_result_summary['output_files'] = list(log_result_summary['output_files'].keys()) # Log only names
    elif isinstance(result, Exception):
        status = f"Tool Error: {type(result).__name__}"
        log_result_summary = str(result)
    elif result is None:
         status = "Completed (No explicit return)"
         log_result_summary = "None"
    else:
        status = "Completed"
        log_result_summary = str(result)[:200] + ('...' if len(str(result)) > 200 else '')

    tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Agent '{agent_name}' <- Tool '{tool_name}' (CallID: {func_call_id}) Finished. Status: {status}")
    tool_calls_logger.debug(f"INVOKE_ID={invocation_id}: Tool '{tool_name}' Result Summary: {log_result_summary}")

    # Artifact saving logic (remains the same, relies on agent setting state)
    if tool_name == "code_execution_tool" and status == "success":
        plot_info = tool_context.state.get("temp:expected_plot_output")
        if plot_info and isinstance(plot_info, dict) and isinstance(result, dict):
            output_files = result.get("output_files", {})
            plot_logical_name = plot_info.get("logical_name")
            plot_output_key = plot_info.get("output_key", "plot")

            if plot_logical_name and plot_output_key in output_files:
                plot_local_path = output_files[plot_output_key]
                tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Found expected plot output '{plot_output_key}' at '{plot_local_path}'. Attempting artifact save.")
                try:
                    # Ensure helper is imported correctly if needed here
                    from core_tools.artifact_helpers import save_plot_artifact
                    artifact_name = await save_plot_artifact(plot_local_path, plot_logical_name, tool_context)
                    if artifact_name:
                        state_key_base = plot_info.get("state_key_base")
                        if state_key_base:
                            tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Plot artifact '{artifact_name}' saved. Agent should update state key like '{state_key_base}.plots'.")
                        else:
                            tool_calls_logger.warning(f"INVOKE_ID={invocation_id}: Plot artifact saved ('{artifact_name}'), but no state_key_base provided.")
                    # Clear the temporary flag
                    if "temp:expected_plot_output" in tool_context.state:
                         tool_context.state.pop("temp:expected_plot_output") # Use pop for safety
                except ImportError:
                     tool_calls_logger.error(f"INVOKE_ID={invocation_id}: Could not import artifact_helpers in callback.")
                except Exception as e:
                     tool_calls_logger.error(f"INVOKE_ID={invocation_id}: Error saving artifact in callback: {e}")

    return None # Callback should not modify the tool result itself here
