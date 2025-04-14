# callbacks/logging_callbacks.py
import logging
from typing import Optional, Dict, Any

# Import ADK types for type hinting
from google.adk.agents.callback_context import CallbackContext, InvocationContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
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
        # Summarize result for logging, avoid huge outputs
        log_result_summary = {k: (v[:200] + '...' if isinstance(v, str) and len(v) > 200 else v) for k, v in result.items()}
        # Don't log output_files content, just keys/paths if needed
        if 'output_files' in log_result_summary:
             log_result_summary['output_files'] = list(log_result_summary['output_files'].keys()) # Log only names

    elif isinstance(result, Exception):
        status = f"Tool Error: {type(result).__name__}"
        log_result_summary = str(result)
    elif result is None:
         status = "Completed (No explicit return)" # e.g. some callbacks might return None
         log_result_summary = "None"
    else:
        # Handle other types if necessary
        status = "Completed"
        log_result_summary = str(result)[:200] + ('...' if len(str(result)) > 200 else '')


    tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Agent '{agent_name}' <- Tool '{tool_name}' (CallID: {func_call_id}) Finished. Status: {status}")
    tool_calls_logger.debug(f"INVOKE_ID={invocation_id}: Tool '{tool_name}' Result Summary: {log_result_summary}")

    # --- Example: Artifact Saving Logic in Callback ---
    # If the tool was code_execution and state indicates a plot was expected, save it.
    # This requires careful state management by the calling agent.
    if tool_name == "code_execution_tool" and status == "success":
        plot_info = tool_context.state.get("temp:expected_plot_output") # Agent sets this before calling code exec
        if plot_info and isinstance(plot_info, dict) and isinstance(result, dict):
            output_files = result.get("output_files", {})
            plot_logical_name = plot_info.get("logical_name") # e.g., 'confusion_matrix_lr_d1'
            plot_output_key = plot_info.get("output_key", "plot") # Key in output_files dict

            if plot_logical_name and plot_output_key in output_files:
                plot_local_path = output_files[plot_output_key]
                tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Found expected plot output '{plot_output_key}' at '{plot_local_path}'. Attempting artifact save.")

                # Import helper within async function if needed, or ensure it's globally available
                from .artifact_helpers import save_plot_artifact

                artifact_name = await save_plot_artifact(plot_local_path, plot_logical_name, tool_context)

                if artifact_name:
                    # Store the artifact name back in state, perhaps under the model/dataset ID
                    state_key_base = plot_info.get("state_key_base") # e.g., "models.LR_d1_run1"
                    if state_key_base:
                         # Need a way to safely update nested state dicts
                         # This is complex to do reliably here. The agent might be better suited
                         # to update its own state after receiving the tool result.
                         # For now, just log it.
                         tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Plot artifact '{artifact_name}' saved. Agent should update state key like '{state_key_base}.plots'.")
                         # Example of how agent might update state later:
                         # current_plots = tool_context.state.get(f'{state_key_base}.plots', [])
                         # current_plots.append(artifact_name)
                         # tool_context.state[f'{state_key_base}.plots'] = current_plots
                    else:
                         tool_calls_logger.warning(f"INVOKE_ID={invocation_id}: Plot artifact saved ('{artifact_name}'), but no state_key_base provided in temp:expected_plot_output to link it.")

                # Clear the temporary flag
                tool_context.state["temp:expected_plot_output"] = None


    return None # Callback should not modify the tool result itself here

