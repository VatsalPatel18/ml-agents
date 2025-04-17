# core_tools/human_approval_tool.py
import logging
from typing import Dict, Any, Optional

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext

from config import tool_calls_logger # Use shared logger

@FunctionTool
def human_approval_tool(
    prompt: str,
    options: Optional[list[str]] = None,
    tool_context: ToolContext = None # Context needed for logging
) -> Dict[str, Any]:
    """
    Pauses the workflow and requests human input from the console.

    Args:
        prompt: The message or question to display to the human user.
        options: Optional list of allowed responses. If provided, input validation is performed.
        tool_context: ADK ToolContext.

    Returns:
        Dict with 'status' ('success'/'error') and 'response' (the user's input).
    """
    invocation_id = tool_context.invocation_id if tool_context else "N/A"
    agent_name = tool_context.agent_name if tool_context else "N/A"
    tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Agent '{agent_name}' requesting human input.")

    print("\n" + "="*20 + " HUMAN INPUT REQUIRED " + "="*20)
    print(f"Agent '{agent_name}' needs your input:")
    print(f"\n{prompt}\n")

    if options:
        print(f"Please respond with one of the following: {', '.join(options)}")

    while True:
        try:
            user_response = input("Your response: ").strip()
            if not options: # No validation needed
                break
            elif user_response.lower() in [opt.lower() for opt in options]: # Case-insensitive check
                # Find the original casing of the option for consistency
                matched_option = next((opt for opt in options if opt.lower() == user_response.lower()), user_response)
                user_response = matched_option
                break
            else:
                print(f"Invalid input. Please choose from: {', '.join(options)}")
        except EOFError:
            # Handle cases where input stream is closed (e.g., running non-interactively)
            tool_calls_logger.error(f"INVOKE_ID={invocation_id}: EOFError received while waiting for human input. Cannot proceed.")
            print("\nERROR: Input stream closed. Cannot get human response.")
            print("="*62 + "\n")
            return {"status": "error", "response": None, "error_message": "Input stream closed (EOFError)."}
        except KeyboardInterrupt:
            tool_calls_logger.warning(f"INVOKE_ID={invocation_id}: KeyboardInterrupt received during human input.")
            print("\nOperation cancelled by user (KeyboardInterrupt).")
            print("="*62 + "\n")
            return {"status": "error", "response": None, "error_message": "Operation cancelled by user (KeyboardInterrupt)."}

    tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Received human response: '{user_response}'")
    print("="*62 + "\n") # Separator after input

    return {"status": "success", "response": user_response}

print("--- human_approval_tool defined ---")

