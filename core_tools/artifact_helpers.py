# core_tools/artifact_helpers.py
import logging
import time
import os
from typing import Optional, Tuple

from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

# Import placeholder for reading file bytes
from placeholders import read_local_file_bytes, get_mime_type
from config import ARTIFACT_PREFIX, tool_calls_logger # Use shared prefix

async def save_plot_artifact(
    plot_local_path: str,
    logical_plot_name: str, # e.g., "confusion_matrix_lr_d1"
    tool_context: ToolContext
) -> Optional[str]:
    """
    Reads a plot file from a local path, saves it as an ADK artifact,
    and returns the artifact name.

    Args:
        plot_local_path: Absolute path to the locally saved plot file.
        logical_plot_name: A descriptive name for the plot.
        tool_context: The ADK ToolContext for saving the artifact.

    Returns:
        The generated artifact name if successful, otherwise None.
    """
    invocation_id = tool_context.invocation_id
    tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Attempting to save plot '{logical_plot_name}' from path '{plot_local_path}' as artifact.")

    plot_bytes = read_local_file_bytes(plot_local_path)

    if plot_bytes:
        try:
            mime_type = get_mime_type(plot_local_path)
            artifact_part = genai_types.Part.from_data(data=plot_bytes, mime_type=mime_type)

            # Generate a unique artifact name
            file_extension = os.path.splitext(plot_local_path)[1]
            artifact_name = f"{ARTIFACT_PREFIX}{logical_plot_name}_{int(time.time())}{file_extension}"

            # --- Save using ToolContext ---
            # Note: save_artifact is synchronous in the current ADK context API
            # If ADK context methods become async, this should be awaited.
            version = tool_context.save_artifact(filename=artifact_name, artifact=artifact_part)

            tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Successfully saved artifact '{artifact_name}' (version {version}).")
            print(f"--- Artifact Helper: Saved plot artifact: {artifact_name} (v{version}) ---")

            # Optionally clean up the local file after saving to artifact store
            try:
                os.remove(plot_local_path)
                tool_calls_logger.info(f"INVOKE_ID={invocation_id}: Cleaned up local plot file: {plot_local_path}")
            except Exception as e:
                 tool_calls_logger.warning(f"INVOKE_ID={invocation_id}: Failed to clean up local plot file {plot_local_path}: {e}")


            return artifact_name
        except ValueError as e:
            # Likely ArtifactService not configured
            tool_calls_logger.error(f"INVOKE_ID={invocation_id}: Failed to save artifact '{logical_plot_name}'. Is ArtifactService configured? Error: {e}")
            print(f"ERROR: Failed to save artifact '{logical_plot_name}'. ArtifactService might not be configured.")
        except Exception as e:
            tool_calls_logger.error(f"INVOKE_ID={invocation_id}: Unexpected error saving artifact '{logical_plot_name}': {e}")
            print(f"ERROR: Unexpected error saving artifact '{logical_plot_name}': {e}")
    else:
        tool_calls_logger.error(f"INVOKE_ID={invocation_id}: Failed to read plot bytes from '{plot_local_path}' for artifact saving.")
        print(f"ERROR: Failed to read plot file bytes from '{plot_local_path}'.")

    return None

