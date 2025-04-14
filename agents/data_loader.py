# agents/data_loader.py
import logging
import json
import os
from typing import Optional, Dict, Any, AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from config import TASK_AGENT_MODEL, WORKSPACE_DIR, agent_flow_logger
# Assuming tools are passed during Orchestrator setup or globally available
# from core_tools import code_execution_tool, logging_tool
# from agents import code_generator_tool # AgentTool

class DataLoadingAgent(LlmAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="DataLoadingAgent",
            model=TASK_AGENT_MODEL,
            instruction="""
Your task is to manage the loading of a specified dataset.
1. You will receive the dataset identifier (e.g., 'd1') and potentially a path or source description via state.
2. Retrieve the source path from state using the identifier (e.g., state['datasets']['d1']['raw_path_source']).
3. Formulate a detailed prompt for the 'CodeGeneratorAgent' to write Python code (using pandas) to load the data from the source path. The code MUST save the loaded DataFrame to a CSV file in the WORKSPACE directory (e.g., '{WORKSPACE_DIR}/loaded_data_d1.csv'). The code MUST print the exact path of the saved file using the convention: `print(f"SAVED_OUTPUT: loaded_data=/path/to/saved/file.csv")`. Include error handling (e.g., try-except for file reading).
4. Call the 'CodeGeneratorAgent' tool with this prompt.
5. Receive the generated code string.
6. Call the 'code_execution_tool' with the generated code.
7. Check the result status from the code execution tool. If it failed, log the error using 'logging_tool' (key 'data_loader_log') and yield an error event. # TODO: Implement retry logic if desired.
8. If successful, parse the 'output_files' dictionary from the execution result to get the absolute local path of the 'loaded_data'.
9. Update the state for the dataset identifier: set `state['datasets'][dataset_id]['raw_data_path'] = '/absolute/path/to/local/data.csv'` and set `state['datasets'][dataset_id]['load_status'] = 'success'`. Use EventActions state_delta for this.
10. Use the 'logging_tool' to log start, code generation request, execution attempt, and final status for this dataset ID (key 'data_loader_log').
11. Yield a final event indicating success or failure for this stage, including the state_delta.
""",
            description="Loads a dataset using generated code and updates state with its local path.",
            **kwargs # Pass tools, callbacks etc. from Orchestrator
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: ---> Entering {self.name}")
        final_status = "Failure"
        error_message = None
        state_delta = {}

        # 1 & 2: Get dataset ID and source path from state (set by Orchestrator)
        dataset_id = ctx.session.state.get("current_dataset_id", "d1") # Assume Orchestrator sets this
        dataset_info = ctx.session.state.get("datasets", {}).get(dataset_id, {})
        source_path = dataset_info.get("raw_path_source")

        await self._log_message(f"Starting data loading for dataset ID: {dataset_id}, Source: {source_path}", ctx)

        if not source_path:
            error_message = f"Source path not found in state for dataset ID '{dataset_id}'."
            await self._log_message(error_message, ctx, level="ERROR")
            yield self._create_final_event(ctx, final_status, error_message)
            return

        # 3. Formulate prompt for Code Generator
        output_filename = f"loaded_data_{dataset_id}.csv"
        # Ensure output path is within the workspace
        absolute_output_path = os.path.abspath(os.path.join(WORKSPACE_DIR, output_filename))
        # Create workspace if it doesn't exist
        os.makedirs(WORKSPACE_DIR, exist_ok=True)

        code_gen_prompt = f"""
Write Python code using pandas to load data from the source: '{source_path}'.
Handle potential errors during loading (e.g., FileNotFoundError).
Save the loaded DataFrame to a CSV file at the following absolute path: '{absolute_output_path}'. Create the directory if it doesn't exist.
After saving successfully, print the exact path using the convention: `print(f"SAVED_OUTPUT: loaded_data={absolute_output_path}")`
Also print basic info: `print(f"INFO: Loaded dataframe shape: {{df.shape}}")`
Code:
"""
        await self._log_message(f"Prompting CodeGeneratorAgent for data loading code.", ctx)

        # 4. Call CodeGeneratorAgent (as a tool)
        code_gen_tool = self.tools_map.get("CodeGeneratorAgent") # Tool name matches agent name by default
        generated_code = None
        if code_gen_tool:
            try:
                # Call the agent tool - this yields events internally
                async for event in code_gen_tool.run_async(ctx, user_content=genai_types.Content(parts=[genai_types.Part(text=code_gen_prompt)])):
                    if event.is_final_response() and event.content and event.content.parts:
                        generated_code = event.content.parts[0].text
                        break # Expecting a single code response
                if generated_code:
                     # Clean up potential markdown backticks
                     generated_code = generated_code.strip().strip('`').strip()
                     if generated_code.startswith('python'):
                          generated_code = generated_code[len('python'):].strip()
                     await self._log_message(f"Received generated code from CodeGeneratorAgent.", ctx)
                else:
                     error_message = "CodeGeneratorAgent did not return code."
                     await self._log_message(error_message, ctx, level="ERROR")

            except Exception as e:
                error_message = f"Error calling CodeGeneratorAgent: {e}"
                await self._log_message(error_message, ctx, level="ERROR")
        else:
            error_message = "CodeGeneratorAgent tool not found."
            await self._log_message(error_message, ctx, level="ERROR")

        # 5 & 6. Call Code Execution Tool
        execution_result = None
        if generated_code and not error_message:
            exec_tool = self.tools_map.get("code_execution_tool")
            if exec_tool:
                try:
                    # Call the tool directly (it's a FunctionTool)
                    execution_result = await exec_tool.func(code_string=generated_code, tool_context=ctx) # Pass context
                    await self._log_message(f"Code execution attempted. Status: {execution_result.get('status')}", ctx)
                except Exception as e:
                    error_message = f"Error calling code_execution_tool: {e}"
                    await self._log_message(error_message, ctx, level="ERROR")
            else:
                error_message = "code_execution_tool not found."
                await self._log_message(error_message, ctx, level="ERROR")

        # 7 & 8. Check result and parse output
        loaded_data_path = None
        if execution_result and execution_result.get("status") == "success":
            output_files = execution_result.get("output_files", {})
            loaded_data_path = output_files.get("loaded_data") # Name matches SAVED_OUTPUT convention
            if not loaded_data_path:
                 error_message = "Code executed successfully but did not report the expected 'loaded_data' output file path via SAVED_OUTPUT convention."
                 await self._log_message(error_message, ctx, level="WARNING")
                 # Attempt to find *any* csv in the output as a fallback
                 for name, path in output_files.items():
                      if path.endswith(".csv"):
                           loaded_data_path = path
                           await self._log_message(f"Fallback: Found output CSV: {path}", ctx, level="WARNING")
                           break
                 if not loaded_data_path:
                      await self._log_message("No CSV found in output files.", ctx, level="ERROR")

            elif not os.path.exists(loaded_data_path):
                 error_message = f"Code reported saving data to '{loaded_data_path}', but the file does not exist."
                 await self._log_message(error_message, ctx, level="ERROR")
                 loaded_data_path = None # Invalidate path

        elif execution_result: # Execution failed
            error_message = f"Code execution failed. Stderr: {execution_result.get('stderr', 'N/A')}"
            # TODO: Implement retry logic here if desired. Could involve:
            # 1. Incrementing a retry counter in state.
            # 2. Re-prompting CodeGeneratorAgent with the error message.
            # 3. Calling code_execution_tool again.
            # 4. Yielding a specific event to Orchestrator to handle retry strategy.
            await self._log_message(f"Code execution failed. Stderr: {execution_result.get('stderr')}", ctx, level="ERROR")

        # 9. Update state if successful
        if loaded_data_path:
            final_status = "Success"
            # Prepare state update using nested structure
            state_delta = {
                f"datasets.{dataset_id}.raw_data_path": loaded_data_path,
                f"datasets.{dataset_id}.load_status": "success"
            }
            await self._log_message(f"Data loading successful. Path: {loaded_data_path}", ctx)
        else:
            final_status = "Failure"
            state_delta = {
                f"datasets.{dataset_id}.load_status": "failure",
                f"datasets.{dataset_id}.error": error_message or "Unknown loading error."
            }

        # 10. Log final status handled above

        # 11. Yield final event
        yield self._create_final_event(ctx, final_status, error_message, state_delta)
        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: <--- Exiting {self.name}. Status: {final_status}")


    async def _log_message(self, message: str, ctx: InvocationContext, level: str = "INFO"):
        """Helper to log messages using the logging_tool."""
        log_tool = self.tools_map.get("logging_tool")
        if log_tool:
            try:
                # Logging tool is FunctionTool, call its func directly
                await log_tool.func(message=message, log_file_key="data_loader_log", tool_context=ctx, level=level)
            except Exception as e:
                agent_flow_logger.error(f"INVOKE_ID={ctx.invocation_id}: Failed to log message via tool: {e}")
                print(f"ERROR logging: {message}") # Fallback print
        else:
            agent_flow_logger.warning(f"INVOKE_ID={ctx.invocation_id}: logging_tool not found for agent {self.name}.")
            print(f"LOG ({self.name}): {message}") # Fallback print

    def _create_final_event(self, ctx: InvocationContext, status: str, error_msg: Optional[str] = None, state_delta: Optional[Dict] = None) -> Event:
         """Helper to create the final event for this agent."""
         message = f"{self.name} finished with status: {status}."
         if error_msg and status != "Success":
             message += f" Error: {error_msg}"

         return Event(
             author=self.name,
             invocation_id=ctx.invocation_id,
             content=genai_types.Content(parts=[genai_types.Part(text=message)]),
             actions=EventActions(state_delta=state_delta) if state_delta else None,
             turn_complete=True,
             error_message=error_msg if status != "Success" else None
         )

