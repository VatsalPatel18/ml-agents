# agents/preprocessor.py
import logging
import json
import os
import time
import base64 # Needed if passing image bytes via state for analysis
from typing import Optional, Dict, Any, AsyncGenerator, List

from google.adk.agents import LlmAgent
from pydantic import Field
from google.adk.agents.invocation_context import InvocationContext # Corrected path
from google.adk.events import Event, EventActions
from google.genai import types as genai_types # For Part creation

from config import (
    TASK_AGENT_MODEL,
    USE_LITELLM, # Check if LiteLLM should be used
    WORKSPACE_DIR,
    ARTIFACT_PREFIX,
    agent_flow_logger,
    tool_calls_logger,
)

# Import LiteLLM wrapper if needed
if USE_LITELLM:
    try:
        from google.adk.models.lite_llm import LiteLlm
        print("LiteLLM imported successfully for Preprocessor.")
    except ImportError:
        print("ERROR: LiteLLM specified in config, but 'litellm' package not found. pip install litellm")
        LiteLlm = None
else:
    LiteLlm = None # Define as None if not used


# Import tools and helpers - assuming they are accessible
# In a real package structure, use relative imports:
# from ..core_tools import code_execution_tool, logging_tool, save_plot_artifact, human_approval_tool
# from .code_generator import code_generator_tool # AgentTool
# from .image_analyzer import image_analysis_tool # AgentTool

class PreprocessingAgent(LlmAgent):
    # Mapping of tool names to their Tool instances (injected post-init)
    tools_map: Dict[str, Any] = Field(default_factory=dict)
    def __init__(self, **kwargs):
        # Determine model configuration
        model_config = LiteLlm(model=TASK_AGENT_MODEL) if USE_LITELLM and LiteLlm else TASK_AGENT_MODEL

        super().__init__(
            name="PreprocessingAgent",
            model=model_config, # Use configured model
            instruction="""
Your task is to manage the preprocessing of a loaded dataset based on a defined strategy.
1. You will receive the dataset identifier (e.g., 'd1') via state (`current_dataset_id`).
2. Retrieve the current data path (usually `raw_data_path`) and the preprocessing strategy (e.g., `preprocess_strategy` dictionary) from the state for this dataset ID (e.g., `state['datasets'][dataset_id]`).
3. Formulate a detailed prompt for the 'CodeGeneratorAgent' to write Python code (using pandas, scikit-learn) to perform the specified preprocessing steps (imputation, encoding, scaling, etc.) on the input data path. The code MUST save the processed data to a new CSV file in the WORKSPACE directory (e.g., '{WORKSPACE_DIR}/processed_data_d1.csv'). The code MUST print the exact path of the saved file using the convention: `print(f"SAVED_OUTPUT: processed_data=/path/to/saved/processed_file.csv")`. It should also print a summary of steps applied like `print(f"INFO: Applied mean imputation to [col1], scaling to [col2]")`. Include error handling.
4. Call the 'CodeGeneratorAgent' tool with this prompt.
5. Receive the generated code string.
6. Call the 'code_execution_tool' with the generated code.
7. Check the result status. If error, analyze stderr, potentially re-prompt CodeGeneratorAgent with error context and retry (basic retry placeholder). Log errors using 'logging_tool' (key 'preprocessor_log'). Yield an error event if retries fail.
8. If successful, parse 'output_files' for the 'processed_data' path. Parse stdout for applied 'INFO' steps.
9. Update the state for the dataset ID: set 'processed_data_path', update 'preprocess_steps' list, set 'preprocess_status' to 'success'. Use EventActions state_delta.
10. Check state if visualization is requested (`visualize_after_preprocess`). If yes:
    a. (Optional HITL): Consider calling `human_approval_tool` to ask user if they want to generate plots now.
    b. Formulate prompt for 'CodeGeneratorAgent' for plotting code (e.g., feature distributions, correlation matrix using matplotlib/seaborn), saving plot to a file (e.g., 'preprocess_plot_d1.png' in WORKSPACE_DIR). Code MUST print 'SAVED_OUTPUT: plot=/path/to/plot.png'.
    c. Call 'CodeGeneratorAgent'.
    d. Call 'code_execution_tool' with plotting code.
    e. If successful, get plot path from 'output_files'. Use the `save_plot_artifact` helper function (requires importing it) to read the local plot file and save it as an ADK artifact.
    f. Update state: add the returned artifact name to `state['datasets'][dataset_id]['plots']`.
    g. If image analysis is requested (`analyze_plots` flag in state): Call 'ImageAnalysisAgent' tool (passing artifact name and question via state). Store analysis result in state.
11. Use 'logging_tool' to log progress and status (key 'preprocessor_log').
12. Yield a final event indicating success/failure for this stage, including the state_delta.
""",
            description="Preprocesses data using generated code, optionally visualizes/analyzes plots, and updates state.",
            **kwargs # Pass tools, callbacks etc.
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        code_execution_tool = self.tools_map.get("code_execution_tool")
        logging_tool_func = self.tools_map.get("logging_tool").func if self.tools_map.get("logging_tool") else None # Get the underlying function
        code_generator_tool = self.tools_map.get("CodeGeneratorAgent")
        image_analysis_tool = self.tools_map.get("ImageAnalysisAgent")
        human_approval_tool_func = self.tools_map.get("human_approval_tool").func if self.tools_map.get("human_approval_tool") else None

        save_plot_artifact_helper = None
        try:
            from core_tools.artifact_helpers import save_plot_artifact
            save_plot_artifact_helper = save_plot_artifact
        except ImportError:
            agent_flow_logger.error(f"{self.name}: Could not import save_plot_artifact helper!")

        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: ---> Entering {self.name}")
        final_status = "Failure"
        error_message = None
        state_delta = {}
        generated_code = None
        execution_result = None
        processed_data_path = None
        applied_steps = []

        # 1 & 2: Get context from state
        dataset_id = ctx.session.state.get("current_dataset_id", "d1") # Assume Orchestrator sets this
        datasets_state = ctx.session.state.get("datasets", {})
        dataset_info = datasets_state.get(dataset_id, {})
        input_data_path = dataset_info.get("raw_data_path") # Path from DataLoadingAgent
        preprocess_strategy = dataset_info.get("preprocess_strategy", {}) # e.g., {'imputation': 'mean', 'scaling': 'standard'}
        visualize_flag = dataset_info.get("visualize_after_preprocess", False) # Check if Orchestrator set this based on user req
        analyze_plots_flag = dataset_info.get("analyze_plots", False) # Check if Orchestrator set this

        await self._log(f"Starting preprocessing for dataset ID: {dataset_id}, Path: {input_data_path}", ctx)

        if not input_data_path or not os.path.exists(input_data_path):
            error_message = f"Input data path '{input_data_path}' not found or invalid for dataset ID '{dataset_id}'."
            await self._log(error_message, ctx, level="ERROR")
            yield self._create_final_event(ctx, final_status, error_message)
            return

        if not preprocess_strategy:
            error_message = f"Preprocessing strategy not found in state for dataset ID '{dataset_id}'."
            await self._log(error_message, ctx, level="ERROR")
            yield self._create_final_event(ctx, final_status, error_message)
            return

        # 3. Formulate prompt for Code Generator
        output_filename = f"processed_data_{dataset_id}.csv"
        absolute_output_path = os.path.abspath(os.path.join(WORKSPACE_DIR, output_filename))
        os.makedirs(WORKSPACE_DIR, exist_ok=True) # Ensure workspace exists

        # Construct strategy description for the prompt
        strategy_desc = ", ".join([f"{k}={v}" for k,v in preprocess_strategy.items()])
        code_gen_prompt = f"""
Write Python code using pandas and scikit-learn to preprocess the dataset at '{input_data_path}'.
Apply the following strategy: {strategy_desc}.
Common steps might include:
- Handle missing values (e.g., using imputation: {preprocess_strategy.get('imputation')}).
- Encode categorical features (e.g., using OneHotEncoder or LabelEncoder: {preprocess_strategy.get('encoding')}). Assume target column should NOT be encoded if present.
- Scale numerical features (e.g., using StandardScaler or MinMaxScaler: {preprocess_strategy.get('scaling')}). Assume target column should NOT be scaled.
- Identify target column if specified: '{dataset_info.get('target_column', 'target')}' and handle it appropriately during encoding/scaling.
Save the fully preprocessed DataFrame (including target column) to a new CSV file at: '{absolute_output_path}'. Create the directory if needed.
After saving, print the exact path: `print(f"SAVED_OUTPUT: processed_data={absolute_output_path}")`
Also print a summary of actions taken, e.g., `print(f"INFO: Applied mean imputation to [col1].")`, `print(f"INFO: Applied StandardScaler to [col2, col3].")`.
Include necessary imports and basic error handling.
Code:
"""
        await self._log(f"Prompting CodeGeneratorAgent for preprocessing code. Strategy: {strategy_desc}", ctx)

        # 4. Call CodeGeneratorAgent
        if self.code_generator_tool:
            try:
                async for event in self.code_generator_tool.run_async(ctx, user_content=genai_types.Content(parts=[genai_types.Part(text=code_gen_prompt)])):
                    if event.is_final_response() and event.content and event.content.parts:
                        generated_code = event.content.parts[0].text
                        if generated_code:
                            generated_code = generated_code.strip().strip('`').strip()
                            if generated_code.startswith('python'):
                                generated_code = generated_code[len('python'):].strip()
                            await self._log(f"Received preprocessing code.", ctx)
                        else:
                            error_message = "CodeGeneratorAgent returned empty code."
                            await self._log(error_message, ctx, level="ERROR")
                        break # Assume single response
                if not generated_code and not error_message: # Handle case where agent finishes without response
                    error_message = "CodeGeneratorAgent finished without returning code."
                    await self._log(error_message, ctx, level="ERROR")

            except Exception as e:
                error_message = f"Error calling CodeGeneratorAgent: {e}"
                await self._log(error_message, ctx, level="ERROR")
        else:
            error_message = "CodeGeneratorAgent tool not configured for PreprocessingAgent."
            await self._log(error_message, ctx, level="ERROR")

        # 5 & 6. Call Code Execution Tool (with basic retry placeholder)
        max_retries = 1 # Allow one retry on failure
        attempt = 0
        while attempt <= max_retries and generated_code and not processed_data_path and not error_message:
            attempt += 1
            if attempt > 1:
                await self._log(f"Retrying code execution (Attempt {attempt}/{max_retries})...", ctx)
                # TODO: Optionally re-prompt code generator with error context here

            if self.code_execution_tool:
                try:
                    execution_result = await self.code_execution_tool.func(code_string=generated_code, tool_context=ctx)
                    await self._log(f"Preprocessing code execution attempt {attempt}. Status: {execution_result.get('status')}", ctx)

                    # 7 & 8. Check result, parse output
                    if execution_result.get("status") == "success":
                        output_files = execution_result.get("output_files", {})
                        processed_data_path = output_files.get("processed_data") # Name matches SAVED_OUTPUT

                        if not processed_data_path or not os.path.exists(processed_data_path):
                            error_message = f"Preprocessing code ran but failed to produce or report 'processed_data' file at expected path convention. Output files found: {output_files}"
                            await self._log(error_message, ctx, level="ERROR")
                            processed_data_path = None # Reset path if invalid
                            # Break retry loop if file not found after successful run
                            break
                        else:
                            # Parse applied steps from stdout
                            stdout_lines = execution_result.get("stdout", "").splitlines()
                            applied_steps = [line.split("INFO:", 1)[1].strip() for line in stdout_lines if line.startswith("INFO:")]
                            await self._log(f"Successfully processed data to: {processed_data_path}. Steps: {applied_steps}", ctx)
                            break # Exit retry loop on success

                    else: # Execution failed
                        error_message = f"Preprocessing code execution failed (Attempt {attempt}). Stderr: {execution_result.get('stderr', 'N/A')}"
                        await self._log(error_message, ctx, level="ERROR")
                        # Stay in loop to potentially retry

                except Exception as e:
                    error_message = f"Error calling code_execution_tool: {e}"
                    await self._log(error_message, ctx, level="ERROR")
                    break # Exit retry loop on tool call error
            else:
                error_message = "code_execution_tool not configured."
                await self._log(error_message, ctx, level="ERROR")
                break # Exit loop

        # 9. Update state if successful
        if processed_data_path:
            final_status = "Success"
            state_updates = {
                f"datasets.{dataset_id}.processed_data_path": processed_data_path,
                f"datasets.{dataset_id}.preprocess_steps": applied_steps,
                f"datasets.{dataset_id}.preprocess_status": "success",
                f"datasets.{dataset_id}.error": None # Clear previous errors
            }
            # Merge updates into the main delta
            state_delta.update(state_updates)

            # 10. Handle Visualization if requested
            if visualize_flag:
                 # --- Optional HITL before visualization ---
                 proceed_with_viz = True
                 if self.human_approval_tool_func:
                     try:
                         hitl_prompt = f"Preprocessing successful. Generate visualization plots for dataset '{dataset_id}'? (yes/no)"
                         hitl_result = await self.human_approval_tool_func(prompt=hitl_prompt, options=["yes", "no"], tool_context=ctx)
                         if hitl_result.get("status") != "success" or hitl_result.get("response", "").lower() != "yes":
                             proceed_with_viz = False
                             await self._log("User opted out of visualization.", ctx, level="INFO")
                     except Exception as e:
                          await self._log(f"Error during visualization HITL check: {e}", ctx, level="WARNING")
                          # Default to proceeding if HITL fails

                 if proceed_with_viz:
                     await self._handle_visualization(ctx, dataset_id, processed_data_path, analyze_plots_flag, state_delta)

        else: # Preprocessing failed after retries
            final_status = "Failure"
            state_delta[f"datasets.{dataset_id}.preprocess_status"] = "failure"
            state_delta[f"datasets.{dataset_id}.error"] = error_message or "Unknown preprocessing error."

        # 11. Log handled above

        # 12. Yield final event
        yield self._create_final_event(ctx, final_status, error_message, state_delta)
        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: <--- Exiting {self.name}. Status: {final_status}")


    async def _handle_visualization(self, ctx: InvocationContext, dataset_id: str, data_path: str, analyze_flag: bool, state_delta: Dict):
        """Handles optional visualization generation and analysis."""
        await self._log(f"Handling visualization request for dataset {dataset_id}", ctx)

        # a. Formulate plotting prompt
        plot_filename_base = f"preprocess_plot_{dataset_id}_{int(time.time())}"
        plot_filename = f"{plot_filename_base}.png" # Default to png
        absolute_plot_path = os.path.abspath(os.path.join(WORKSPACE_DIR, plot_filename))
        os.makedirs(WORKSPACE_DIR, exist_ok=True)

        # Basic plot instructions, could be made more sophisticated
        plot_gen_prompt = f"""
Write Python code using pandas, matplotlib, and seaborn to visualize the preprocessed data at '{data_path}'.
Generate relevant plots like:
- A pairplot or correlation heatmap.
- Histograms or density plots for numerical features.
Save the primary plot to '{absolute_plot_path}'. Create the directory if needed.
After saving, print the path: `print(f"SAVED_OUTPUT: plot={absolute_plot_path}")`
Include necessary imports and basic error handling.
Code:
"""
        await self._log(f"Prompting CodeGeneratorAgent for visualization code.", ctx)

        # b. Call CodeGeneratorAgent
        vis_code = None
        if self.code_generator_tool:
            try:
                async for event in self.code_generator_tool.run_async(ctx, user_content=genai_types.Content(parts=[genai_types.Part(text=plot_gen_prompt)])):
                    if event.is_final_response() and event.content and event.content.parts:
                        vis_code = event.content.parts[0].text
                        if vis_code:
                            vis_code = vis_code.strip().strip('`').strip()
                            if vis_code.startswith('python'):
                                vis_code = vis_code[len('python'):].strip()
                            await self._log("Received visualization code.", ctx)
                        break
                if not vis_code: await self._log("CodeGeneratorAgent returned empty code for visualization.", ctx, level="WARNING")
            except Exception as e:
                await self._log(f"Error calling CodeGeneratorAgent for visualization: {e}", ctx, level="ERROR")
        else:
            await self._log("CodeGeneratorAgent tool not configured.", ctx, level="ERROR")

        # c. Call Code Execution Tool
        plot_local_path = None
        if vis_code and self.code_execution_tool:
            try:
                vis_exec_result = await self.code_execution_tool.func(code_string=vis_code, tool_context=ctx)
                await self._log(f"Visualization code execution status: {vis_exec_result.get('status')}", ctx)
                if vis_exec_result.get("status") == "success":
                    plot_local_path = vis_exec_result.get("output_files", {}).get("plot") # Matches SAVED_OUTPUT key
                    if not plot_local_path or not os.path.exists(plot_local_path):
                        await self._log("Visualization code ran but did not report/save plot file correctly.", ctx, level="WARNING")
                        plot_local_path = None
                else:
                    await self._log(f"Visualization code execution failed. Stderr: {vis_exec_result.get('stderr')}", ctx, level="ERROR")
            except Exception as e:
                await self._log(f"Error calling code_execution_tool for visualization: {e}", ctx, level="ERROR")

        # d & e. Save plot artifact using helper
        if plot_local_path and self.save_plot_artifact_helper:
            logical_plot_name = f"preprocess_{dataset_id}"
            # Call the async helper function
            artifact_name = await self.save_plot_artifact_helper(plot_local_path, logical_plot_name, ctx)

            # f. Update state with artifact name (directly modifying the passed delta)
            if artifact_name:
                plot_list_key = f"datasets.{dataset_id}.plots"
                # Get current plots list *from the state* (not delta) and append
                current_plots = ctx.session.state.get("datasets", {}).get(dataset_id, {}).get("plots", [])
                if isinstance(current_plots, list): # Ensure it's a list
                    new_plots_list = current_plots + [artifact_name]
                    state_delta[plot_list_key] = new_plots_list # Add updated list to delta
                    await self._log(f"Added plot artifact '{artifact_name}' to state delta.", ctx)
                else:
                    await self._log(f"State key '{plot_list_key}' is not a list. Cannot append plot artifact.", ctx, level="WARNING")

                # g. Handle Image Analysis if requested
                if analyze_flag and self.image_analysis_tool:
                    await self._analyze_plot(ctx, dataset_id, artifact_name, "preprocessing plot", state_delta)
            else:
                await self._log(f"Failed to save plot artifact for {plot_local_path}", ctx, level="ERROR")


    async def _analyze_plot(self, ctx: InvocationContext, dataset_id: str, artifact_name: str, plot_context_desc: str, state_delta: Dict):
        """Handles optional plot analysis."""
        await self._log(f"Handling analysis request for plot artifact: {artifact_name}", ctx)

        # Load artifact bytes (needed for simulated analysis)
        artifact_part = None
        if ctx.artifact_service:
            try:
                # Assuming synchronous load_artifact for now based on current ADK API
                artifact_part = ctx.load_artifact(artifact_name)
            except Exception as e:
                await self._log(f"Failed to load artifact {artifact_name} for analysis: {e}", ctx, level="ERROR")
                return
        else:
            await self._log("ArtifactService not available, cannot load image for analysis.", ctx, level="ERROR")
            return

        if artifact_part and artifact_part.inline_data and artifact_part.inline_data.data:
            image_bytes = artifact_part.inline_data.data
            question = f"Analyze this {plot_context_desc} for dataset {dataset_id}. What are the key insights regarding data quality or feature relationships?"

            # --- Call ImageAnalysisAgent Tool ---
            # The override in image_analyzer.py tries to read from state as a workaround.
            # We set the state temporarily. This is a simulation limitation.
            temp_state_updates = {
                f"temp:image_analysis_bytes_b64": base64.b64encode(image_bytes).decode('utf-8'),
                f"temp:image_analysis_question": question
            }
            ctx.session.state.update(temp_state_updates) # Update state directly before call

            await self._log(f"Calling ImageAnalysisAgent for artifact {artifact_name}", ctx)
            analysis_result_text = f"Placeholder: Analysis for {artifact_name} failed." # Default
            analysis_success = False
            if self.image_analysis_tool:
                try:
                    async for event in self.image_analysis_tool.run_async(ctx, user_content=genai_types.Content(parts=[genai_types.Part(text=question)])): # Pass question for context
                        if event.is_final_response():
                            if event.content and event.content.parts:
                                analysis_result_text = event.content.parts[0].text
                                analysis_success = True
                            else:
                                analysis_result_text = f"ImageAnalysisAgent returned no content for {artifact_name}."
                                if event.error_message:
                                     analysis_result_text += f" Error: {event.error_message}"
                            break
                    await self._log(f"Received analysis result for {artifact_name}", ctx)

                except Exception as e:
                    error_message = f"Error calling ImageAnalysisAgent tool: {e}"
                    await self._log(error_message, ctx, level="ERROR")
                    analysis_result_text = f"Error during analysis: {e}"
            else:
                await self._log("ImageAnalysisAgent tool not configured.", ctx, level="ERROR")
                analysis_result_text = "Analysis skipped: Tool not configured."

            # Clean up temporary state
            ctx.session.state.pop("temp:image_analysis_bytes_b64", None)
            ctx.session.state.pop("temp:image_analysis_question", None)

            # Store analysis result in state delta
            if analysis_success:
                analysis_key = f"datasets.{dataset_id}.analysis.{artifact_name.split('.')[0]}" # Use artifact name part as key
                state_delta[analysis_key] = analysis_result_text # Add to the main delta
                await self._log(f"Stored analysis result for {artifact_name} in state delta.", ctx)
            else:
                 await self._log(f"Analysis failed for {artifact_name}. Result: {analysis_result_text}", ctx, level="WARNING")

        else:
            await self._log(f"Could not get image bytes for artifact {artifact_name} to perform analysis.", ctx, level="WARNING")


    # Use helper methods defined in DataLoadingAgent (or move to a base class)
    async def _log(self, message: str, ctx: InvocationContext, level: str = "INFO"):
        """Logs using the logging_tool."""
        if self.logging_tool_func:
            try:
                await self.logging_tool_func(message=message, log_file_key="preprocessor_log", tool_context=ctx, level=level)
            except Exception as e:
                agent_flow_logger.error(f"INVOKE_ID={ctx.invocation_id} ({self.name}): Failed to log via tool: {e}")
                print(f"ERROR logging ({self.name}): {message}")
        else:
            agent_flow_logger.warning(f"INVOKE_ID={ctx.invocation_id}: logging_tool not found for {self.name}")
            print(f"LOG ({self.name}): {message}")

    def _create_final_event(self, ctx: InvocationContext, status: str, error_msg: Optional[str] = None, state_delta: Optional[Dict] = None) -> Event:
        """Creates the final event."""
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

# Instantiate the agent (tools will be added in main.py)
preprocessing_agent = PreprocessingAgent(tools=[])
print(f"--- PreprocessingAgent Instantiated (Model: {preprocessing_agent.model}) ---")
