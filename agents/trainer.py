# agents/trainer.py
import logging
import json
import os
import uuid # To generate unique model run IDs
import base64
import time
from typing import Optional, Dict, Any, AsyncGenerator, List

from google.adk.agents import LlmAgent # Could also use LoopAgent for structured hyperparameter tuning
from google.adk.agents.callback_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from config import (
    TASK_AGENT_MODEL,
    WORKSPACE_DIR,
    ARTIFACT_PREFIX,
    agent_flow_logger,
    tool_calls_logger,
)
# Assuming tools and helpers are accessible
# from ..core_tools import code_execution_tool, logging_tool, save_plot_artifact
# from .code_generator import code_generator_tool # AgentTool
# from .image_analyzer import image_analysis_tool # AgentTool

class TrainingAgent(LlmAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="TrainingAgent",
            model=TASK_AGENT_MODEL,
            instruction="""
Your task is to manage the training of Machine Learning model(s) on a preprocessed dataset.
1. You will receive the dataset identifier (e.g., 'd1') and a list of model configurations to train via state (e.g., `state['datasets'][dataset_id]['models_to_train'] = [{'type': 'LogisticRegression', 'params': {...}, 'model_base_id': 'LR'}, {'type': 'RandomForest', ...}]`).
2. Retrieve the processed data path from state (`state['datasets'][dataset_id]['processed_data_path']`).
3. Iterate through each model configuration in the 'models_to_train' list.
4. For each configuration:
    a. Generate a unique model run ID (e.g., '{model_base_id}_{dataset_id}_run{timestamp}').
    b. Formulate a detailed prompt for 'CodeGeneratorAgent' to write Python code (using scikit-learn or other relevant libraries) to:
        i. Load the processed data from the specified path.
        ii. Split data into training and testing sets (e.g., 80/20 split, stratified if classification).
        iii. Instantiate the specified model type (`config['type']`) with the given hyperparameters (`config['params']`).
        iv. Train the model on the training set.
        v. Save the trained model (using joblib or pickle) to a unique path in the WORKSPACE directory (e.g., '{WORKSPACE_DIR}/model_{model_run_id}.pkl').
        vi. Print the exact path of the saved model file using the convention: `print(f"SAVED_OUTPUT: model=/path/to/saved/model.pkl")`.
        vii. Optionally, generate and save training-related plots (like learning curves) if specified in the config, printing their paths using `SAVED_OUTPUT: plot_lc=...`.
        viii. Include necessary imports and error handling.
    c. Call 'CodeGeneratorAgent' tool.
    d. Call 'code_execution_tool' with the generated code.
    e. Check status. Handle errors/retries. Log errors using 'logging_tool' (key 'trainer_log'). If a model fails, record status in state and continue to the next model if possible. # TODO: Implement robust retry/error handling.
    f. If successful, parse 'output_files' for the 'model' path (and any 'plot' paths).
    g. Update the main state dictionary under `state['models'][model_run_id]` with: `path`, `type`, `params`, `dataset_id`, `status='trained'`, and any plot paths found. Use EventActions state_delta.
    h. If plots were generated, use the `save_plot_artifact` helper to save them as artifacts and update the state (`state['models'][model_run_id]['plots']`) with the artifact names.
5. Use 'logging_tool' to log progress for each model training attempt (key 'trainer_log').
6. Yield a final event summarizing the training process (e.g., "Trained models: [list of successful model_run_ids]"). Include the cumulative state_delta.
""",
            description="Trains one or more ML models using generated code based on configurations in state.",
            **kwargs # Pass tools, callbacks etc.
        )
        # Store tool references
        self.code_execution_tool = self.tools_map.get("code_execution_tool")
        self.logging_tool_func = self.tools_map.get("logging_tool").func if self.tools_map.get("logging_tool") else None
        self.code_generator_tool = self.tools_map.get("CodeGeneratorAgent")
        # Import helper
        try:
            from core_tools.artifact_helpers import save_plot_artifact
            self.save_plot_artifact_helper = save_plot_artifact
        except ImportError:
             agent_flow_logger.error(f"{self.name}: Could not import save_plot_artifact helper!")
             self.save_plot_artifact_helper = None


    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: ---> Entering {self.name}")
        final_status = "Success" # Overall status
        error_message = None
        state_delta = {} # Accumulate all state changes
        successful_model_ids = []
        failed_model_configs = []

        # 1 & 2: Get context from state
        dataset_id = ctx.session.state.get("current_dataset_id", "d1")
        datasets_state = ctx.session.state.get("datasets", {})
        dataset_info = datasets_state.get(dataset_id, {})
        processed_data_path = dataset_info.get("processed_data_path")
        models_to_train = dataset_info.get("models_to_train", []) # List of dicts

        await self._log(f"Starting training for dataset ID: {dataset_id}, Path: {processed_data_path}", ctx)

        if not processed_data_path or not os.path.exists(processed_data_path):
            error_message = f"Processed data path '{processed_data_path}' not found for dataset ID '{dataset_id}'."
            await self._log(error_message, ctx, level="ERROR")
            yield self._create_final_event(ctx, "Failure", error_message)
            return

        if not models_to_train:
            error_message = f"No model configurations found in state ('datasets.{dataset_id}.models_to_train') to train."
            await self._log(error_message, ctx, level="WARNING")
            # Not necessarily a failure of this agent, maybe just no models requested
            yield self._create_final_event(ctx, "Success", error_message)
            return

        # 3. Iterate through model configurations
        for model_config in models_to_train:
            model_type = model_config.get("type", "UnknownModel")
            model_params = model_config.get("params", {})
            model_base_id = model_config.get("model_base_id", model_type) # e.g., 'LR' or 'RandomForest'
            visualize_training = model_config.get("visualize_training", False) # Flag for learning curves etc.

            # 4a. Generate unique model run ID
            model_run_id = f"{model_base_id}_{dataset_id}_run{int(time.time())}"
            await self._log(f"Starting training for model config: {model_type}, Run ID: {model_run_id}", ctx)

            # 4b. Formulate prompt for Code Generator
            model_filename = f"model_{model_run_id}.pkl" # Or .joblib
            absolute_model_path = os.path.abspath(os.path.join(WORKSPACE_DIR, model_filename))
            os.makedirs(WORKSPACE_DIR, exist_ok=True)

            plot_instructions = ""
            plot_output_convention = ""
            if visualize_training:
                 plot_lc_filename = f"learning_curve_{model_run_id}.png"
                 absolute_plot_lc_path = os.path.abspath(os.path.join(WORKSPACE_DIR, plot_lc_filename))
                 plot_instructions = f"""
- After training, generate a learning curve plot using `sklearn.model_selection.learning_curve`.
- Save the plot to '{absolute_plot_lc_path}'. Create the directory if needed.
- Print the plot path: `print(f"SAVED_OUTPUT: plot_lc={absolute_plot_lc_path}")`"""
                 plot_output_convention = "SAVED_OUTPUT: plot_lc=/path/to/plot.png"


            code_gen_prompt = f"""
Write Python code using scikit-learn and joblib/pickle to train a '{model_type}' model.
- Load the preprocessed data from: '{processed_data_path}'.
- Assume the target column is named '{dataset_info.get('target_column', 'target')}'.
- Split the data into training and testing sets (e.g., 80/20 split, use stratify for classification if target is present). Use a fixed random_state=42 for reproducibility.
- Instantiate the '{model_type}' model with parameters: {json.dumps(model_params)}. Handle potential parameter errors gracefully.
- Train the model on the training set.
- Save the trained model object to: '{absolute_model_path}' using joblib or pickle. Create the directory if needed.
- Print the exact path of the saved model: `print(f"SAVED_OUTPUT: model={absolute_model_path}")`
{plot_instructions}
Include necessary imports (sklearn, pandas, joblib/pickle, os, json, matplotlib if plotting). Include basic try-except blocks.
Code:
"""
            await self._log(f"Prompting CodeGeneratorAgent for training code (Model ID: {model_run_id})", ctx)

            # 4c. Call CodeGeneratorAgent
            generated_code = None
            current_model_error = None
            if self.code_generator_tool:
                try:
                    async for event in self.code_generator_tool.run_async(ctx, user_content=genai_types.Content(parts=[genai_types.Part(text=code_gen_prompt)])):
                         if event.is_final_response() and event.content and event.content.parts:
                             generated_code = event.content.parts[0].text
                             if generated_code:
                                 generated_code = generated_code.strip().strip('`').strip()
                                 if generated_code.startswith('python'): generated_code = generated_code[len('python'):].strip()
                                 await self._log(f"Received training code for {model_run_id}.", ctx)
                             else: current_model_error = "CodeGeneratorAgent returned empty code."
                             break
                    if not generated_code and not current_model_error: current_model_error = "CodeGeneratorAgent finished without returning code."
                except Exception as e:
                    current_model_error = f"Error calling CodeGeneratorAgent: {e}"
            else:
                current_model_error = "CodeGeneratorAgent tool not configured."

            if current_model_error:
                 await self._log(current_model_error, ctx, level="ERROR")
                 failed_model_configs.append(model_config)
                 state_delta[f"models.{model_run_id}.status"] = "generation_failed"
                 state_delta[f"models.{model_run_id}.error"] = current_model_error
                 continue # Skip to the next model config

            # 4d & 4e. Call Code Execution Tool
            execution_result = None
            model_local_path = None
            plot_local_path = None # For potential plot output
            if generated_code and self.code_execution_tool:
                try:
                    execution_result = await self.code_execution_tool.func(code_string=generated_code, tool_context=ctx)
                    await self._log(f"Training code execution status for {model_run_id}: {execution_result.get('status')}", ctx)

                    if execution_result.get("status") != "success":
                        current_model_error = f"Training code execution failed for {model_run_id}. Stderr: {execution_result.get('stderr', 'N/A')}"
                        await self._log(current_model_error, ctx, level="ERROR")
                        # TODO: Implement retry logic for execution failure if desired
                        failed_model_configs.append(model_config)
                        state_delta[f"models.{model_run_id}.status"] = "execution_failed"
                        state_delta[f"models.{model_run_id}.error"] = current_model_error
                        continue # Skip to next model

                    # 4f. Parse output files
                    output_files = execution_result.get("output_files", {})
                    model_local_path = output_files.get("model") # Matches SAVED_OUTPUT: model=...
                    plot_local_path = output_files.get("plot_lc") # Matches SAVED_OUTPUT: plot_lc=...

                    if not model_local_path or not os.path.exists(model_local_path):
                         current_model_error = f"Training code ran but failed to produce/report model file for {model_run_id}. Found files: {output_files}"
                         await self._log(current_model_error, ctx, level="ERROR")
                         failed_model_configs.append(model_config)
                         state_delta[f"models.{model_run_id}.status"] = "output_missing"
                         state_delta[f"models.{model_run_id}.error"] = current_model_error
                         continue # Skip to next model

                except Exception as e:
                    current_model_error = f"Error during code_execution_tool call for {model_run_id}: {e}"
                    await self._log(current_model_error, ctx, level="ERROR")
                    failed_model_configs.append(model_config)
                    state_delta[f"models.{model_run_id}.status"] = "tool_error"
                    state_delta[f"models.{model_run_id}.error"] = current_model_error
                    continue # Skip to next model
            elif not generated_code:
                 # Error handled above, just continue
                 continue
            else: # Tool not configured
                 error_message = "code_execution_tool not configured." # Overall error
                 await self._log(error_message, ctx, level="ERROR")
                 break # Stop processing further models if tool is missing


            # 4g. Update state for successful model
            if model_local_path:
                successful_model_ids.append(model_run_id)
                model_state_updates = {
                    f"models.{model_run_id}.path": model_local_path,
                    f"models.{model_run_id}.type": model_type,
                    f"models.{model_run_id}.params": model_params,
                    f"models.{model_run_id}.dataset_id": dataset_id,
                    f"models.{model_run_id}.status": "trained",
                    f"models.{model_run_id}.plots": [], # Initialize plots list
                }
                state_delta.update(model_state_updates)
                await self._log(f"Successfully trained model {model_run_id}. Path: {model_local_path}", ctx)

                # 4h. Save plot artifact if generated
                if plot_local_path and self.save_plot_artifact_helper:
                    plot_logical_name = f"learning_curve_{model_run_id}"
                    artifact_name = await self.save_plot_artifact_helper(plot_local_path, plot_logical_name, ctx)
                    if artifact_name:
                         # Update the plots list for this specific model in the delta
                         plot_list_key = f"models.{model_run_id}.plots"
                         # Get potentially existing list from delta or initialize
                         current_plots = state_delta.get(plot_list_key, [])
                         current_plots.append(artifact_name)
                         state_delta[plot_list_key] = current_plots
                         await self._log(f"Saved training plot artifact '{artifact_name}' for model {model_run_id}", ctx)
                    else:
                         await self._log(f"Failed to save training plot artifact for {model_run_id}", ctx, level="WARNING")


        # End of loop through model configs

        # 6. Yield final event
        if not successful_model_ids and failed_model_configs:
            final_status = "Failure"
            error_message = f"All {len(failed_model_configs)} model training attempts failed."
        elif failed_model_configs:
             final_status = "Partial Success"
             error_message = f"Successfully trained {len(successful_model_ids)} models, but {len(failed_model_configs)} failed."
        elif not successful_model_ids and not failed_model_configs:
             final_status = "No Action" # Should not happen if models_to_train was not empty
             error_message = "No models were successfully trained or failed."
        else:
             final_status = "Success"

        summary_message = f"Training completed. Status: {final_status}. Successful models: {successful_model_ids}."
        if error_message:
             summary_message += f" Issues: {error_message}"

        yield self._create_final_event(ctx, final_status, error_message, state_delta, summary_message)
        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: <--- Exiting {self.name}. Status: {final_status}")


    # Helper methods (copy from DataLoadingAgent or move to base class)
    async def _log(self, message: str, ctx: InvocationContext, level: str = "INFO"):
        """Logs using the logging_tool."""
        if self.logging_tool_func:
            try:
                await self.logging_tool_func(message=message, log_file_key="trainer_log", tool_context=ctx, level=level)
            except Exception as e:
                agent_flow_logger.error(f"INVOKE_ID={ctx.invocation_id} ({self.name}): Failed to log via tool: {e}")
                print(f"ERROR logging ({self.name}): {message}")
        else:
            agent_flow_logger.warning(f"INVOKE_ID={ctx.invocation_id}: logging_tool not found for {self.name}")
            print(f"LOG ({self.name}): {message}")

    def _create_final_event(self, ctx: InvocationContext, status: str, error_msg: Optional[str] = None, state_delta: Optional[Dict] = None, final_message: Optional[str] = None) -> Event:
         """Creates the final event."""
         message = final_message or f"{self.name} finished with status: {status}."
         if error_msg and status != "Success":
             message += f" Details: {error_msg}"

         return Event(
             author=self.name,
             invocation_id=ctx.invocation_id,
             content=genai_types.Content(parts=[genai_types.Part(text=message)]),
             actions=EventActions(state_delta=state_delta) if state_delta else None,
             turn_complete=True,
             error_message=error_msg if status != "Success" else None
         )
