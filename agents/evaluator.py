# agents/evaluator.py
import logging
import json
import os
import base64
import time
from typing import Optional, Dict, Any, AsyncGenerator, List

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext # Corrected path
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

class EvaluationAgent(LlmAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="EvaluationAgent",
            model=TASK_AGENT_MODEL,
            instruction="""
Your task is to manage the evaluation of one or more trained Machine Learning models.
1. You will receive the dataset identifier (e.g., 'd1') and a list of model run IDs to evaluate via state (e.g., `state['evaluate_models'] = ['LR_d1_run1', 'RF_d1_run1']`).
2. Retrieve the processed data path (`state['datasets'][dataset_id]['processed_data_path']`) and the target column name.
3. Iterate through each model run ID in the list.
4. For each model run ID:
    a. Retrieve the model path from state (`state['models'][model_run_id]['path']`).
    b. Formulate a detailed prompt for 'CodeGeneratorAgent' to write Python code (using scikit-learn, pandas) to:
        i. Load the processed data and the specific trained model from their paths.
        ii. Ensure the data used for evaluation is appropriate (e.g., the test split if created during training, or the full dataset if specified). The training code should ideally have saved the test set indices or the test set itself. If not, assume the evaluation needs to re-split or use the whole dataset based on context (or ask Orchestrator/user). For now, assume the code needs to load the full processed data and potentially re-split using the same random_state=42 as training.
        iii. Make predictions on the test set.
        iv. Calculate standard evaluation metrics (e.g., accuracy, precision, recall, F1-score, AUC for classification; MSE, MAE, R2 for regression). Define the metrics needed based on the task type (e.g., `state['task']`).
        v. Print the calculated metrics as a JSON string using the convention: `print(f"METRICS: {json.dumps(metrics_dict)}")`.
        vi. Optionally, generate evaluation plots (e.g., confusion matrix, ROC curve) if requested (`visualize_evaluation` flag in state), save them to unique files in WORKSPACE_DIR, and print their paths using `print(f"SAVED_OUTPUT: plot_cm=/path/to/cm.png")`, `print(f"SAVED_OUTPUT: plot_roc=/path/to/roc.png")`.
        vii. Include necessary imports and error handling.
    c. Call 'CodeGeneratorAgent' tool.
    d. Call 'code_execution_tool' with the generated code.
    e. Check status. Handle errors/retries. Log errors using 'logging_tool' (key 'evaluator_log'). If evaluation fails for a model, record status in state and continue. # TODO: Implement robust retry/error handling.
    f. If successful, parse stdout for the 'METRICS' JSON string. Parse 'output_files' for any plot paths (e.g., 'plot_cm', 'plot_roc').
    g. Update state: `state['models'][model_run_id]['metrics'] = parsed_metrics`, `state['models'][model_run_id]['status'] = 'evaluated'`. Use EventActions state_delta.
    h. If plots were generated, use the `save_plot_artifact` helper to save them as artifacts and update the state (`state['models'][model_run_id]['plots']`) with the artifact names.
    i. If image analysis is requested (`analyze_plots` flag), call 'ImageAnalysisAgent' tool for generated plot artifacts and store results in state.
5. Use 'logging_tool' to log progress for each model evaluation attempt (key 'evaluator_log').
6. Yield a final event summarizing the evaluation process (e.g., "Evaluated models: [list of model_run_ids]"). Include the cumulative state_delta.
""",
            description="Evaluates trained models using generated code, calculates metrics, optionally visualizes/analyzes plots, and updates state.",
            **kwargs # Pass tools, callbacks etc.
        )
        # Store tool references
        self.code_execution_tool = self.tools_map.get("code_execution_tool")
        self.logging_tool_func = self.tools_map.get("logging_tool").func if self.tools_map.get("logging_tool") else None
        self.code_generator_tool = self.tools_map.get("CodeGeneratorAgent")
        self.image_analysis_tool = self.tools_map.get("ImageAnalysisAgent")
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
        evaluated_model_ids = []
        failed_model_ids = []

        # 1 & 2: Get context from state
        dataset_id = ctx.session.state.get("current_dataset_id", "d1")
        model_ids_to_evaluate = ctx.session.state.get("evaluate_models", []) # List of model run IDs
        datasets_state = ctx.session.state.get("datasets", {})
        models_state = ctx.session.state.get("models", {})
        dataset_info = datasets_state.get(dataset_id, {})
        processed_data_path = dataset_info.get("processed_data_path")
        target_column = dataset_info.get("target_column", "target")
        task_type = ctx.session.state.get("task", "classification") # Default to classification
        visualize_flag = ctx.session.state.get("visualize_evaluation", True) # Default to visualize
        analyze_plots_flag = ctx.session.state.get("analyze_plots", False)

        await self._log(f"Starting evaluation for dataset ID: {dataset_id}, Models: {model_ids_to_evaluate}", ctx)

        if not processed_data_path or not os.path.exists(processed_data_path):
            error_message = f"Processed data path '{processed_data_path}' not found for dataset ID '{dataset_id}'."
            await self._log(error_message, ctx, level="ERROR")
            yield self._create_final_event(ctx, "Failure", error_message)
            return

        if not model_ids_to_evaluate:
            error_message = "No model IDs found in state ('evaluate_models') to evaluate."
            await self._log(error_message, ctx, level="WARNING")
            yield self._create_final_event(ctx, "Success", error_message) # No models to eval isn't failure
            return

        # 3. Iterate through model IDs
        for model_run_id in model_ids_to_evaluate:
            await self._log(f"Starting evaluation for model run ID: {model_run_id}", ctx)
            model_info = models_state.get(model_run_id, {})
            model_path = model_info.get("path")
            model_type = model_info.get("type", "Unknown")

            if not model_path or not os.path.exists(model_path):
                error_message = f"Model path '{model_path}' not found for model ID '{model_run_id}'."
                await self._log(error_message, ctx, level="ERROR")
                failed_model_ids.append(model_run_id)
                state_delta[f"models.{model_run_id}.status"] = "evaluation_failed"
                state_delta[f"models.{model_run_id}.error"] = error_message
                continue # Skip to next model

            # 4b. Formulate prompt for Code Generator
            metrics_convention = f"print(f\"METRICS: {{json.dumps(metrics_dict)}}\")" # Requires json import
            plot_outputs = {} # To store expected plot outputs like {'plot_cm': '/path/...', 'plot_roc': '/path/...'}
            plot_instructions = ""
            if visualize_flag:
                # Define expected plot outputs
                cm_filename = f"confusion_matrix_{model_run_id}.png"
                roc_filename = f"roc_curve_{model_run_id}.png"
                abs_cm_path = os.path.abspath(os.path.join(WORKSPACE_DIR, cm_filename))
                abs_roc_path = os.path.abspath(os.path.join(WORKSPACE_DIR, roc_filename))
                plot_outputs["plot_cm"] = abs_cm_path
                plot_outputs["plot_roc"] = abs_roc_path

                plot_instructions = f"""
- Generate evaluation plots:
  - Confusion Matrix: Save to '{abs_cm_path}'. Print path `print(f"SAVED_OUTPUT: plot_cm={abs_cm_path}")`.
  - ROC Curve and AUC: Save to '{abs_roc_path}'. Print path `print(f"SAVED_OUTPUT: plot_roc={abs_roc_path}")`.
- Create directories if needed."""

            code_gen_prompt = f"""
Write Python code using scikit-learn, pandas, joblib/pickle, json, and matplotlib/seaborn to evaluate a trained '{model_type}' model.
- Load the trained model from: '{model_path}'.
- Load the preprocessed data from: '{processed_data_path}'.
- Assume the target column is named '{target_column}'.
- Split the data into training and testing sets using the SAME 80/20 split and random_state=42 as potentially used in training (important for consistent evaluation). Use stratify if task is classification.
- Make predictions on the test set.
- Calculate evaluation metrics appropriate for a '{task_type}' task (e.g., accuracy, precision, recall, f1, roc_auc for classification; mse, mae, r2 for regression). Store them in a dictionary called `metrics_dict`.
- Print the metrics dictionary as a JSON string: {metrics_convention}.
{plot_instructions}
Include necessary imports and basic error handling.
Code:
"""
            await self._log(f"Prompting CodeGeneratorAgent for evaluation code (Model ID: {model_run_id})", ctx)

            # 4c. Call CodeGeneratorAgent
            generated_code = None
            current_model_error = None
            # (Error handling similar to TrainingAgent...)
            if self.code_generator_tool:
                try:
                    async for event in self.code_generator_tool.run_async(ctx, user_content=genai_types.Content(parts=[genai_types.Part(text=code_gen_prompt)])):
                         if event.is_final_response() and event.content and event.content.parts:
                             generated_code = event.content.parts[0].text
                             if generated_code:
                                 generated_code = generated_code.strip().strip('`').strip()
                                 if generated_code.startswith('python'): generated_code = generated_code[len('python'):].strip()
                                 await self._log(f"Received evaluation code for {model_run_id}.", ctx)
                             else: current_model_error = "CodeGeneratorAgent returned empty code."
                             break
                    if not generated_code and not current_model_error: current_model_error = "CodeGeneratorAgent finished without returning code."
                except Exception as e:
                    current_model_error = f"Error calling CodeGeneratorAgent: {e}"
            else:
                current_model_error = "CodeGeneratorAgent tool not configured."

            if current_model_error:
                 await self._log(current_model_error, ctx, level="ERROR")
                 failed_model_ids.append(model_run_id)
                 state_delta[f"models.{model_run_id}.status"] = "evaluation_failed"
                 state_delta[f"models.{model_run_id}.error"] = current_model_error
                 continue

            # 4d & 4e. Call Code Execution Tool
            execution_result = None
            parsed_metrics = None
            generated_plot_paths = {} # Store paths reported by the code
            if generated_code and self.code_execution_tool:
                try:
                    execution_result = await self.code_execution_tool.func(code_string=generated_code, tool_context=ctx)
                    await self._log(f"Evaluation code execution status for {model_run_id}: {execution_result.get('status')}", ctx)

                    if execution_result.get("status") != "success":
                        current_model_error = f"Evaluation code execution failed for {model_run_id}. Stderr: {execution_result.get('stderr', 'N/A')}"
                        await self._log(current_model_error, ctx, level="ERROR")
                        # TODO: Implement retry logic if desired
                        failed_model_ids.append(model_run_id)
                        state_delta[f"models.{model_run_id}.status"] = "evaluation_failed"
                        state_delta[f"models.{model_run_id}.error"] = current_model_error
                        continue

                    # 4f. Parse metrics and plot paths from stdout/output_files
                    stdout = execution_result.get("stdout", "")
                    for line in stdout.splitlines():
                        if line.startswith("METRICS:"):
                            try:
                                metrics_json = line.split(":", 1)[1].strip()
                                parsed_metrics = json.loads(metrics_json)
                                await self._log(f"Parsed metrics for {model_run_id}: {parsed_metrics}", ctx)
                                break # Assume only one metrics line
                            except Exception as e:
                                await self._log(f"Failed to parse METRICS line for {model_run_id}: '{line}'. Error: {e}", ctx, level="WARNING")

                    output_files = execution_result.get("output_files", {})
                    for key, path in output_files.items():
                         if key.startswith("plot_"): # Convention from prompt
                             generated_plot_paths[key] = path

                    if not parsed_metrics:
                         await self._log(f"Warning: Metrics not found in stdout for {model_run_id}.", ctx, level="WARNING")
                         # Consider this a failure or partial success? For now, treat as warning.
                         # parsed_metrics = {"warning": "Metrics not found in output"}


                except Exception as e:
                    current_model_error = f"Error during code_execution_tool call for {model_run_id}: {e}"
                    await self._log(current_model_error, ctx, level="ERROR")
                    failed_model_ids.append(model_run_id)
                    state_delta[f"models.{model_run_id}.status"] = "tool_error"
                    state_delta[f"models.{model_run_id}.error"] = current_model_error
                    continue
            elif not generated_code:
                 continue # Error handled previously
            else: # Tool not configured
                 error_message = "code_execution_tool not configured."
                 await self._log(error_message, ctx, level="ERROR")
                 break # Stop processing further models

            # 4g. Update state for successful evaluation
            evaluated_model_ids.append(model_run_id)
            eval_state_updates = {
                f"models.{model_run_id}.metrics": parsed_metrics or {}, # Store empty dict if parsing failed
                f"models.{model_run_id}.status": "evaluated",
                f"models.{model_run_id}.error": None # Clear previous errors if any
            }
            state_delta.update(eval_state_updates)
            await self._log(f"Successfully evaluated model {model_run_id}.", ctx)

            # 4h. Save plot artifacts
            saved_artifact_names = []
            if generated_plot_paths and self.save_plot_artifact_helper:
                for plot_key, plot_local_path in generated_plot_paths.items():
                     if os.path.exists(plot_local_path):
                         plot_logical_name = f"{plot_key}_{model_run_id}" # e.g., plot_cm_LR_d1_run123
                         artifact_name = await self.save_plot_artifact_helper(plot_local_path, plot_logical_name, ctx)
                         if artifact_name:
                             saved_artifact_names.append(artifact_name)
                         else:
                             await self._log(f"Failed to save evaluation plot artifact for {plot_local_path}", ctx, level="WARNING")
                     else:
                         await self._log(f"Plot file reported by code execution not found: {plot_local_path}", ctx, level="WARNING")

                if saved_artifact_names:
                     plot_list_key = f"models.{model_run_id}.plots"
                     current_plots = state_delta.get(plot_list_key, models_state.get(model_run_id, {}).get('plots', [])) # Get existing plots
                     if isinstance(current_plots, list):
                         current_plots.extend(saved_artifact_names)
                         state_delta[plot_list_key] = current_plots # Add new plots to delta
                         await self._log(f"Updated plot artifacts in state for {model_run_id}: {saved_artifact_names}", ctx)
                     else:
                         await self._log(f"State key '{plot_list_key}' is not a list. Cannot append plot artifacts.", ctx, level="WARNING")


            # 4i. Analyze plots if requested
            if analyze_plots_flag and saved_artifact_names and self.image_analysis_tool:
                 for artifact_name in saved_artifact_names:
                     # Use helper from PreprocessingAgent (or move to base class/util)
                     # Need to handle the state passing mechanism carefully
                     # await self._analyze_plot(ctx, dataset_id, artifact_name, "evaluation plot", state_delta)
                     await self._log(f"Image analysis requested for {artifact_name}, but skipping in this version (requires state update before tool call).", ctx, level="WARNING")


        # End of loop through model IDs

        # 6. Yield final event
        if not evaluated_model_ids and failed_model_ids:
            final_status = "Failure"
            error_message = f"All {len(failed_model_ids)} model evaluation attempts failed."
        elif failed_model_ids:
             final_status = "Partial Success"
             error_message = f"Successfully evaluated {len(evaluated_model_ids)} models, but {len(failed_model_ids)} failed."
        elif not evaluated_model_ids and not failed_model_ids:
             final_status = "No Action"
             error_message = "No models were successfully evaluated or failed."
        else:
             final_status = "Success"

        summary_message = f"Evaluation completed. Status: {final_status}. Evaluated models: {evaluated_model_ids}."
        if error_message:
             summary_message += f" Issues: {error_message}"

        yield self._create_final_event(ctx, final_status, error_message, state_delta, summary_message)
        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: <--- Exiting {self.name}. Status: {final_status}")


    # Helper methods (copy or use inheritance)
    async def _log(self, message: str, ctx: InvocationContext, level: str = "INFO"):
        """Logs using the logging_tool."""
        if self.logging_tool_func:
            try:
                await self.logging_tool_func(message=message, log_file_key="evaluator_log", tool_context=ctx, level=level)
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

