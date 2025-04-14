# agents/reporter.py
import logging
import json
from typing import Optional, Dict, Any, AsyncGenerator, List

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from config import (
    TASK_AGENT_MODEL, # Could use a model optimized for summarization/writing
    agent_flow_logger,
)
# Assuming tools are accessible
# from ..core_tools import logging_tool
# from .image_analyzer import image_analysis_tool # AgentTool

class ReportingAgent(LlmAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="ReportingAgent",
            model=TASK_AGENT_MODEL, # Or a model like Gemini 1.5 Pro for long context/summarization
            instruction="""
Your task is to generate a comprehensive report summarizing the completed ML workflow based on the information provided in the session state.
1. You will receive instructions via state about what to report on (e.g., `report_config = {'dataset_ids': ['d1'], 'model_ids': ['LR_d1_run1', 'RF_d1_run1']}`). If no specific config is found, summarize everything available in the state.
2. Access the session state (`ctx.session.state`) to retrieve all relevant details:
    - Dataset information (`state['datasets']`): raw paths, processed paths, preprocessing steps applied, associated plot artifact names, image analysis results for dataset plots.
    - Model information (`state['models']`): type, parameters, training status, evaluation metrics, associated plot artifact names, image analysis results for model plots.
3. Structure the report logically. A good structure might be:
    - **Workflow Summary:** Overview of the goal and steps taken.
    - **Dataset(s) Processed:** For each dataset ID:
        - Source path.
        - Preprocessing steps applied.
        - Path to processed data.
        - Key visualization artifact names (e.g., 'artifact_preprocess_plot_d1_123.png') and any available analysis results from state (e.g., `state['datasets']['d1']['analysis']['artifact_preprocess_plot_d1_123']`).
    - **Model(s) Trained & Evaluated:** For each model ID:
        - Model type and parameters.
        - Dataset used for training.
        - Training status.
        - Evaluation metrics (present clearly, perhaps in a table).
        - Key visualization artifact names (e.g., 'artifact_eval_plot_lr_d1_456.png') and any available analysis results from state (e.g., `state['models']['LR_d1_run1']['analysis']['artifact_eval_plot_lr_d1_456']`).
    - **Comparison & Conclusion:** If multiple models were evaluated, compare their performance based on metrics and analysis. Provide concluding remarks or potential next steps.
4. Generate the report in **Markdown format**. Make it clear, concise, and well-organized. Refer to plots by their artifact names.
5. Use the 'logging_tool' to log the start and completion of report generation (key 'reporter_log').
6. Yield a final event containing the generated Markdown report in the text part of the content.
(Optional Extension: Could later be modified to save the report as a .md artifact instead of returning text).
""",
            description="Generates a Markdown summary report of the ML workflow based on session state.",
            **kwargs # Pass tools, callbacks etc.
        )
        # Store tool references
        self.logging_tool_func = self.tools_map.get("logging_tool").func if self.tools_map.get("logging_tool") else None
        # Image analysis tool might be called by the LLM based on instructions if needed,
        # but primarily this agent reads analysis results from state.
        # self.image_analysis_tool = self.tools_map.get("ImageAnalysisAgent")


    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: ---> Entering {self.name}")
        final_status = "Success"
        error_message = None
        report_content = "Report generation failed." # Default

        await self._log(f"Starting report generation.", ctx)

        # 1. Get report config from state (optional)
        report_config = ctx.session.state.get("report_config", {})
        dataset_ids_to_report = report_config.get("dataset_ids") # If None, report all found
        model_ids_to_report = report_config.get("model_ids")     # If None, report all found

        # 2. Retrieve all relevant state
        full_state = ctx.session.state.copy() # Get a copy to avoid modifying during iteration
        datasets_data = full_state.get("datasets", {})
        models_data = full_state.get("models", {})
        user_goal = full_state.get("user_goal", "Not specified")
        task_type = full_state.get("task", "Not specified")

        # Filter data based on report_config if provided
        if dataset_ids_to_report:
            datasets_data = {k: v for k, v in datasets_data.items() if k in dataset_ids_to_report}
        if model_ids_to_report:
            models_data = {k: v for k, v in models_data.items() if k in model_ids_to_report}
        else:
            # If specific models aren't requested, filter by reported datasets
            if dataset_ids_to_report:
                 models_data = {k: v for k, v in models_data.items() if v.get("dataset_id") in dataset_ids_to_report}


        # 3 & 4. Generate Report Prompt for the LLM
        # Create a structured summary of the state data to pass to the LLM
        state_summary_for_prompt = {
            "user_goal": user_goal,
            "task_type": task_type,
            "datasets": datasets_data,
            "models": models_data
        }
        # Convert complex objects (like params) to str/JSON for prompt if needed
        try:
            state_summary_json = json.dumps(state_summary_for_prompt, indent=2, default=str) # Use default=str for non-serializable items
        except Exception as e:
            error_message = f"Failed to serialize state for report prompt: {e}"
            await self._log(error_message, ctx, level="ERROR")
            yield self._create_final_event(ctx, "Failure", error_message)
            return

        report_prompt = f"""
Generate a comprehensive Machine Learning workflow report in Markdown format based on the following state summary:

```json
{state_summary_json}
```

**Report Structure Guidelines:**
- **Workflow Summary:** Briefly state the user goal and the overall ML task type.
- **Dataset(s) Processed:** For each dataset ID found in the 'datasets' section:
    - Mention the source (`raw_path_source` or `raw_data_path`).
    - List the preprocessing steps applied (`preprocess_steps`).
    - State the path to the final processed data (`processed_data_path`).
    - List associated plot artifact names (`plots`) and include any analysis text found under the dataset's 'analysis' key (e.g., `datasets.d1.analysis.artifact_name_123`).
- **Model(s) Trained & Evaluated:** For each model ID found in the 'models' section:
    - Specify the model type (`type`) and parameters (`params`).
    - Mention the dataset ID it was trained on (`dataset_id`).
    - State the training status (`status`, e.g., 'trained', 'evaluated').
    - Present the evaluation metrics (`metrics`) clearly (e.g., in a list or table).
    - List associated plot artifact names (`plots`) and include any analysis text found under the model's 'analysis' key (e.g., `models.LR_d1_run1.analysis.artifact_name_456`).
- **Comparison & Conclusion:** If multiple models were evaluated for the same dataset, briefly compare their key metrics. Provide concluding remarks about the workflow outcome.

Ensure the report is well-organized and easy to read. Refer to plots by their artifact names (e.g., "See plot: artifact_preprocess_plot_d1_123.png").
"""

        # 5. Call self (LLM) to generate the report content
        # The LlmAgent's core logic handles the LLM call based on its instruction and the provided prompt
        try:
            # We pass the detailed prompt as the user message for this agent's internal run
            report_request_content = genai_types.Content(role='user', parts=[genai_types.Part(text=report_prompt)])
            # Use the agent's own run mechanism
            async for event in super(ReportingAgent, self)._run_async_impl(ctx, initial_user_content=report_request_content):
                 if event.is_final_response() and event.content and event.content.parts:
                     report_content = event.content.parts[0].text or "LLM generated empty report."
                     await self._log("Report content generated successfully.", ctx)
                     break # Got the report
            if report_content == "Report generation failed.": # Check if loop finished without result
                 error_message = "LLM did not generate report content."
                 await self._log(error_message, ctx, level="ERROR")
                 final_status = "Failure"

        except Exception as e:
            error_message = f"Error during report generation LLM call: {e}"
            await self._log(error_message, ctx, level="ERROR")
            final_status = "Failure"

        # 6. Yield final event with the report content
        yield self._create_final_event(ctx, final_status, error_message, final_message=report_content)
        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: <--- Exiting {self.name}. Status: {final_status}")


    # Helper methods (copy or use inheritance)
    async def _log(self, message: str, ctx: InvocationContext, level: str = "INFO"):
        """Logs using the logging_tool."""
        if self.logging_tool_func:
            try:
                await self.logging_tool_func(message=message, log_file_key="reporter_log", tool_context=ctx, level=level)
            except Exception as e:
                agent_flow_logger.error(f"INVOKE_ID={ctx.invocation_id} ({self.name}): Failed to log via tool: {e}")
                print(f"ERROR logging ({self.name}): {message}")
        else:
            agent_flow_logger.warning(f"INVOKE_ID={ctx.invocation_id}: logging_tool not found for {self.name}")
            print(f"LOG ({self.name}): {message}")

    def _create_final_event(self, ctx: InvocationContext, status: str, error_msg: Optional[str] = None, state_delta: Optional[Dict] = None, final_message: Optional[str] = None) -> Event:
         """Creates the final event."""
         message = final_message or f"{self.name} finished with status: {status}."
         # Don't include internal error messages in the final user-facing report event by default
         user_facing_error = error_msg if status != "Success" and final_message is None else None

         return Event(
             author=self.name,
             invocation_id=ctx.invocation_id,
             content=genai_types.Content(parts=[genai_types.Part(text=message)]),
             actions=EventActions(state_delta=state_delta) if state_delta else None,
             turn_complete=True,
             error_message=user_facing_error
         )
