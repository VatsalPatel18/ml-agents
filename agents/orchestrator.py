# agents/orchestrator.py
import logging
import json
import uuid
from typing import Optional, Dict, Any, AsyncGenerator, List

from google.adk.agents import LlmAgent, AgentTool
from google.adk.agents.callback_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from config import (
    ORCHESTRATOR_MODEL,
    agent_flow_logger,
)
# Assuming tools are accessible - these will be passed during Runner setup
# from ..core_tools import logging_tool
# from . import ( # Import AgentTools for sub-agents
#     data_loading_tool, preprocessing_tool, training_tool,
#     evaluation_tool, reporting_tool
# )

class MLOrchestratorAgent(LlmAgent):
    def __init__(self, **kwargs):
        # Ensure AgentTools for task agents are passed in kwargs['tools']
        super().__init__(
            name="MLOrchestratorAgent",
            model=ORCHESTRATOR_MODEL,
            instruction="""
You are the ML Copilot Orchestrator, an expert AI assistant managing a team of specialized agents to perform Machine Learning tasks for a user. Your goal is to understand the user's high-level objective, dynamically plan the necessary steps, coordinate execution by invoking specialist agent tools, manage state, and communicate results.

**Workflow Management:**
1.  **Greet & Understand Goal:** Start by understanding the user's request from the initial message (`ctx.user_content`). Identify the primary goal (e.g., classification, regression, evaluation, preprocessing only) and the target dataset(s). Store this in state (`user_goal`, `task`, `datasets`).
2.  **Initial Plan/Next Step:** Determine the first logical step based on the goal (usually 'load'). Set `state['current_step']`.
3.  **Step Execution Loop:**
    a. Read the `current_step` from state.
    b. Determine the appropriate specialist `AgentTool` to call for this step (e.g., `DataLoadingAgent` tool for 'load' step).
    c. Prepare necessary context in state for the specialist (e.g., set `current_dataset_id`).
    d. Use the `logging_tool` (key 'orchestrator_log') to log the step being initiated.
    e. **Invoke Specialist:** Call the specialist `AgentTool`. Pass minimal user content, relying on state for context.
    f. **Process Result:** Receive the final event from the specialist agent. Check its status.
    g. **Update State:** Merge any `state_delta` from the specialist's event into the main context state. Log the outcome using `logging_tool`.
    h. **Handle Errors:** If the specialist failed, decide the next action: ask the user for clarification (yield text event), log failure and stop, or potentially trigger a retry mechanism (future enhancement). For now, log and stop on critical failures.
    i. **Determine Next Step:** If the step succeeded, decide the next logical step based on the original goal and the current state (e.g., after 'load' comes 'preprocess', after 'evaluate' might come 'report' or 'done'). Update `state['current_step']`. If no more steps, set to 'done'.
    j. **User Update:** Optionally yield an intermediate text event to inform the user about progress (e.g., "Preprocessing complete. Starting model training...").
    k. Repeat the loop until `current_step` is 'done' or an unrecoverable error occurs.
4.  **Non-Expert Handling:** If the user goal is vague (e.g., "build a model"), infer the task type (e.g., default to classification based on data?) and define a default strategy (e.g., try standard classifiers like 'LogisticRegression', 'RandomForest'). Store this strategy in state (`models_to_train`) for the `TrainingAgent`.
5.  **Final Output:** When `current_step` is 'done', yield a final summary message to the user.

**State Management:**
- Use `state['user_goal']`, `state['task']`
- Use `state['current_step']` to track workflow progress.
- Use `state['datasets'][dataset_id]` for dataset-specific info.
- Use `state['models'][model_run_id]` for model-specific info.
- Use `state['evaluate_models']` list to tell EvaluationAgent which models to evaluate.
- Use `state['report_config']` dict to configure the ReportingAgent.

**Tool Usage:**
- Call specialist agents via their `AgentTool` wrappers.
- Use `logging_tool` extensively.
""",
            description="The main ML Copilot orchestrator. Manages the workflow dynamically by coordinating specialist agents and tools based on user goals.",
            **kwargs # Pass tools (AgentTools for sub-agents, logging_tool), callbacks
        )
        # Store tool references
        self.logging_tool_func = self.tools_map.get("logging_tool").func if self.tools_map.get("logging_tool") else None
        # Store references to task agent tools
        self.task_agent_tools = {
            name: tool for name, tool in self.tools_map.items()
            if isinstance(tool, AgentTool) and tool.agent.name != self.name # Exclude self if passed
        }
        agent_flow_logger.info(f"{self.name} initialized with task agent tools: {list(self.task_agent_tools.keys())}")


    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: ---> Entering {self.name}")

        # --- Initial Goal Understanding (Simplified) ---
        if ctx.session.state.get("current_step") is None: # First time run for this session/invocation
            initial_query = ""
            if ctx.user_content and ctx.user_content.parts:
                initial_query = ctx.user_content.parts[0].text or ""

            await self._log(f"Received initial user query: {initial_query}", ctx)

            # TODO: Use LLM to parse the initial query into goal, task, dataset source etc.
            # For now, we'll use placeholder logic based on the example query.
            # This is a critical part where the Orchestrator's LLM reasoning is needed.
            # --- Placeholder Parsing Logic ---
            user_goal = initial_query # Store raw query for now
            task = "classification" # Infer from query / default
            source_path = "./my_data.csv" # Extract from query
            dataset_id = "d1" # Generate or assign ID
            models_to_try_config = [ # Default strategy for classification
                {"type": "LogisticRegression", "params": {"C": 1.0}, "model_base_id": "LR"},
                {"type": "RandomForestClassifier", "params": {"n_estimators": 100, "random_state": 42}, "model_base_id": "RF"}
            ]
            workflow_plan = ['load', 'preprocess', 'train', 'evaluate', 'report', 'done'] # Default plan
            # --- End Placeholder Parsing Logic ---

            # Initialize state
            initial_state_delta = {
                "user_goal": user_goal,
                "task": task,
                "current_step": workflow_plan[0], # Start with 'load'
                "workflow_plan": workflow_plan,
                "datasets": {
                    dataset_id: {
                        "raw_path_source": source_path,
                        "preprocess_strategy": {"imputation": "mean", "scaling": "standard"}, # Default strategy
                        "models_to_train": models_to_try_config,
                        "plots": [], # Initialize plot list
                        "analysis": {}, # Initialize analysis dict
                    }
                },
                "models": {}, # Initialize models dict
                "evaluate_models": [], # Will be populated after training
                "report_config": {"dataset_ids": [dataset_id]}, # Default report config
                "current_dataset_id": dataset_id, # Track current focus
            }
            ctx.session.state.update(initial_state_delta) # Direct update for initial setup
            await self._log(f"Initialized state. Plan: {workflow_plan}. Starting step: {workflow_plan[0]}", ctx)
            # Yield an initial event acknowledging the goal
            yield Event(
                author=self.name, invocation_id=ctx.invocation_id,
                content=genai_types.Content(parts=[genai_types.Part(text=f"Okay, I understand you want to perform '{task}' on '{source_path}'. I will proceed with the following steps: {workflow_plan[:-1]}.")])
                # actions=EventActions(state_delta=initial_state_delta) # Runner handles state commit
            )
            # Need to ensure state is committed before proceeding, runner handles this after yield

        # --- Step Execution Loop ---
        while True:
            current_step = ctx.session.state.get("current_step", "done")
            dataset_id = ctx.session.state.get("current_dataset_id", "d1") # Get current dataset focus
            workflow_plan = ctx.session.state.get("workflow_plan", ['done'])

            await self._log(f"Current step: {current_step}", ctx)

            if current_step == "done":
                await self._log("Workflow complete.", ctx)
                yield self._create_final_event(ctx, "Success", final_message="Workflow finished successfully.")
                break

            step_success = False
            step_error_message = None
            next_step_index = workflow_plan.index(current_step) + 1 if current_step in workflow_plan else -1

            # Map step name to AgentTool name (assuming convention)
            agent_tool_name_map = {
                "load": "DataLoadingAgent",
                "preprocess": "PreprocessingAgent",
                "train": "TrainingAgent",
                "evaluate": "EvaluationAgent",
                "report": "ReportingAgent",
            }
            target_agent_tool_name = agent_tool_name_map.get(current_step)
            target_agent_tool = self.task_agent_tools.get(target_agent_tool_name)

            if target_agent_tool:
                await self._log(f"Invoking {target_agent_tool_name} tool...", ctx)
                try:
                    # Pass minimal content, agent uses state
                    step_input_content = genai_types.Content(parts=[genai_types.Part(text=f"Execute step: {current_step} for dataset {dataset_id}")])

                    # Invoke the specialist agent tool
                    async for event in target_agent_tool.run_async(ctx, user_content=step_input_content):
                        # Orchestrator mainly cares about the *final* event from the sub-agent
                        if event.is_final_response():
                            await self._log(f"Received final response from {event.author}: {event.content.parts[0].text if event.content else 'No content'}", ctx)
                            if event.error_message:
                                step_success = False
                                step_error_message = f"{event.author} failed: {event.error_message}"
                            else:
                                # Assume success if no error message from agent
                                step_success = True
                                # State delta is handled by the runner from the yielded event's actions
                            break # Processed final event for this step

                except Exception as e:
                    step_success = False
                    step_error_message = f"Error invoking {target_agent_tool_name}: {e}"
                    agent_flow_logger.exception(f"INVOKE_ID={ctx.invocation_id}: Unhandled error invoking {target_agent_tool_name}")


            else: # No agent found for this step
                step_success = False
                step_error_message = f"No specialist agent tool found for step: {current_step}"
                await self._log(step_error_message, ctx, level="ERROR")


            # --- Process Step Outcome ---
            state_delta = {} # Delta for this orchestrator step
            if step_success:
                await self._log(f"Step '{current_step}' completed successfully.", ctx)
                # Determine next step
                if next_step_index >= 0 and next_step_index < len(workflow_plan):
                    next_step = workflow_plan[next_step_index]
                    state_delta["current_step"] = next_step
                    # Optionally yield progress update to user
                    yield Event(
                         author=self.name, invocation_id=ctx.invocation_id,
                         content=genai_types.Content(parts=[genai_types.Part(text=f"Step '{current_step}' complete. Proceeding to '{next_step}'...")]),
                         actions=EventActions(state_delta=state_delta.copy()) # Commit next step
                    )
                    # Clear delta after yielding
                    state_delta = {}
                else: # Should be 'done'
                    state_delta["current_step"] = "done"
                    # Don't yield here, let loop handle 'done' state

            else: # Step failed
                await self._log(f"Step '{current_step}' failed. Error: {step_error_message}", ctx, level="ERROR")
                # TODO: Implement more sophisticated error handling/retry strategy
                # For now, stop the workflow
                state_delta["current_step"] = "error" # Mark workflow as errored
                yield self._create_final_event(ctx, "Failure", step_error_message, state_delta)
                break # Exit loop on failure

            # If state delta wasn't yielded as progress, yield it now
            if state_delta:
                 yield Event(author=self.name, invocation_id=ctx.invocation_id, actions=EventActions(state_delta=state_delta))


        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: <--- Exiting {self.name} Workflow Loop.")


    # Helper methods (copy or use inheritance/util)
    async def _log(self, message: str, ctx: InvocationContext, level: str = "INFO"):
        """Logs using the logging_tool."""
        if self.logging_tool_func:
            try:
                # Ensure log_file_key is specific to the orchestrator
                await self.logging_tool_func(message=message, log_file_key="orchestrator_log", tool_context=ctx, level=level)
            except Exception as e:
                agent_flow_logger.error(f"INVOKE_ID={ctx.invocation_id} ({self.name}): Failed to log via tool: {e}")
                print(f"ERROR logging ({self.name}): {message}")
        else:
            agent_flow_logger.warning(f"INVOKE_ID={ctx.invocation_id}: logging_tool not found for {self.name}")
            print(f"LOG ({self.name}): {message}")

    def _create_final_event(self, ctx: InvocationContext, status: str, error_msg: Optional[str] = None, state_delta: Optional[Dict] = None, final_message: Optional[str] = None) -> Event:
         """Creates the final event for the orchestrator."""
         message = final_message or f"{self.name} finished workflow execution with status: {status}."
         if error_msg and status != "Success":
             message += f" Error encountered: {error_msg}"

         return Event(
             author=self.name,
             invocation_id=ctx.invocation_id,
             content=genai_types.Content(parts=[genai_types.Part(text=message)]),
             actions=EventActions(state_delta=state_delta) if state_delta else None,
             turn_complete=True, # Mark orchestrator turn as complete
             error_message=error_msg if status != "Success" else None
         )

