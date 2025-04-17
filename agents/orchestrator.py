# agents/orchestrator.py (Refactored for ADK Web)
import logging
import json
import uuid
import time
import asyncio # Added for sleep
from typing import Optional, Dict, Any, AsyncGenerator, List
from pydantic import Field

from google.adk.agents import LlmAgent, BaseAgent # Added BaseAgent for type hint
from google.adk.tools import agent_tool # Correct way to import AgentTool module
from google.adk.agents.invocation_context import InvocationContext # Corrected path
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

# --- Project Imports ---
# Configuration
from config import (
    ORCHESTRATOR_MODEL,
    USE_LITELLM,
    agent_flow_logger,
)
# Core Tools needed by Orchestrator
from core_tools import logging_tool, human_approval_tool

# Task Agents (to be wrapped as tools)
from .data_loader import data_loading_agent
from .preprocessor import preprocessing_agent
from .trainer import training_agent
from .evaluator import evaluation_agent
from .reporter import reporting_agent

# Callbacks
from callbacks import (
    log_before_agent,
    log_after_agent,
    log_before_tool,
    log_after_tool,
)

# LiteLLM Import
if USE_LITELLM:
    try:
        from google.adk.models.lite_llm import LiteLlm
        print("LiteLLM imported successfully for Orchestrator.")
    except ImportError:
        print("ERROR: LiteLLM specified in config, but 'litellm' package not found. pip install litellm")
        LiteLlm = None
else:
    LiteLlm = None

# --- Agent Definition ---
class MLOrchestratorAgent(LlmAgent):
    # Mapping of tool names to AgentTool instances (initialized in __init__)
    tools_map: Dict[str, agent_tool.AgentTool] = Field(default_factory=dict)

    def __init__(self, **kwargs):
        # Determine model configuration
        model_config = LiteLlm(model=ORCHESTRATOR_MODEL) if USE_LITELLM and LiteLlm else ORCHESTRATOR_MODEL
        agent_flow_logger.info(f"Initializing MLOrchestratorAgent with model: {model_config}")

        # --- Prepare Tools List BEFORE super().__init__ ---
        orchestrator_tools = []
        task_agent_instances: List[BaseAgent] = [
            data_loading_agent,
            preprocessing_agent,
            training_agent,
            evaluation_agent,
            reporting_agent,
        ]

        # Create AgentTool wrappers for task agents
        # Use a local temporary map for logging if needed
        task_agent_tools_map_temp = {}
        for agent_instance in task_agent_instances:
            if agent_instance:
                try:
                    # Wrap the agent as an AgentTool for orchestrator invocation
                    tool_wrapper = agent_tool.AgentTool(
                        agent=agent_instance
                    )
                    orchestrator_tools.append(tool_wrapper)
                    task_agent_tools_map_temp[agent_instance.name] = tool_wrapper
                    agent_flow_logger.debug(f"Orchestrator prepared AgentTool for: {agent_instance.name}")
                except Exception as e:
                    agent_flow_logger.error(f"Failed to create AgentTool for {agent_instance.name} in Orchestrator: {e}", exc_info=True)
            else:
                agent_flow_logger.error(f"Agent instance {type(agent_instance)} is None, cannot create AgentTool.")

        # Add core tools needed directly by the orchestrator
        if logging_tool:
            orchestrator_tools.append(logging_tool)
        if human_approval_tool:
             orchestrator_tools.append(human_approval_tool)
        # ------------------------------------

        # Pass the list of tools to the parent class initializer
        # The parent class will create self.tools and self.tools_map
        super().__init__(
            name="MLOrchestratorAgent", # Name used for discovery in UI
            model=model_config,
            instruction="""
You are the ML Copilot Orchestrator, an expert AI assistant managing a team of specialized agents to perform Machine Learning tasks for a user. Your goal is to understand the user's high-level objective, dynamically plan the necessary steps, coordinate execution by invoking specialist agent tools, manage state, and communicate results.

**Workflow Management:**
1.  **Greet & Understand Goal:** Start by understanding the user's request from the initial message (`ctx.user_content`). Identify the primary goal (e.g., classification, regression, evaluation, preprocessing only), the target dataset(s), and any specific parameters or strategies mentioned. Store this parsed information in state (e.g., `user_goal`, `task`, `datasets`, `models_to_train`). Use your reasoning capabilities to fill in defaults if the user is vague (e.g., default to classification, try standard models).
2.  **Initial Plan/Next Step:** Determine the first logical step based on the goal (usually 'load'). Define the sequence of steps in `state['workflow_plan']`. Set `state['current_step']` to the first step.
3.  **Step Execution Loop:**
    a. Read the `current_step` from state. If 'done' or 'error', stop.
    b. Determine the appropriate specialist `AgentTool` to call for this step (e.g., `DataLoadingAgent` tool for 'load' step). Use the agent's name (e.g., "DataLoadingAgent") to look up the corresponding tool in `self.tools_map`.
    c. Prepare necessary context in state for the specialist (e.g., set `current_dataset_id`).
    d. **(Optional HITL):** Before critical/costly steps like 'train' or potentially 'preprocess' with complex strategies, consider using the `human_approval_tool`. Ask the user to confirm the plan or parameters (e.g., "About to train 2 models (LogisticRegression, RandomForestClassifier). Proceed? (yes/no)"). If the user responds 'no', update state to 'error' and stop.
    e. Use the `logging_tool` (key 'orchestrator_log') to log the step being initiated.
    f. **Invoke Specialist:** Call the specialist `AgentTool` found in `self.tools_map`. Pass minimal user content (e.g., "Execute step: {current_step}"), relying on state for detailed context.
    g. **Process Result:** Receive the final event from the specialist agent. Check its status (`event.error_message`).
    h. **Update State:** Merge any `state_delta` from the specialist's event into the main context state (the runner handles the actual merge based on the yielded event). Log the outcome using `logging_tool`.
    i. **Handle Errors:** If the specialist failed (`event.error_message` is present), log the failure. Decide the next action: stop the workflow by setting `current_step` to 'error', or potentially implement retry logic (not implemented yet). For now, stop on failure.
    j. **Determine Next Step:** If the step succeeded, find the current step in `state['workflow_plan']` and set `state['current_step']` to the next step in the list. If the last step was completed, set `current_step` to 'done'.
    k. **User Update:** Yield an intermediate text event to inform the user about progress (e.g., "Preprocessing complete. Starting model training..."). Include the state delta for the next step in the event's actions.
    l. Repeat the loop.
4.  **Final Output:** When `current_step` is 'done', yield a final summary message. If `current_step` is 'error', yield a message indicating the failure point.

**State Management Keys:**
- `user_goal`: Text description of user's objective.
- `task`: 'classification', 'regression', etc.
- `workflow_plan`: List of step names (e.g., ['load', 'preprocess', ... 'done']).
- `current_step`: The name of the step currently being executed or next to execute.
- `current_dataset_id`: Identifier for the dataset being worked on (e.g., 'd1').
- `datasets`: Dict mapping dataset_id to its info (paths, strategies, plots, analysis).
- `models`: Dict mapping model_run_id to its info (path, type, params, metrics, plots, analysis).
- `evaluate_models`: List of model_run_ids to be evaluated (populated after training).
- `report_config`: Dict to configure the ReportingAgent.

**Tool Usage:**
- Call specialist agents via their `AgentTool` wrappers (e.g., `DataLoadingAgent`, `PreprocessingAgent`, etc.). Use the agent name (e.g., "DataLoadingAgent") as the key to look up the tool in `self.tools_map`.
- Use `human_approval_tool` for optional user confirmation.
- Use `logging_tool` extensively.
""",
            description="The main ML Copilot orchestrator. Manages the ML workflow dynamically.",
            tools=orchestrator_tools,  # Provide orchestrator tools
            before_agent_callback=log_before_agent,
            after_agent_callback=log_after_agent,
            before_tool_callback=log_before_tool,
            after_tool_callback=log_after_tool,
            **kwargs
        )
        # Ensure tools and tools_map attributes are set for this orchestrator
        self.tools = orchestrator_tools
        self.tools_map = {t.name: t for t in orchestrator_tools}
        agent_flow_logger.info(f"{self.name} initialized.")
        # Debug: list the names of tools assigned to this orchestrator
        agent_flow_logger.debug(f"{self.name} tools_map keys: {[t.name for t in orchestrator_tools]}")


    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Get tool references/functions within the run context from self.tools_map
        logging_tool_func = self.tools_map.get("logging_tool").func if self.tools_map.get("logging_tool") else None
        human_approval_tool_func = self.tools_map.get("human_approval_tool").func if self.tools_map.get("human_approval_tool") else None
        # We will get the specific task agent tools from self.tools_map inside the loop using agent names

        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: ---> Entering {self.name}")

        # --- Initial Goal Understanding (Interactive) ---
        if ctx.session.state.get("current_step") is None:
            # Interactive setup: ask user for core workflow parameters
            print("\n--- ML Workflow Setup ---")
            dataset_path = input("Dataset path (e.g., ./my_data.csv): ").strip()
            if not dataset_path:
                print("No dataset path provided. Exiting.")
                return
            task_type = input("Task type (classification/regression) [classification]: ").strip() or "classification"
            if task_type.lower() == "classification":
                class_type = input("Classification type (binary/multiclass) [binary]: ").strip() or "binary"
                task_type = f"{task_type}-{class_type}"
            # Default workflow steps
            workflow_plan = ["load", "preprocess", "train", "evaluate", "report", "done"]
            # Build initial dataset and model configs
            models_to_train: List[Dict[str, Any]] = []
            print("Enter models to train (e.g., LogisticRegression). Leave blank to finish.")
            while True:
                model_name = input("Model type: ").strip()
                if not model_name:
                    break
                params_text = input("Model params as JSON (e.g., {\"C\":1.0}): ").strip() or "{}"
                try:
                    model_params = json.loads(params_text)
                except Exception:
                    print("Invalid JSON. Using empty params.")
                    model_params = {}
                models_to_train.append({"type": model_name, "params": model_params, "model_base_id": model_name})
            if not models_to_train:
                print("No models specified. Exiting.")
                return
            # Assemble state delta
            state_delta: Dict[str, Any] = {
                "user_goal": f"Train {task_type} model(s)",
                "task": task_type,
                "workflow_plan": workflow_plan,
                "datasets": {
                    "d1": {
                        "path": dataset_path,
                        "target_column": "target",
                        "models_to_train": models_to_train,
                        "plots": [],
                        "analysis": {},
                        "preprocess_strategy": {}
                    }
                },
                "current_dataset_id": "d1",
                "current_step": "load",
                "models": {},
                "evaluate_models": [],
                "report_config": {"dataset_ids": ["d1"]}
            }
            # Log and yield initial planning event
            await self._log(
                f"Interactive setup complete. Starting workflow with plan {workflow_plan}.",
                ctx, logging_tool_func
            )
            yield Event(
                author=self.name, invocation_id=ctx.invocation_id,
                content=genai_types.Content(parts=[genai_types.Part(text=f"Starting workflow: {workflow_plan}" )]),
                actions=EventActions(state_delta=state_delta)
            )
            await asyncio.sleep(0.1)


        # --- Step Execution Loop ---
        loop_count = 0
        max_loops = len(ctx.session.state.get("workflow_plan", [])) + 5 # Safety break

        while loop_count < max_loops:
            loop_count += 1
            current_state = ctx.session.state
            current_step = current_state.get("current_step", "done")
            workflow_plan = current_state.get("workflow_plan", ['done'])
            current_dataset_id = current_state.get("current_dataset_id")

            if current_step not in ["done", "error"] and not current_dataset_id:
                 datasets = current_state.get("datasets", {})
                 if datasets:
                     current_dataset_id = list(datasets.keys())[0]
                     await self._log(f"Setting current_dataset_id to first found: {current_dataset_id}", ctx, logging_tool_func, level="WARNING")
                     state_delta_for_next_step = {"current_dataset_id": current_dataset_id}
                     yield Event(author=self.name, invocation_id=ctx.invocation_id, actions=EventActions(state_delta=state_delta_for_next_step))
                     await asyncio.sleep(0.1)
                 else:
                     await self._log(f"Error: No dataset ID available for step {current_step}.", ctx, logging_tool_func, level="ERROR")
                     yield self._create_final_event(ctx, "Failure", "Workflow error: No dataset specified.", {"current_step": "error"})
                     break

            await self._log(f"--- Orchestrator Loop {loop_count}: Current Step = {current_step} (Dataset: {current_dataset_id}) ---", ctx, logging_tool_func)

            if current_step == "done":
                await self._log("Workflow complete.", ctx, logging_tool_func)
                break
            if current_step == "error":
                 await self._log("Workflow stopped due to error.", ctx, logging_tool_func, level="ERROR")
                 break

            step_success = False
            step_error_message = None
            next_step_index = workflow_plan.index(current_step) + 1 if current_step in workflow_plan else -1
            next_step = workflow_plan[next_step_index] if 0 <= next_step_index < len(workflow_plan) else "done"
            state_delta_for_next_step = {}

            # --- Optional HITL ---
            hitl_veto = False
            if current_step in ["train", "preprocess"] and human_approval_tool_func:
                prompt_text = f"About to start step '{current_step}' for dataset '{current_dataset_id}'."
                if current_step == "train":
                     models_to_train_cfg = ctx.session.state.get("datasets", {}).get(current_dataset_id, {}).get("models_to_train", [])
                     model_names = [m.get('type', 'Unknown') for m in models_to_train_cfg]
                     prompt_text += f" This will train the following models: {', '.join(model_names)}."
                prompt_text += "\nDo you want to proceed? (yes/no)"
                try:
                    hitl_result = await human_approval_tool_func(prompt=prompt_text, options=["yes", "no"], tool_context=ctx)
                    if hitl_result.get("status") != "success" or hitl_result.get("response", "").lower() != "yes":
                        hitl_veto = True
                        step_error_message = f"Workflow cancelled by user before step '{current_step}'."
                        await self._log(step_error_message, ctx, logging_tool_func, level="WARNING")
                except Exception as e:
                     hitl_veto = True
                     step_error_message = f"Error during human approval step: {e}"
                     await self._log(step_error_message, ctx, logging_tool_func, level="ERROR")

            if hitl_veto:
                 state_delta_for_next_step["current_step"] = "error"
                 yield self._create_final_event(ctx, "Failure", step_error_message, state_delta_for_next_step)
                 break

            # --- Map step name to AgentTool instance ---
            agent_tool_name_map = {
                "load": "DataLoadingAgent",
                "preprocess": "PreprocessingAgent",
                "train": "TrainingAgent",
                "evaluate": "EvaluationAgent",
                "report": "ReportingAgent",
            }
            target_agent_name = agent_tool_name_map.get(current_step)
            # Get the AgentTool instance directly from self.tools_map
            # AgentTool default name is the wrapped agent's name
            target_agent_tool = self.tools_map.get(target_agent_name)

            if target_agent_tool and isinstance(target_agent_tool, agent_tool.AgentTool): # Check it's an AgentTool
                await self._log(f"Invoking {target_agent_name} tool for step '{current_step}'...", ctx, logging_tool_func)
                try:
                    # Prepare ToolContext for specialist agent call
                    from google.adk.tools.tool_context import ToolContext
                    tool_ctx = ToolContext(ctx)
                    # Call the specialist agent tool
                    tool_input = {"request": f"Execute step: {current_step} for dataset {current_dataset_id}"}
                    tool_result = await target_agent_tool.run_async(
                        args=tool_input,
                        tool_context=tool_ctx,
                    )
                    await self._log(
                        f"{target_agent_name} result: {tool_result}",
                        ctx,
                        logging_tool_func,
                    )
                    step_success = True
                    step_error_message = None
                except Exception as e:
                    step_success = False
                    step_error_message = f"Error invoking {target_agent_name}: {e}"

                # No additional error handling needed here

            else:
                step_success = False
                step_error_message = f"No specialist agent tool found for step: {current_step} (Mapped Agent Name: {target_agent_name})"
                await self._log(step_error_message, ctx, logging_tool_func, level="ERROR")

            # --- Process Step Outcome ---
            if step_success:
                await self._log(f"Step '{current_step}' completed successfully.", ctx, logging_tool_func)

                if current_step == "train":
                    current_models_state = ctx.session.state.get("models", {})
                    trained_model_ids = [
                        model_id for model_id, model_data in current_models_state.items()
                        if model_data.get("status") == "trained" and model_data.get("dataset_id") == current_dataset_id
                    ]
                    if trained_model_ids:
                        state_delta_for_next_step["evaluate_models"] = trained_model_ids
                        await self._log(f"Models queued for evaluation: {trained_model_ids}", ctx, logging_tool_func)
                    else:
                         await self._log(f"No models were successfully trained in this step, skipping evaluation.", ctx, logging_tool_func, level="WARNING")
                         if "evaluate" in workflow_plan:
                             eval_index = workflow_plan.index("evaluate")
                             if next_step_index == eval_index:
                                  next_step_index += 1
                                  next_step = workflow_plan[next_step_index] if next_step_index < len(workflow_plan) else "done"

                state_delta_for_next_step["current_step"] = next_step
                progress_message = f"Step '{current_step}' complete."
                if next_step != "done":
                    progress_message += f" Proceeding to '{next_step}'..."

                yield Event(
                    author=self.name, invocation_id=ctx.invocation_id,
                    content=genai_types.Content(parts=[genai_types.Part(text=progress_message)]),
                    actions=EventActions(state_delta=state_delta_for_next_step)
                )

            else: # Step failed
                await self._log(f"Step '{current_step}' failed. Error: {step_error_message}", ctx, logging_tool_func, level="ERROR")
                state_delta_for_next_step["current_step"] = "error"
                yield self._create_final_event(ctx, "Failure", step_error_message, state_delta_for_next_step)
                break

            await asyncio.sleep(0.1)

        # --- End of Loop ---
        if loop_count >= max_loops:
             await self._log(f"Workflow exceeded maximum loop count ({max_loops}). Stopping.", ctx, logging_tool_func, level="ERROR")
             yield self._create_final_event(ctx, "Failure", "Workflow timed out (max loops exceeded).", {"current_step": "error"})
        elif ctx.session.state.get("current_step") != "error":
             final_state_check = ctx.session.state.get("current_step")
             if final_state_check == "done":
                  await self._log("Workflow loop finished normally.", ctx, logging_tool_func)
                  # Check if the last step was 'report' which should yield the final message
                  if not workflow_plan or workflow_plan[-2] != 'report': # Check second to last step before 'done'
                       yield self._create_final_event(ctx, "Success", final_message="Workflow finished successfully.")


        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: <--- Exiting {self.name} Workflow Loop.")


    async def _log(self, message: str, ctx: InvocationContext, logging_tool_func: Optional[callable], level: str = "INFO"):
        """Logs using the logging_tool function."""
        if logging_tool_func:
            try:
                await logging_tool_func(message=message, log_file_key="orchestrator_log", tool_context=ctx, level=level)
            except Exception as e:
                agent_flow_logger.error(f"INVOKE_ID={ctx.invocation_id} ({self.name}): Failed to log via tool: {e}")
                print(f"ERROR logging ({self.name}): {message}")
        else:
            # agent_flow_logger.warning(f"INVOKE_ID={ctx.invocation_id}: logging_tool_func not available for {self.name}")
            print(f"LOG ({self.name}): {message}")

    def _create_final_event(self, ctx: InvocationContext, status: str, error_msg: Optional[str] = None, state_delta: Optional[Dict] = None, final_message: Optional[str] = None) -> Event:
        """Creates the final event for the orchestrator."""
        message = final_message or f"{self.name} finished workflow execution with status: {status}."
        if error_msg and status != "Success":
            message += f" Error encountered: {error_msg}"
        # Always supply an EventActions instance (avoid passing None)
        actions_obj = EventActions(state_delta=state_delta or {})
        return Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=genai_types.Content(parts=[genai_types.Part(text=message)]),
            actions=actions_obj,
            turn_complete=True,
            error_message=error_msg if status != "Success" else None,
        )

# --- Instantiate the Agent ---
ml_orchestrator_agent = MLOrchestratorAgent()
print(f"--- MLOrchestratorAgent Instantiated for ADK Web (Instance: {ml_orchestrator_agent.name}) ---")
