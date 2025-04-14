# agents/orchestrator.py (Refactored for ADK Web)
import logging
import json
import uuid
import time
import asyncio # Added for sleep
from typing import Optional, Dict, Any, AsyncGenerator, List

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
    # Define fields for Pydantic validation if needed, especially for agents/tools
    # model_config = {"arbitrary_types_allowed": True} # Example

    def __init__(self, **kwargs):
        # Determine model configuration
        model_config = LiteLlm(model=ORCHESTRATOR_MODEL) if USE_LITELLM and LiteLlm else ORCHESTRATOR_MODEL
        agent_flow_logger.info(f"Initializing MLOrchestratorAgent with model: {model_config}")

        # --- Tool Setup within __init__ ---
        orchestrator_tools = []
        task_agent_instances: List[BaseAgent] = [ # List of agents to wrap
            data_loading_agent,
            preprocessing_agent,
            training_agent,
            evaluation_agent,
            reporting_agent,
        ]

        # Create AgentTool wrappers for task agents
        self.task_agent_tools_map = {} # Store for internal use if needed
        for agent_instance in task_agent_instances:
            if agent_instance:
                try:
                    # Ensure the agent instance itself has its tools configured if needed
                    # (This assumes task agents configure their own tools in their __init__)
                    tool_wrapper = agent_tool.AgentTool(
                        agent=agent_instance,
                        description=getattr(agent_instance, 'description', f"Tool wrapper for {agent_instance.name}")
                    )
                    orchestrator_tools.append(tool_wrapper)
                    self.task_agent_tools_map[agent_instance.name] = tool_wrapper # Map name to tool
                    agent_flow_logger.debug(f"Orchestrator created AgentTool for: {agent_instance.name}")
                except Exception as e:
                    agent_flow_logger.error(f"Failed to create AgentTool for {agent_instance.name} in Orchestrator: {e}", exc_info=True)
            else:
                 agent_flow_logger.error(f"Agent instance is None, cannot create AgentTool.")


        # Add core tools needed directly by the orchestrator
        if logging_tool:
            orchestrator_tools.append(logging_tool)
        if human_approval_tool:
             orchestrator_tools.append(human_approval_tool)
        # ------------------------------------

        super().__init__(
            name="MLOrchestratorAgent", # Name used for discovery in UI
            model=model_config,
            instruction="""
You are the ML Copilot Orchestrator... (Instruction remains the same as before)
""", # Keep the detailed instruction
            description="The main ML Copilot orchestrator. Manages the ML workflow dynamically.",
            tools=orchestrator_tools, # Assign the collected tools
            # Assign callbacks directly
            before_agent_callback=log_before_agent,
            after_agent_callback=log_after_agent,
            before_tool_callback=log_before_tool,
            after_tool_callback=log_after_tool,
            **kwargs # Pass any other standard BaseAgent args
        )
        # Store tool functions/maps for easier access in _run_async_impl
        self.logging_tool_func = self.tools_map.get("logging_tool").func if self.tools_map.get("logging_tool") else None
        self.human_approval_tool_func = self.tools_map.get("human_approval_tool").func if self.tools_map.get("human_approval_tool") else None
        agent_flow_logger.info(f"{self.name} initialized with tools: {[t.name for t in self.tools]}")


    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # --- _run_async_impl Logic Remains Largely the Same ---
        # It now uses self.task_agent_tools_map to find the correct AgentTool instance
        # based on the agent name derived from the current step.

        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: ---> Entering {self.name}")

        # --- Initial Goal Understanding (if first run) ---
        if ctx.session.state.get("current_step") is None:
            initial_query = ""
            if ctx.user_content and ctx.user_content.parts:
                initial_query = ctx.user_content.parts[0].text or ""
            await self._log(f"Received initial user query: {initial_query[:500]}...", ctx)

            parsing_prompt = f"""
Parse the following user request... (same parsing prompt as before)

User Request:
"{initial_query}"

JSON Output:
"""
            initial_state_delta = {}
            try:
                async for event in super(MLOrchestratorAgent, self)._run_async_impl(ctx, initial_user_content=genai_types.Content(parts=[genai_types.Part(text=parsing_prompt)])):
                    if event.is_final_response() and event.content and event.content.parts:
                        parsed_json_str = event.content.parts[0].text
                        parsed_json_str = parsed_json_str.strip().removeprefix("```json").removesuffix("```").strip()
                        agent_flow_logger.info(f"LLM parsing result: {parsed_json_str}")
                        parsed_state = json.loads(parsed_json_str)

                        # (Same state initialization logic as before...)
                        initial_state_delta["user_goal"] = parsed_state.get("user_goal", initial_query)
                        initial_state_delta["task"] = parsed_state.get("task", "classification")
                        initial_state_delta["workflow_plan"] = parsed_state.get("workflow_plan", ['load', 'preprocess', 'train', 'evaluate', 'report', 'done'])
                        initial_state_delta["datasets"] = parsed_state.get("datasets", {})
                        models_to_train_list = parsed_state.get("models_to_train", [])
                        if initial_state_delta["datasets"]:
                            first_dataset_id = list(initial_state_delta["datasets"].keys())[0]
                            initial_state_delta["datasets"][first_dataset_id]["models_to_train"] = models_to_train_list
                            # Ensure other dataset keys are initialized
                            for ds_id in initial_state_delta["datasets"]:
                                if "plots" not in initial_state_delta["datasets"][ds_id]: initial_state_delta["datasets"][ds_id]["plots"] = []
                                if "analysis" not in initial_state_delta["datasets"][ds_id]: initial_state_delta["datasets"][ds_id]["analysis"] = {}
                                if "preprocess_strategy" not in initial_state_delta["datasets"][ds_id]: initial_state_delta["datasets"][ds_id]["preprocess_strategy"] = {} # Add default if missing
                            initial_state_delta["current_dataset_id"] = first_dataset_id
                        else:
                             agent_flow_logger.error("LLM failed to parse dataset information.")
                             yield self._create_final_event(ctx, "Failure", "Could not identify the dataset from your request.")
                             return

                        initial_state_delta["models"] = {}
                        initial_state_delta["evaluate_models"] = []
                        initial_state_delta["report_config"] = {"dataset_ids": list(initial_state_delta["datasets"].keys())}
                        initial_state_delta["current_step"] = initial_state_delta["workflow_plan"][0] if initial_state_delta["workflow_plan"] else "done"

                        await self._log(f"Initialized state from LLM parsing. Plan: {initial_state_delta['workflow_plan']}. Starting step: {initial_state_delta['current_step']}", ctx)
                        break
                if not initial_state_delta:
                     raise ValueError("LLM parsing failed to produce state.")

            except Exception as e:
                agent_flow_logger.error(f"Failed to parse initial query using LLM: {e}", exc_info=True)
                yield self._create_final_event(ctx, "Failure", f"Sorry, I couldn't understand the initial request structure: {e}")
                return

            # Update the actual session state - Use EventActions for subsequent updates
            # For the *very first* update, direct modification before yielding might be okay,
            # but using EventActions is generally safer.
            # ctx.session.state.update(initial_state_delta) # Direct update for initial setup
            yield Event(
                author=self.name, invocation_id=ctx.invocation_id,
                content=genai_types.Content(parts=[genai_types.Part(text=f"Okay, planning workflow for task '{initial_state_delta['task']}'. Starting step: '{initial_state_delta['current_step']}'.")]),
                actions=EventActions(state_delta=initial_state_delta) # Commit initial state via actions
            )
            # Allow runner to process the state update
            await asyncio.sleep(0.1)


        # --- Step Execution Loop ---
        loop_count = 0
        max_loops = len(ctx.session.state.get("workflow_plan", [])) + 5 # Safety break

        while loop_count < max_loops:
            loop_count += 1
            # --- Read current state ---
            # It's crucial to read the state *within* the loop, as it gets updated by sub-agents
            current_state = ctx.session.state # Get current state dict
            current_step = current_state.get("current_step", "done")
            workflow_plan = current_state.get("workflow_plan", ['done'])
            current_dataset_id = current_state.get("current_dataset_id") # May be None initially

            # Ensure dataset_id is valid if needed for the step
            if current_step not in ["done", "error"] and not current_dataset_id:
                 datasets = current_state.get("datasets", {})
                 if datasets:
                     current_dataset_id = list(datasets.keys())[0] # Default to first dataset if missing
                     await self._log(f"Setting current_dataset_id to first found: {current_dataset_id}", ctx, level="WARNING")
                     # Update state immediately? Or via delta? Let's use delta.
                     state_delta_for_next_step = {"current_dataset_id": current_dataset_id}
                 else:
                     await self._log(f"Error: No dataset ID available for step {current_step}.", ctx, level="ERROR")
                     yield self._create_final_event(ctx, "Failure", "Workflow error: No dataset specified.", {"current_step": "error"})
                     break

            await self._log(f"--- Orchestrator Loop {loop_count}: Current Step = {current_step} (Dataset: {current_dataset_id}) ---", ctx)

            if current_step == "done":
                await self._log("Workflow complete.", ctx)
                # Final event yielded outside loop if needed, or ensure it was yielded previously
                break
            if current_step == "error":
                 await self._log("Workflow stopped due to error.", ctx, level="ERROR")
                 break # Final event should have been yielded

            step_success = False
            step_error_message = None
            next_step_index = workflow_plan.index(current_step) + 1 if current_step in workflow_plan else -1
            next_step = workflow_plan[next_step_index] if 0 <= next_step_index < len(workflow_plan) else "done"
            state_delta_for_next_step = {} # Reset delta for this iteration

            # --- Optional HITL ---
            hitl_veto = False
            # (HITL logic remains the same...)
            if current_step in ["train", "preprocess"] and self.human_approval_tool_func:
                prompt_text = f"About to start step '{current_step}' for dataset '{current_dataset_id}'."
                # ... (add details based on step as before) ...
                prompt_text += "\nDo you want to proceed? (yes/no)"
                try:
                    hitl_result = await self.human_approval_tool_func(prompt=prompt_text, options=["yes", "no"], tool_context=ctx)
                    if hitl_result.get("status") != "success" or hitl_result.get("response", "").lower() != "yes":
                        hitl_veto = True
                        step_error_message = f"Workflow cancelled by user before step '{current_step}'."
                        await self._log(step_error_message, ctx, level="WARNING")
                except Exception as e:
                     hitl_veto = True
                     step_error_message = f"Error during human approval step: {e}"
                     await self._log(step_error_message, ctx, level="ERROR")

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
            # Use the map created in __init__ to get the tool instance
            target_agent_tool = self.task_agent_tools_map.get(target_agent_name)

            if target_agent_tool:
                await self._log(f"Invoking {target_agent_name} tool for step '{current_step}'...", ctx)
                try:
                    step_input_content = genai_types.Content(parts=[genai_types.Part(text=f"Execute step: {current_step} for dataset {current_dataset_id}")])
                    step_final_event = None

                    async for event in target_agent_tool.run_async(ctx, user_content=step_input_content):
                        if event.is_final_response():
                            step_final_event = event
                            break

                    if step_final_event:
                        agent_name = step_final_event.author
                        final_content = step_final_event.content.parts[0].text if step_final_event.content and step_final_event.content.parts else "No content"
                        await self._log(f"Received final response from {agent_name}: {final_content[:200]}...", ctx)

                        if step_final_event.error_message:
                            step_success = False
                            step_error_message = f"Step '{current_step}' ({agent_name}) failed: {step_final_event.error_message}"
                        else:
                            step_success = True
                            # IMPORTANT: The runner automatically merges the state_delta from the sub-agent's
                            # final event (step_final_event.actions.state_delta) into ctx.session.state
                            # We don't need to manually merge it here.
                    else:
                         step_success = False
                         step_error_message = f"Step '{current_step}' ({target_agent_name}) did not return a final event."

                except Exception as e:
                    step_success = False
                    step_error_message = f"Error invoking {target_agent_name}: {e}"
                    agent_flow_logger.exception(f"INVOKE_ID={ctx.invocation_id}: Unhandled error invoking {target_agent_name}")

            else: # No agent found for this step
                step_success = False
                step_error_message = f"No specialist agent tool found for step: {current_step} (Agent Name: {target_agent_name})"
                await self._log(step_error_message, ctx, level="ERROR")

            # --- Process Step Outcome ---
            if step_success:
                await self._log(f"Step '{current_step}' completed successfully.", ctx)

                # --- Special logic after certain steps ---
                if current_step == "train":
                    # Read the *updated* state after the training agent ran
                    current_models_state = ctx.session.state.get("models", {})
                    trained_model_ids = [
                        model_id for model_id, model_data in current_models_state.items()
                        if model_data.get("status") == "trained" and model_data.get("dataset_id") == current_dataset_id
                    ]
                    if trained_model_ids:
                        state_delta_for_next_step["evaluate_models"] = trained_model_ids
                        await self._log(f"Models queued for evaluation: {trained_model_ids}", ctx)
                    else:
                         await self._log(f"No models were successfully trained in this step, skipping evaluation.", ctx, level="WARNING")
                         if "evaluate" in workflow_plan:
                             eval_index = workflow_plan.index("evaluate")
                             if next_step_index == eval_index:
                                  next_step_index += 1
                                  next_step = workflow_plan[next_step_index] if next_step_index < len(workflow_plan) else "done"

                # Determine next step and set state delta
                state_delta_for_next_step["current_step"] = next_step
                progress_message = f"Step '{current_step}' complete."
                if next_step != "done":
                    progress_message += f" Proceeding to '{next_step}'..."

                yield Event(
                    author=self.name, invocation_id=ctx.invocation_id,
                    content=genai_types.Content(parts=[genai_types.Part(text=progress_message)]),
                    actions=EventActions(state_delta=state_delta_for_next_step) # Commit next step
                )

            else: # Step failed
                await self._log(f"Step '{current_step}' failed. Error: {step_error_message}", ctx, level="ERROR")
                state_delta_for_next_step["current_step"] = "error"
                yield self._create_final_event(ctx, "Failure", step_error_message, state_delta_for_next_step)
                break # Exit loop on failure

            # Small delay allows runner to process event/state updates
            await asyncio.sleep(0.1)

        # --- End of Loop ---
        if loop_count >= max_loops:
             await self._log(f"Workflow exceeded maximum loop count ({max_loops}). Stopping.", ctx, level="ERROR")
             yield self._create_final_event(ctx, "Failure", "Workflow timed out (max loops exceeded).", {"current_step": "error"})
        elif ctx.session.state.get("current_step") != "error": # Ensure final 'done' message if loop finished normally
             final_state_check = ctx.session.state.get("current_step")
             if final_state_check == "done":
                  await self._log("Workflow loop finished normally.", ctx)
                  # Final success message might have been yielded by the last step ('report')
                  # Or yield a generic one here if needed.
                  # yield self._create_final_event(ctx, "Success", final_message="Workflow finished successfully.")


        agent_flow_logger.info(f"INVOKE_ID={ctx.invocation_id}: <--- Exiting {self.name} Workflow Loop.")


    # Helper methods (_log, _create_final_event) remain the same...
    async def _log(self, message: str, ctx: InvocationContext, level: str = "INFO"):
        """Logs using the logging_tool."""
        if self.logging_tool_func:
            try:
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

# --- Instantiate the Agent ---
# This instance will be discovered by 'adk web' via agents/__init__.py
ml_orchestrator_agent = MLOrchestratorAgent()
print(f"--- MLOrchestratorAgent Instantiated for ADK Web (Instance: {ml_orchestrator_agent.name}) ---")

