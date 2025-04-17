# main.py
# Allow selecting model provider and API key via CLI before loading config
import argparse, os, sys
parser = argparse.ArgumentParser(description="Run the ML Copilot ADK application.")
parser.add_argument("--provider", choices=["google", "openai", "ollama"],
                    help="Primary model provider to use (overrides PRIMARY_PROVIDER env var).")
parser.add_argument("--api-key", dest="api_key",
                    help="API key for the selected provider (sets GOOGLE_API_KEY or OPENAI_API_KEY).")
parser.add_argument("--openai-api-base", dest="openai_api_base",
                    help="Custom OpenAI API base URL (sets OPENAI_API_BASE env var).")
parser.add_argument("--ollama-api-base", dest="ollama_api_base",
                    help="Ollama API base URL (sets OLLAMA_API_BASE env var).")
parser.add_argument("--google-model", dest="google_model",
                    help="Override default Google model ID (sets GOOGLE_DEFAULT_MODEL env var).")
parser.add_argument("--openai-model", dest="openai_model",
                    help="Override default OpenAI model ID (sets OPENAI_DEFAULT_MODEL env var).")
parser.add_argument("--ollama-model", dest="ollama_model",
                    help="Override default Ollama model ID (sets OLLAMA_DEFAULT_MODEL env var).")
args, _unknown = parser.parse_known_args()
if args.provider:
    os.environ["PRIMARY_PROVIDER"] = args.provider
if args.api_key:
    # Assign API key to the appropriate env var
    prov = os.environ.get("PRIMARY_PROVIDER", "google").lower()
    if prov == "google":
        os.environ["GOOGLE_API_KEY"] = args.api_key
    elif prov == "openai":
        os.environ["OPENAI_API_KEY"] = args.api_key
    # Ollama does not use API key
if args.openai_api_base:
    os.environ["OPENAI_API_BASE"] = args.openai_api_base
if args.ollama_api_base:
    os.environ["OLLAMA_API_BASE"] = args.ollama_api_base
if args.google_model:
    os.environ["GOOGLE_DEFAULT_MODEL"] = args.google_model
    os.environ["GOOGLE_IMAGE_ANALYSIS_MODEL"] = args.google_model
if args.openai_model:
    os.environ["OPENAI_DEFAULT_MODEL"] = args.openai_model
    os.environ["OPENAI_IMAGE_ANALYSIS_MODEL"] = args.openai_model
if args.ollama_model:
    os.environ["OLLAMA_DEFAULT_MODEL"] = args.ollama_model

import asyncio
import os
import shutil
import logging
import time
import uuid # Added for session IDs
import json # Added for state inspection

# --- ADK Core ---
from google.adk.tools.agent_tool import AgentTool  # Correct import for AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService # Or GcsArtifactService
from google.genai import types as genai_types

# --- Project Imports ---
# Configuration and Logging
import config # This also initializes logging via config.setup_logging()
from config import agent_flow_logger, tool_calls_logger, WORKSPACE_DIR, LOG_DIR

# Placeholders (Image Analysis uses placeholder function)
import placeholders

# Core Tools
from core_tools import (
    code_execution_tool,
    logging_tool,
    human_approval_tool, # Added HITL tool
    save_plot_artifact # Helper function, not a tool itself
)

# Agents & AgentTools
# Agents are instantiated in their modules now
from agents import (
    ml_orchestrator_agent,
    code_generator_agent,
    image_analysis_agent,
    data_loading_agent,
    preprocessing_agent,
    training_agent,
    evaluation_agent,
    reporting_agent,
    code_generator_tool, # AgentTool instance from code_generator.py
    image_analysis_tool  # AgentTool instance from image_analyzer.py
)

# Callbacks
from callbacks import (
    log_before_agent,
    log_after_agent,
    log_before_tool,
    log_after_tool,
)

# --- Main Application Setup ---

async def main():
    """Sets up and runs the ML Copilot ADK agent workflow."""
    agent_flow_logger.info("--- Starting ML Copilot ADK Application ---")

    # --- 1. Instantiate Services ---
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService() # Use GcsArtifactService for persistence
    agent_flow_logger.info("Services initialized (InMemorySessionService, InMemoryArtifactService).")

    # --- 2. Instantiate Core Tools ---
    # FunctionTools are instantiated directly when defined with @FunctionTool
    # We just need the references imported from core_tools
    core_function_tools = [code_execution_tool, logging_tool, human_approval_tool]
    agent_flow_logger.info(f"Core FunctionTools available: {[t.name for t in core_function_tools]}")
    if not config.ALLOW_INSECURE_CODE_EXECUTION:
         agent_flow_logger.warning("Code execution tool is disabled by config.")
         # Remove code_execution_tool if disabled, agents need to handle its absence
         core_function_tools = [t for t in core_function_tools if t.name != "code_execution_tool"]


    # --- 3. Verify Specialized AgentTools ---
    # AgentTools are created in their respective agent modules
    specialized_agent_tools = []
    if code_generator_tool:
        specialized_agent_tools.append(code_generator_tool)
        agent_flow_logger.info(f"AgentTool verified for: {code_generator_agent.name}")
    else:
        agent_flow_logger.error("CodeGeneratorAgent tool wrapper failed to create.")
        # This is critical, likely stop execution
        print("CRITICAL ERROR: CodeGeneratorAgent tool failed to initialize.")
        return

    if image_analysis_tool:
        specialized_agent_tools.append(image_analysis_tool)
        agent_flow_logger.info(f"AgentTool verified for: {image_analysis_agent.name}")
    else:
        # This might be less critical depending on workflow
        agent_flow_logger.warning("ImageAnalysisAgent tool wrapper failed to create.")


    # --- 4. Instantiate Task-Specific Agents & Prepare Orchestrator Tools ---
    task_agents = [
        data_loading_agent,
        preprocessing_agent,
        training_agent,
        evaluation_agent,
        reporting_agent,
    ]
    orchestrator_agent_tools = [] # Tools specifically for the orchestrator

    for agent in task_agents:
        if agent: # Check if agent was instantiated successfully in its module
            try:
                # --- Assign Tools to Task Agents ---
                # Define tools needed by THIS task agent
                current_agent_tools = [logging_tool] # All agents need logging
                if code_generator_tool:
                    current_agent_tools.append(code_generator_tool)
                if code_execution_tool in core_function_tools: # Only add if enabled
                     current_agent_tools.append(code_execution_tool)

                # Add image analysis tool only to agents that might need it
                if image_analysis_tool and agent.name in ["PreprocessingAgent", "EvaluationAgent", "ReportingAgent"]:
                    current_agent_tools.append(image_analysis_tool)

                # Re-assign tools list and map to the agent instance
                agent.tools = current_agent_tools
                agent.tools_map = {t.name: t for t in current_agent_tools}
                agent_flow_logger.debug(f"Assigned tools to {agent.name}: {[t.name for t in current_agent_tools]}")

                # --- Wrap the task agent as an AgentTool for the Orchestrator ---
                agent_tool_wrapper = AgentTool(
                    agent=agent,
                    # Use description from agent definition (ensure it's set properly there)
                    description=getattr(agent, 'description', f"Tool wrapper for {agent.name}")
                )
                orchestrator_agent_tools.append(agent_tool_wrapper)
                agent_flow_logger.info(f"AgentTool wrapper created for Orchestrator: {agent.name} -> {agent_tool_wrapper.name}")

            except Exception as e:
                agent_flow_logger.error(f"Failed to configure or wrap agent {agent.name}: {e}", exc_info=True)
                print(f"CRITICAL ERROR: Failed to set up agent {agent.name}.")
                return # Stop if setup fails
        else:
            agent_flow_logger.error(f"A task agent instance is None, skipping tool wrapping.")
            print("CRITICAL ERROR: A required task agent failed to initialize.")
            return # Stop if agent is missing

    # Add core tools needed directly by the orchestrator
    orchestrator_agent_tools.append(logging_tool)
    if human_approval_tool in core_function_tools:
        orchestrator_agent_tools.append(human_approval_tool)

    # --- 5. Configure Orchestrator Agent ---
    if not ml_orchestrator_agent:
        agent_flow_logger.critical("MLOrchestratorAgent instance not found. Cannot proceed.")
        print("CRITICAL ERROR: MLOrchestratorAgent failed to initialize.")
        return

    # Assign the collected tools and callbacks to the Orchestrator
    ml_orchestrator_agent.tools = orchestrator_agent_tools
    ml_orchestrator_agent.tools_map = {t.name: t for t in orchestrator_agent_tools}
    # Add logging callbacks
    ml_orchestrator_agent.before_agent_callback = log_before_agent
    ml_orchestrator_agent.after_agent_callback = log_after_agent
    ml_orchestrator_agent.before_tool_callback = log_before_tool
    ml_orchestrator_agent.after_tool_callback = log_after_tool

    # Re-initialize the internal tool context helper if the ADK version requires it after modifying tools/callbacks
    # This might not be necessary in recent versions but good practice if encountering issues.
    if hasattr(ml_orchestrator_agent, '_init_tool_context_helper'):
        try:
            ml_orchestrator_agent._init_tool_context_helper()
        except Exception as e:
             agent_flow_logger.warning(f"Could not re-initialize tool context helper for {ml_orchestrator_agent.name}: {e}")

    agent_flow_logger.info(f"Tools and Callbacks assigned to {ml_orchestrator_agent.name}: {[t.name for t in ml_orchestrator_agent.tools]}")


    # --- 6. Instantiate Runner ---
    try:
        runner = Runner(
            agent=ml_orchestrator_agent, # Start with the orchestrator
            app_name="ML_Copilot_ADK_App",
            session_service=session_service,
            artifact_service=artifact_service,
            # memory_service=... # Add if using MemoryService
        )
        agent_flow_logger.info(f"Runner initialized with root agent: {ml_orchestrator_agent.name}")
    except Exception as e:
        agent_flow_logger.critical(f"Failed to initialize Runner: {e}", exc_info=True)
        print(f"CRITICAL ERROR: Failed to initialize ADK Runner: {e}")
        return

    # --- 7. Session Setup ---
    user_id = "ml_user_main"
    # Create a unique session ID for each run
    session_id = f"session_{uuid.uuid4()}"
    app_name = runner.app_name

    try:
        # Session state is initialized by the Orchestrator in its first run
        session = session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state={} # Start with empty state
        )
        agent_flow_logger.info(f"Session created: {session_id} for user {user_id}")
        print(f"\n--- Session {session_id} Created ---")
    except Exception as e:
        agent_flow_logger.critical(f"Error creating session: {e}", exc_info=True)
        print(f"CRITICAL ERROR: Failed to create session: {e}")
        return

    # --- 8. Initial User Query (Simulated) ---
    # Example query triggering a multi-step process
    # Modify this query to test different scenarios
    initial_query = """
    Please analyze the dataset './my_data.csv'.
    My goal is classification.
    Handle missing values using median imputation and use standard scaling for preprocessing. Generate plots after preprocessing.
    Train both a Logistic Regression (with C=0.5) and a RandomForestClassifier (n_estimators=50).
    Evaluate both models using accuracy and F1 score.
    Generate a confusion matrix plot for each model.
    Finally, give me a report summarizing the process.
    """
    # Example: Preprocessing only
    # initial_query = "Load './my_data.csv', preprocess it using mean imputation and standard scaling, and show me some plots."
    # Example: Vague request
    # initial_query = "Analyze './my_data.csv' and build a model."

    print(f"\n>>> Simulating User Query:\n{initial_query}\n")
    agent_flow_logger.info(f"SESSION={session_id}: Initial query received.")

    # Prepare initial message content
    content = genai_types.Content(role='user', parts=[genai_types.Part(text=initial_query)])

    # --- 9. Run the Orchestrator Workflow ---
    print("--- Starting ML Copilot Workflow Execution ---")
    agent_flow_logger.info(f"SESSION={session_id}: Invoking runner.run_async...")
    final_report = "Workflow did not produce a final report."

    start_time = time.time()
    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            # Optionally print intermediate user-facing messages from the orchestrator
            if event.author == ml_orchestrator_agent.name and event.content and not event.is_final_response():
                if event.content.parts and event.content.parts[0].text:
                    print(f"--- Orchestrator Update: {event.content.parts[0].text} ---")

            # Capture the final output from the orchestrator
            if event.is_final_response() and event.author == ml_orchestrator_agent.name:
                if event.content and event.content.parts and event.content.parts[0].text:
                    final_report = event.content.parts[0].text
                elif event.error_message:
                    final_report = f"Workflow ended with error: {event.error_message}"
                else:
                    final_report = "Workflow finished." # Agent might finish without text

                # Log final event details
                agent_flow_logger.info(f"SESSION={session_id}: Orchestrator yielded final event. Status: {'Error' if event.error_message else 'Success'}. Content snippet: {final_report[:200]}...")
                # No break here, let the runner finish naturally

    except Exception as e:
        final_report = f"Workflow failed with unhandled exception: {e}"
        agent_flow_logger.exception(f"SESSION={session_id}: Unhandled error during runner execution:") # Log traceback
        print(f"\n--- !!! WORKFLOW FAILED !!! ---")
        print(f"Error: {e}")
    finally:
        end_time = time.time()
        duration = end_time - start_time
        agent_flow_logger.info(f"SESSION={session_id}: Runner execution finished. Duration: {duration:.2f} seconds.")
        print("\n--- ML Copilot Workflow Execution Finished ---")
        print(f"Duration: {duration:.2f} seconds")
        print(f"\n>>> Final Orchestrator Output / Report:\n\n{final_report}\n")

        # --- 10. Inspect Final State & Artifacts ---
        try:
            final_session = session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
            if final_session:
                print("\n--- Final Session State ---")
                # Pretty print the state dictionary
                # Use default=str for non-serializable items like numpy arrays if they sneak in
                print(json.dumps(final_session.state, indent=2, default=str))
                print("--------------------------")
                # List final artifacts for this session
                final_artifacts = artifact_service.list_artifact_keys(app_name=app_name, user_id=user_id, session_id=session_id)
                print(f"\nFinal Artifacts ({len(final_artifacts)}): {final_artifacts}")
                agent_flow_logger.info(f"SESSION={session_id}: Final artifacts list: {final_artifacts}")

            else:
                print("Could not retrieve final session state.")
                agent_flow_logger.error(f"SESSION={session_id}: Failed to retrieve final session state.")
        except Exception as e:
            agent_flow_logger.error(f"SESSION={session_id}: Error retrieving final session state/artifacts: {e}", exc_info=True)
            print(f"Error retrieving final session state/artifacts: {e}")


# --- Script Entry Point ---
if __name__ == "__main__":
    # --- Pre-run Setup (for testing) ---
    # 1. Create dummy data file if it doesn't exist
    dummy_data_path = "./my_data.csv"
    if not os.path.exists(dummy_data_path):
        print(f"Creating dummy data file: {dummy_data_path}")
        try:
            with open(dummy_data_path, "w", encoding="utf-8") as f:
                f.write("feature1,feature2,feature3,target\n")
                f.write("1.0,2.5,A,0\n")
                f.write("2.1,,B,1\n") # Missing value
                f.write("3.5,4.1,A,0\n")
                f.write("4.0,5.5,C,1\n")
                f.write("1.2,2.8,B,0\n")
                f.write("NA,3.0,C,1\n") # Missing value as NA
                f.write("0.5,1.1,A,0\n")
                f.write("1.8,1.9,B,1\n")
                f.write("5.0,5.0,C,1\n")
                f.write("2.9,3.9,A,0\n")
        except Exception as e:
            print(f"ERROR: Failed to create dummy data file: {e}")
            # Optionally exit if data is critical
            # exit(1)

    # 2. Clean up previous workspace (optional, good for reruns)
    if os.path.exists(WORKSPACE_DIR):
        print(f"Cleaning up previous workspace: {WORKSPACE_DIR}")
        try:
            shutil.rmtree(WORKSPACE_DIR)
        except Exception as e:
            print(f"Warning: Could not remove workspace directory {WORKSPACE_DIR}: {e}")
    # Recreate workspace directory
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    # 3. Ensure log directory exists (config.py should handle this, but double-check)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Run the main async function ---
    print("\nStarting ML Copilot ADK main execution...")
    try:
        asyncio.run(main())
    except Exception as e:
        agent_flow_logger.critical(f"Unhandled exception in main asyncio run: {e}", exc_info=True)
        print(f"\n--- !!! APPLICATION CRASHED !!! ---")
        print(f"Error: {e}")
    finally:
        print("\nML Copilot ADK execution finished.")

