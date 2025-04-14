# main.py
import asyncio
import os
import shutil
import logging
import time
from typing import List, Dict, Optional

# --- ADK Core ---
from google.adk.agents import AgentTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types as genai_types

# --- Project Imports ---
# Configuration and Logging
import config # This also initializes logging via config.setup_logging()
from config import agent_flow_logger, tool_calls_logger

# Placeholders (MUST BE REPLACED WITH SECURE/REAL IMPLEMENTATIONS)
import placeholders

# Core Tools
from core_tools import (
    code_execution_tool,
    logging_tool,
    save_plot_artifact # Helper function, not a tool itself
)

# Agents & AgentTools
from agents import (
    ml_orchestrator_agent,
    code_generator_agent,
    image_analysis_agent,
    data_loading_agent,
    preprocessing_agent,
    training_agent,
    evaluation_agent,
    reporting_agent,
    code_generator_tool, # AgentTool instance
    image_analysis_tool  # AgentTool instance
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
    core_function_tools = [code_execution_tool, logging_tool]
    agent_flow_logger.info(f"Core FunctionTools instantiated: {[t.name for t in core_function_tools]}")

    # --- 3. Instantiate Specialized Agents & Wrap as Tools ---
    # Agents are already instantiated in their respective modules
    # We need the AgentTool wrappers
    specialized_agent_tools = []
    if code_generator_tool:
        specialized_agent_tools.append(code_generator_tool)
        agent_flow_logger.info(f"AgentTool wrapper created for: {code_generator_agent.name}")
    else:
         agent_flow_logger.error("CodeGeneratorAgent tool wrapper failed to create.")
         # Decide how to handle this - maybe exit? For now, continue.

    if image_analysis_tool:
        specialized_agent_tools.append(image_analysis_tool)
        agent_flow_logger.info(f"AgentTool wrapper created for: {image_analysis_agent.name}")
    else:
         agent_flow_logger.warning("ImageAnalysisAgent tool wrapper failed to create.")


    # --- 4. Instantiate Task-Specific Agents & Wrap as Tools ---
    task_agents = [
        data_loading_agent,
        preprocessing_agent,
        training_agent,
        evaluation_agent,
        reporting_agent,
    ]
    task_agent_tools_map = {} # To pass to Orchestrator
    orchestrator_agent_tools = [] # Tools for the orchestrator

    for agent in task_agents:
        if agent: # Check if agent was instantiated successfully
            try:
                # Define tools needed by THIS task agent
                # All task agents need code gen, code exec, logging. Some need image analysis.
                current_agent_tools = [code_execution_tool, logging_tool]
                if code_generator_tool:
                    current_agent_tools.append(code_generator_tool)
                if image_analysis_tool and agent.name in ["PreprocessingAgent", "EvaluationAgent", "ReportingAgent"]: # Only agents needing image analysis
                     current_agent_tools.append(image_analysis_tool)

                # Re-assign tools list to the agent instance
                agent.tools = current_agent_tools
                agent.tools_map = {t.name: t for t in current_agent_tools} # Rebuild tools_map

                # Wrap the task agent as an AgentTool for the Orchestrator
                agent_tool_wrapper = AgentTool(
                    agent=agent,
                    description=agent.description # Use description from agent definition
                )
                task_agent_tools_map[agent.name] = agent_tool_wrapper
                orchestrator_agent_tools.append(agent_tool_wrapper)
                agent_flow_logger.info(f"AgentTool wrapper created for: {agent.name}")

            except Exception as e:
                agent_flow_logger.error(f"Failed to wrap agent {agent.name if agent else 'Unknown'} as AgentTool: {e}")
        else:
             agent_flow_logger.error(f"A task agent instance is None, skipping tool wrapping.")

    # Add logging tool for the orchestrator itself
    orchestrator_agent_tools.append(logging_tool)

    # --- 5. Instantiate Orchestrator Agent ---
    # Ensure the orchestrator instance exists
    if 'ml_orchestrator_agent' not in globals() or not ml_orchestrator_agent:
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
    # Re-initialize the internal tool context helper if needed (depends on ADK version)
    if hasattr(ml_orchestrator_agent, '_init_tool_context_helper'):
         ml_orchestrator_agent._init_tool_context_helper()
    agent_flow_logger.info(f"Tools and Callbacks assigned to {ml_orchestrator_agent.name}.")


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
        agent_flow_logger.critical(f"Failed to initialize Runner: {e}")
        print(f"CRITICAL ERROR: Failed to initialize ADK Runner: {e}")
        return

    # --- 7. Session Setup ---
    user_id = "ml_user_main"
    # Create a unique session ID for each run
    session_id = f"session_{uuid.uuid4()}"
    app_name = runner.app_name

    try:
        session = session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state={} # Start with empty state, Orchestrator will populate
        )
        agent_flow_logger.info(f"Session created: {session_id} for user {user_id}")
        print(f"\n--- Session {session_id} Created ---")
    except Exception as e:
        agent_flow_logger.critical(f"Error creating session: {e}")
        print(f"CRITICAL ERROR: Failed to create session: {e}")
        return

    # --- 8. Initial User Query (Simulated) ---
    # Example query triggering a multi-step process
    initial_query = """
    Please analyze the dataset './my_data.csv'.
    My goal is classification.
    Handle missing values using median imputation and use standard scaling for preprocessing.
    Train both a Logistic Regression (with C=0.5) and a RandomForestClassifier (n_estimators=50).
    Evaluate both models using accuracy and F1 score.
    Generate a confusion matrix plot for each model.
    Finally, give me a report summarizing the process.
    """
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
            # Optionally print intermediate user-facing messages
            if event.author == ml_orchestrator_agent.name and event.content and not event.is_final_response():
                 if event.content.parts and event.content.parts[0].text:
                      print(f"--- Orchestrator Update: {event.content.parts[0].text} ---")

            # Capture the final output from the orchestrator
            if event.is_final_response():
                if event.content and event.content.parts and event.content.parts[0].text:
                    final_report = event.content.parts[0].text
                elif event.error_message:
                     final_report = f"Workflow ended with error: {event.error_message}"
                else:
                     final_report = "Workflow finished." # Agent might finish without text

                # Log final event details
                agent_flow_logger.info(f"SESSION={session_id}: Orchestrator yielded final event. Content snippet: {final_report[:200]}...")
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
                print(json.dumps(final_session.state, indent=2, default=str)) # Use default=str for non-serializable items
                print("--------------------------")
                # List final artifacts for this session
                final_artifacts = artifact_service.list_artifact_keys(app_name=app_name, user_id=user_id, session_id=session_id)
                print(f"\nFinal Artifacts ({len(final_artifacts)}): {final_artifacts}")
                agent_flow_logger.info(f"SESSION={session_id}: Final artifacts list: {final_artifacts}")

            else:
                print("Could not retrieve final session state.")
                agent_flow_logger.error(f"SESSION={session_id}: Failed to retrieve final session state.")
        except Exception as e:
            agent_flow_logger.error(f"SESSION={session_id}: Error retrieving final session state/artifacts: {e}")
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
    if os.path.exists(config.WORKSPACE_DIR):
      print(f"Cleaning up previous workspace: {config.WORKSPACE_DIR}")
      try:
          shutil.rmtree(config.WORKSPACE_DIR)
      except Exception as e:
          print(f"Warning: Could not remove workspace directory {config.WORKSPACE_DIR}: {e}")
    # Recreate workspace directory
    os.makedirs(config.WORKSPACE_DIR, exist_ok=True)

    # 3. Ensure log directory exists (config.py should handle this, but double-check)
    os.makedirs(config.LOG_DIR, exist_ok=True)

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

