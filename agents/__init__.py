# agents/__init__.py

# Import agent instances directly
from .orchestrator import ml_orchestrator_agent
from .code_generator import code_generator_agent, code_generator_tool
from .image_analyzer import image_analysis_agent, image_analysis_tool
from .data_loader import data_loading_agent
from .preprocessor import preprocessing_agent
from .trainer import training_agent
from .evaluator import evaluation_agent
from .reporter import reporting_agent

# AgentTool instances (like code_generator_tool, image_analysis_tool)
# are now created in main.py after the agents are instantiated,
# so they are not exported directly from here anymore.

__all__ = [
    "ml_orchestrator_agent",
    "code_generator_agent",
    "code_generator_tool",
    "image_analysis_agent",
    "image_analysis_tool",
    "data_loading_agent",
    "preprocessing_agent",
    "training_agent",
    "evaluation_agent",
    "reporting_agent",
]

print("--- Agents Imported ---")
