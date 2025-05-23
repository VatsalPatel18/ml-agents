# agents/code_generator.py
from google.adk.agents import LlmAgent
# Corrected import: Import the 'agent_tool' module
from google.adk.tools import agent_tool
from config import CODE_GEN_MODEL, USE_LITELLM

# Import LiteLLM wrapper if needed
if USE_LITELLM:
    try:
        from google.adk.models.lite_llm import LiteLlm
        print("LiteLLM imported successfully for CodeGenerator.")
    except ImportError:
        print("ERROR: LiteLLM specified in config, but 'litellm' package not found. pip install litellm")
        LiteLlm = None
else:
    LiteLlm = None # Define as None if not used

# --- Code Generator Agent Definition ---
# Determine model configuration
model_config = LiteLlm(model=CODE_GEN_MODEL) if USE_LITELLM and LiteLlm else CODE_GEN_MODEL

code_generator_agent = LlmAgent(
    name="CodeGeneratorAgent",
    model=model_config, # Use configured model
    instruction="""
You are a specialized AI assistant that ONLY generates Python code for Machine Learning tasks.
You will receive detailed prompts specifying the task, context (like data paths, column names, data types), required libraries (pandas, scikit-learn, matplotlib, seaborn, joblib), and desired output (e.g., save a file, print metrics, print file paths).
- Generate ONLY the Python code requested. Do not add explanations before or after the code block unless explicitly asked.
- Ensure the code includes necessary imports (including `import os` and `import json` if needed for paths/metrics).
- If asked to save files (data, models, plots), use the provided file paths in the prompt. Ensure the paths are used correctly within the code (e.g., `df.to_csv('/path/to/output.csv')`). Create directories if needed using `os.makedirs(os.path.dirname(filepath), exist_ok=True)`.
- If the code needs to output information for subsequent steps (like metrics or file paths), ensure it PRINTS this information clearly to standard output using the specific convention:
    - For saved files: `print(f"SAVED_OUTPUT: logical_name=/path/to/actual/file.ext")` (replace logical_name and path).
    - For metrics dictionary: `print(f"METRICS: {json.dumps(metrics_dict)}")` (ensure metrics_dict is JSON serializable).
    - For general info: `print(f"INFO: Your message here")`
- Write clean, efficient, and readable code.
- Assume standard ML libraries are installed in the execution environment.
- Handle potential errors within the generated code where appropriate (e.g., using try-except blocks for file operations or library calls).
- Do NOT execute any code yourself. Your sole output is the Python code string, enclosed in markdown triple backticks if necessary for clarity, but preferably just the raw code.
""",
    description="Generates Python code for ML tasks based on detailed prompts.",
    # No tools needed for this agent itself.
)

# --- Wrap as AgentTool ---
from google.adk.tools import agent_tool

# --- Wrap as AgentTool ---
# Provides other agents a function-like tool to invoke the CodeGeneratorAgent
try:
    # Wrap the CodeGeneratorAgent so it can be invoked as a tool
    code_generator_tool = agent_tool.AgentTool(
        agent=code_generator_agent
    )
    print(f"--- CodeGeneratorAgent wrapped as tool: {code_generator_tool.name} ---")
except Exception as e:
    print(f"ERROR: Could not create code_generator_tool: {e}")
    code_generator_tool = None

print(f"--- CodeGeneratorAgent Defined (Model: {code_generator_agent.model}) ---")

