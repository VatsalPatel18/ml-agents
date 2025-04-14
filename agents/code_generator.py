# agents/code_generator.py
from google.adk.agents import LlmAgent, AgentTool
from config import CODE_GEN_MODEL

# --- Code Generator Agent Definition ---
code_generator_agent = LlmAgent(
    name="CodeGeneratorAgent",
    model=CODE_GEN_MODEL,
    instruction="""
You are a specialized AI assistant that ONLY generates Python code for Machine Learning tasks.
You will receive detailed prompts specifying the task, context (like data paths, column names, data types), required libraries (pandas, scikit-learn, matplotlib, seaborn), and desired output (e.g., save a file, print metrics, print file paths).
- Generate ONLY the Python code requested. Do not add explanations before or after the code block unless explicitly asked.
- Ensure the code includes necessary imports.
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
# This allows other agents to call the CodeGeneratorAgent easily
try:
    code_generator_tool = AgentTool(
        agent=code_generator_agent,
        description="Use this tool to generate Python code for specific ML tasks like data loading, preprocessing, training, evaluation, or plotting. Provide a detailed prompt describing the exact task, input/output paths/variables from state, libraries, and expected output format/convention (e.g., print 'SAVED_OUTPUT: name=path').",
    )
except Exception as e:
    print(f"ERROR: Failed to wrap CodeGeneratorAgent as AgentTool: {e}")
    code_generator_tool = None

