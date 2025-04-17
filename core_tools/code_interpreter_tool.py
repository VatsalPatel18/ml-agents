import logging
try:
    from llama_index.core.tools import CodeInterpreterToolSpec
    # Instantiate the Code Interpreter tool (first tool in the spec list)
    _spec = CodeInterpreterToolSpec()
    code_interpreter_tool = _spec.to_tool_list()[0]
    print("--- code_interpreter_tool defined ---")
except ImportError:
    code_interpreter_tool = None
    logging.getLogger("ToolCalls").warning(
        "CodeInterpreterToolSpec not available; code_interpreter_tool disabled."
    )