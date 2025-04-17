 # ML Copilot ADK

 An agent‑based system using the Google Agent Development Kit (ADK) to automate end‑to‑end machine learning workflows, from data loading and preprocessing through model training, evaluation, image analysis, and reporting.

 ## Features

 - **Agent‑Orchestrator Architecture** powered by ADK  
 - **Dynamic Code Generation** via LLM (`CodeGeneratorAgent`)  
 - **Secure (Toggleable) Code Execution** with `code_execution_tool`  
 - **In‑Memory State Management**, artifact storage, and session tracking  
 - **Image Analysis Agent** (placeholder for multimodal LLM)  
 - **Modular Task Agents**: DataLoading, Preprocessing, Training, Evaluation, ImageAnalysis, Reporting  
 - **Human‑in‑the‑Loop** and **Structured Logging** support  

 ## Prerequisites

 - Python 3.9+ (3.10+ recommended)  
 - Git  
 - (Optional) Docker or other sandbox if you later lock down code execution  
 - pip dependencies (see `requirements.txt`)

 ## Setup

 1. **Clone & enter the repo**  
    ```bash
    git clone <your‑repo‑url>
    cd <your‑repo‑dir>
    ```
 2. **(Optional) Create a virtual environment**  
    ```bash
    python -m venv venv
    source venv/bin/activate     # Windows: venv\Scripts\activate
    ```
 3. **Install Python dependencies**  
    ```bash
    pip install -r requirements.txt
    ```
 4. **Configure credentials & models**  
    You can either set environment variables, or pass them as CLI flags to `main.py` (see next section):
    - `PRIMARY_PROVIDER` = `google` | `openai` | `ollama` (default: `google`)  
    - `GOOGLE_API_KEY`, `OPENAI_API_KEY`  
    - `OPENAI_API_BASE`, `OLLAMA_API_BASE` (for custom endpoints)  
    - `GOOGLE_DEFAULT_MODEL`, `OPENAI_DEFAULT_MODEL`, `OLLAMA_DEFAULT_MODEL`  
    - `GOOGLE_IMAGE_ANALYSIS_MODEL` (for the multimodal agent)  
    - Toggle insecure execution in `config.py` via `ALLOW_INSECURE_CODE_EXECUTION`

 ## Running

 ### 1) Quick start (defaults to `PRIMARY_PROVIDER` from your env/config)

 ```bash
 python main.py
 ```

 On first run, if `./my_data.csv` does not exist, the script will generate a small dummy file for you.

 ### 2) Override via CLI flags

 ```bash
 # OpenAI with a specific model
 python main.py \
   --provider openai \
   --api-key "$OPENAI_API_KEY" \
   --openai-model "gpt-4"

 # Ollama on localhost with a named model
 python main.py \
   --provider ollama \
   --ollama-api-base http://localhost:11434 \
   --ollama-model "llama3.2:3b-instruct-q8_0"
 ```

 ### 3) Or set everything via environment

 ```bash
 export PRIMARY_PROVIDER=google
 export GOOGLE_API_KEY="your-google-api-key"
 export GOOGLE_DEFAULT_MODEL="models/text-bison-001"
 export GOOGLE_IMAGE_ANALYSIS_MODEL="models/gemini-image-alpha-001"
 python main.py
 ```

 ## What to Expect

 - **Orchestrator Updates** prefixed with

   ```
   --- Orchestrator Update: <status message>
   ```

 - A **final report** printed when the workflow completes  
 - A dump of the **session state** (all the tracked keys)  
 - A list of **artifact keys** (plots, CSVs, models) at the end  

 ## Example Prompt

 > “Please analyze the dataset `./my_data.csv`. My goal is classification. Handle missing values using median imputation and apply standard scaling. Generate some preprocessing plots. Train both a Logistic Regression (C=0.5) and a RandomForestClassifier (n_estimators=50). Evaluate using accuracy and F1. Produce confusion matrix plots for each. Finally, give me a report summarizing everything.”

 ## Next Steps

 - Swap in a sandboxed code executor (Docker/Jail)  
 - Hook up a real multimodal LLM for `ImageAnalysisAgent`  
 - Add automated tests, linting, and CI for quality and safety  

 ---

 _With this in place, you can get the end‑to‑end agent “bot” up and running and start iterating on the code‑executor or model‑choices next._