# Machine Learning Agent Swarm

## Overview

This project implements a sophisticated, agent-based system designed to automate various machine learning workflows. It utilizes a swarm of specialized agents, coordinated by a central orchestrator, to handle tasks ranging from data loading and preprocessing to model training, evaluation, and reporting. The system dynamically generates and executes necessary code, making it flexible and adaptable to different ML goals.

## Features

*   **Agent-Based Architecture:** Modular design with specialized agents for distinct ML tasks.
*   **Dynamic Workflow Orchestration:** An `MLOrchestratorAgent` interprets user goals and plans the execution sequence.
*   **Automated Code Generation:** A dedicated `CodeGeneratorAgent` writes Python code for ML operations (using libraries like Pandas, Scikit-learn, etc.).
*   **Code Execution & Error Handling:** Executes generated code securely and manages potential errors.
*   **State Management:** Tracks datasets, models, metrics, and workflow progress.
*   **Artifact Handling:** Manages outputs like plots, reports, and intermediate data.
*   **Task Specialization:** Agents dedicated to Data Loading, Preprocessing, Training, Evaluation, Visualization, Image Analysis, and Reporting.

## Architecture

The system follows a hierarchical agent structure:

1.  **MLOrchestratorAgent:** The top-level agent that interacts with the user, understands the overall goal, plans the workflow dynamically, and delegates tasks to specialized agents.
2.  **CodeGeneratorAgent:** A specialized LLM agent responsible solely for generating Python code based on detailed prompts from other agents.
3.  **Task-Specific Agents:**
    *   `DataLoadingAgent`: Handles loading datasets.
    *   `PreprocessingAgent`: Performs data cleaning, transformation, and feature engineering.
    *   `TrainingAgent`: Trains specified machine learning models.
    *   `EvaluationAgent`: Evaluates trained models using various metrics.
    *   `ImageAnalysisAgent`: Interprets image artifacts (like plots) using multimodal capabilities.
    *   `ReportingAgent`: Synthesizes results and generates final reports.
4.  **Core Tools:**
    *   `CodeExecutionTool`: Executes the generated Python code in a controlled environment.
    *   `VisualizationTool`: Generates plots and saves them as artifacts.
    *   `LoggingTool`: Provides structured logging capabilities.

*(Note: This architecture likely relies on the Google Agent Development Kit - ADK)*

## Getting Started

### Prerequisites

*   Python (specify version, e.g., 3.10+)
*   Required Python packages (see `requirements.txt`)
*   (Potentially) Access keys or configuration for specific LLMs or cloud services used by the agents.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configuration:**
    *   Update `config.py` or relevant configuration files with necessary API keys, paths, or settings. (Add more specific instructions here if needed).

### Running the System

Execute the main entry point script:

```bash
python main.py
```

Follow the prompts from the `MLOrchestratorAgent`.

## Usage Examples

Interact with the orchestrator agent by providing high-level goals:

*   `"Load data from 'path/to/data.csv' and preprocess it."`
*   `"Build a classification model using the preprocessed data 'dataset_id_1'."`
*   `"Evaluate models 'model_id_A' and 'model_id_B' on dataset 'dataset_id_1'."`
*   `"Generate a report summarizing the experiment."`

*(Add more specific examples based on how users interact with `main.py`)*

## Technologies

*   Python
*   Google Agent Development Kit (ADK) - *Assumed*
*   Pandas, Scikit-learn, Matplotlib, etc. (via Code Generation)

## Contributing

*(Optional: Add guidelines for contributing if this is an open project)*

## License

*(Optional: Specify the license for your project, e.g., MIT, Apache 2.0)*
