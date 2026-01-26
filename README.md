# Evidence level grader using LLMs and Machine Learning

## Project structure
Here is a brief overview of the project structure:

- config/: used to control input and output directories, hyperparameters, dataset split ratios, base models for embeddings and LLM models
- data/: used to store raw and preprocessed CIViC data (.csv)
- src/: scripts for data spliting and preprocessing, machine learning models training, i.e., xgboost (xgb) and decision tree (dt), and LLMs (gemini, gpt) inference for classification of evidence level
    - src/utils/prompts.py: stores prompts wrappers for one- and few-shot inferences. 
- runs/: used to store output data from src/ (e.g., confusino matrix, performance metrics, trained models, predictions)

## Setup and Installation

### Option 1: Dev Container (Recommended)
This project uses a Docker dev container with GPU support. 

**Prerequisites:**
- Docker with NVIDIA GPU support
- VS Code with Dev Containers extension

**Setup:**
```bash
git clone https://github.com/UKEIAM/evidence-level-ml-llm
cd evidence-level-ml-llm
```
Then open in VS Code and select "Reopen in Container" when prompted. Dependencies will be installed automatically.

### Option 2: Local Environment
```bash
git clone https://github.com/UKEIAM/evidence-level-ml-llm
cd evidence-level-ml-llm
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (order matters for compatibility)
pip install 'numpy>=1.22,<1.25'
pip install pandas>=2.3.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Note:** The project requires PyTorch with CUDA 12.1 support. For CPU-only or different CUDA versions, modify the torch installation URL accordingly.

### Define API keys
If you plan to run LLM inferences, create a .env file with your OpenAI and Gemini API keys following this format:

``` 
OPENAI_API_KEY='yourkey'
GEMINI_API_KEY='yourkey'
```

### Usage
1. First, perform the data preprocessing pipeline: 
```
python src/run_data_preprocessing.py
```
To modify input and output directories, change config/data_config.yaml accordingly. config/splits_config.py can be modified for changing split ratios, output directories and seed.

2. To train all ML models (xgb, dt) with both feature representations (TF-IDF, embeddings), and store performance reports run: 
```
python src/run_ml_models_training.py
```
To modify hyperparameters, base models and directories, change runs_config.yaml accordingly 
3. To infer LLMs (GPT, Gemini) and store performance reports, run:
```
python src/llm/gpt.py
``` 
or
```
python src/llm/gemini.py
```
